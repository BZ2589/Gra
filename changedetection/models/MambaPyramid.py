import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM
# from thop import profile
# from transformer import 
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from changedetection.models.MDP import Mamba_Decoder_Pyramid
# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
# from Mamba_decoder import ChangeDecoder
import torchvision.models as models
import torch
import torch.nn as nn
import timm
import torchvision.models as models
# Load MiT-B0 model (SegFormer backbone)
class SwinFPN(nn.Module):
    def __init__(self,dim):
        super(SwinFPN, self).__init__()
        # Load pre-trained MiT-B0 from timm
        # print(timm.models.create_model('swin_base_patch4_window7_224').default_cfg)
        # self.backbone = timm.create_model('mit_b0', pretrained=True, features_only=True)
        # 加载预训练的 Segformer 模型 (基于 MIT-B0)
        self.backbone = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        # self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        # self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True,num_classes=10, features_only=True,global_pool='')
        self.dims=dim
        self.channel_first=False
    def forward(self, x):
        # Extract feature maps at different pyramid levels
        # Feature maps will be at 4 different stages as per the MiT-B0 architecture
        features = self.backbone(x)
        return features  # List of feature maps from 4 stages
class ResNetFPN(nn.Module):
    def __init__(self, backbone,dim):
        super(ResNetFPN,self).__init__()
        self.dims = dim
        self.channel_first = False
        # Extract layers from the pre-trained ResNet model
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
    def forward(self, x):
        # Feature pyramid levels from ResNet
        # Stage 0: output from conv1
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)  # Downsample after first block
        
        # Stage 1: output from layer1
        c2 = self.layer1(c1)
                # Stage 2: output from layer2
        c3 = self.layer2(c2)
        
        # Stage 3: output from layer3
        c4 = self.layer3(c3)
        
        # Stage 4: output from layer4
        c5 = self.layer4(c4)
        # import pdb
        # pdb.set_trace()
        # Return the feature maps (c2 to c5 are usually used for FPN)
        return [ c2, c3, c4, c5]

# Load a pre-trained ResNet model
resnet = models.resnet101(pretrained=True)
# resnet = models.resnet50(pretrained=True)
# resnet = models.resnet18(pretrained=True)
# resnet = models.resnet34(pretrained=True)
# Create the ResNet-FPN model
# fpn_model = ResNetFPN(resnet)

# # # Example input tensor
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size

# # Extract feature pyramids
# features = fpn_model(input_tensor)

# # Print the shapes of the feature maps
# for i, f in enumerate(features):
#     print(f"Feature map {i+1} shape: {f.shape}")
# =====================================================================
# 模块一：高阶张量交互融合模块 (适配单卡运行，防止 OOM)
# =====================================================================
class HighOrderBilinearFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels * 2
        
        # 降维以防止在 4090 上计算协方差时 OOM
        self.mid_channels = min(64, in_channels // 2) 
        self.proj1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False)
        
        # 投影回指定的输出维度，以保持与原 torch.cat 输出维度一致，不改变 decoder 的输入
        self.interaction_proj = nn.Conv2d(self.mid_channels ** 2, self.out_channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(self.out_channels)
        self.act = nn.GELU()
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x1.shape
        p1 = self.proj1(x1)
        p2 = self.proj2(x2)
        
        # 二阶双线性池化计算高阶特征
        covar = torch.einsum('b c h w, b d h w -> b c d h w', p1, p2)
        covar = covar.reshape(B, self.mid_channels ** 2, H, W)
        
        out = self.interaction_proj(covar)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return self.act(out)

# =====================================================================
# 模块二 & 三：协同选择性扫描算子与 Block (适配单卡运行)
# =====================================================================
class CoSelectiveScan2D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=True, kernel_size=d_conv, padding=(d_conv - 1) // 2
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _cross_directional_scan(self, x_main: torch.Tensor, x_guide: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_main.shape
        L = H * W
        xs_main = [
            x_main.flatten(2),
            torch.flip(x_main.flatten(2), dims=[-1]),
            x_main.transpose(2, 3).flatten(2),
            torch.flip(x_main.transpose(2, 3).flatten(2), dims=[-1])
        ]
        xs_guide = [
            x_guide.flatten(2),
            torch.flip(x_guide.flatten(2), dims=[-1]),
            x_guide.transpose(2, 3).flatten(2),
            torch.flip(x_guide.transpose(2, 3).flatten(2), dims=[-1])
        ]
        out_ys = []
        for i in range(4):
            xm = xs_main[i].transpose(1, 2)
            xg = xs_guide[i].transpose(1, 2)
            
            # 交叉协同：让 Guide 的特征映射生成 Main 的时相动态权重参数
            x_proj_out = self.x_proj(xg)
            dt, B_param, C_param = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt))
            
            # 参数级时相协同机制，通过特征交叉调制打破孤岛
            y = xm * dt * F.sigmoid(C_param.mean(dim=-1, keepdim=True))
            out_ys.append(y.transpose(1, 2))
            
        y0 = out_ys[0].view(B, -1, H, W)
        y1 = torch.flip(out_ys[1], dims=[-1]).view(B, -1, H, W)
        y2 = out_ys[2].view(B, -1, W, H).transpose(2, 3)
        y3 = torch.flip(out_ys[3], dims=[-1]).view(B, -1, W, H).transpose(2, 3)
        
        y_fused = y0 + y1 + y2 + y3
        y_fused = y_fused.permute(0, 2, 3, 1)
        y_fused = self.out_norm(y_fused)
        return self.out_proj(y_fused).permute(0, 3, 1, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1_proj = self.in_proj(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2_proj = self.in_proj(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x1_inner, x1_res = x1_proj.chunk(2, dim=1)
        x2_inner, x2_res = x2_proj.chunk(2, dim=1)
        
        x1_inner = self.act(self.conv2d(x1_inner))
        x2_inner = self.act(self.conv2d(x2_inner))
        
        out1 = self._cross_directional_scan(x_main=x1_inner, x_guide=x2_inner)
        out2 = self._cross_directional_scan(x_main=x2_inner, x_guide=x1_inner)
        
        out1 = out1 * self.act(x1_res)
        out2 = out2 * self.act(x2_res)
        return out1, out2

class CoTemporalVSSBlock(nn.Module):
    def __init__(self, hidden_dim: int, drop_path: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = CoSelectiveScan2D(d_model=hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1_ln = x1.permute(0, 2, 3, 1)
        x2_ln = x2.permute(0, 2, 3, 1)
        
        x1_ln = self.ln_1(x1_ln).permute(0, 3, 1, 2)
        x2_ln = self.ln_1(x2_ln).permute(0, 3, 1, 2)
        
        out1, out2 = self.self_attention(x1_ln, x2_ln)
        
        x1 = x1 + self.drop_path(out1)
        x2 = x2 + self.drop_path(out2)
        return x1, x2

class MambaPyramid(nn.Module):
    def __init__(self, pretrained,**kwargs):
        super(MambaPyramid, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        # self.out_ch = out_ch
        # self.encoder=ResNetFPN(resnet,dim=[256,512,1024,2048])
        # self.encoder=ResNetFPN(resnet,dim=[64,128,256,512])
        # self.encoder = SwinFPN(dim=[256,512,1024,2048])
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 
        self.depth = kwargs['decoder_depths']
        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        # 实例化协同选择性扫描 Block 和 高阶张量交互融合模块
        self.co_vss_blocks = nn.ModuleList([
            CoTemporalVSSBlock(hidden_dim=dim) for dim in self.encoder.dims
        ])
        self.fusions = nn.ModuleList([
            HighOrderBilinearFusion(in_channels=dim, out_channels=dim * 2) for dim in self.encoder.dims
        ])

        self.decoder = Mamba_Decoder_Pyramid(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        self.ds = nn.ModuleList([])
        for i in range(self.depth-1):
            self.ds.append(nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1))
        # self.ds1 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        # self.ds2 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
        # self.ds3 = nn.Conv2d(in_channels=128*2, out_channels=2, kernel_size=1)
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        feature = []
        for index in range(len(pre_features)):
            # 参数级时相协同 (Co-Selective Scan)
            pre_f, post_f = self.co_vss_blocks[index](pre_features[index], post_features[index])
            # 高阶张量交互融合 (High-Order Bilinear Pooling)
            fused_feature = self.fusions[index](pre_f, post_f)
            feature.append(fused_feature)
            
        output,output_ds = self.decoder(feature)
        output = self.main_clf(output)
        for i in range(self.depth-1):
            output_ds[i] = self.ds[i](output_ds[i])
        # output_ds[0] = self.ds1(output_ds[0])
        # output_ds[1] = self.ds2(output_ds[1])
        # output_ds[2] = self.ds3(output_ds[2])
      
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear',align_corners=False)
        for f in range(len(output_ds)):
            output_ds[f] = F.interpolate(output_ds[f], size=pre_data.size()[-2:], mode='bilinear',align_corners=False) 
        return output,output_ds
    

