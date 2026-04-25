import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM
# from thop import profile
# from transformer import 
try:
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
except ImportError:
    from transformers import SegformerImageProcessor as SegformerFeatureExtractor
    from transformers import SegformerForSemanticSegmentation
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
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
# from Mamba_decoder import ChangeDecoder
import torchvision.models as models
import torch
import torch.nn as nn
import timm
import torchvision.models as models
import importlib.util
import sys
import types
# Load MiT-B0 model (SegFormer backbone)


def _load_hoi_interaction_class():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    hoi_dir = os.path.join(root_dir, "HOIF-main")

    package_name = "hoif_main"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [hoi_dir]
        sys.modules[package_name] = package

    for submodule in ("modules", "refine", "highorder"):
        full_name = f"{package_name}.{submodule}"
        if full_name in sys.modules:
            continue
        module_file = os.path.join(hoi_dir, f"{submodule}.py")
        spec = importlib.util.spec_from_file_location(full_name, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load HOI module from: {module_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.highorder"].highOrderInteraction


class HOI_Fusion_Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        high_order_interaction = _load_hoi_interaction_class()
        self.hoi = high_order_interaction(channelin=in_channels, channelout=in_channels)
        self.ho_gain = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        self.pre_norm_t1 = nn.GroupNorm(1, in_channels, eps=1e-6, affine=True)
        self.pre_norm_t2 = nn.GroupNorm(1, in_channels, eps=1e-6, affine=True)

        # 【修改点1】：因为我们要把高阶模块的两个输出 concat 起来
        # 所以进入 1x1 卷积的通道数变成了 2 * in_channels
        self.align = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_T1, feat_T2):
        orig_dtype = feat_T1.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            self.hoi = self.hoi.float()
            feat_T1 = self.pre_norm_t1(feat_T1.float())
            feat_T2 = self.pre_norm_t2(feat_T2.float())
            feat_T1 = F.normalize(feat_T1, p=2.0, dim=1, eps=1e-6)
            feat_T2 = F.normalize(feat_T2, p=2.0, dim=1, eps=1e-6)

            # 【修改点2】：不再丢弃第二个特征，用 feat_t1_fused 和 feat_t2_evolved 全部接收
            feat_t1_fused, feat_t2_evolved = self.hoi(feat_T1, feat_T2, 0, 1)

            # 【修改点3】：将双时相的高阶特征拼接 (B, 2C, H, W)
            hoi_feat = torch.cat([feat_t1_fused, feat_t2_evolved], dim=1)

            hoi_feat = torch.nan_to_num(hoi_feat, nan=0.0, posinf=1e4, neginf=-1e4)
            hoi_feat = hoi_feat * torch.clamp(self.ho_gain, min=0.0, max=1.0)
            hoi_feat = torch.clamp(hoi_feat, min=-60000.0, max=60000.0)

        # 映射回解码器需要的 out_channels
        return self.align(hoi_feat.to(orig_dtype))


class SwinFPN(nn.Module):
    def __init__(self,dim):
        super(SwinFPN, self).__init__()
        # Load pre-trained MiT-B0 from timm
        # print(timm.models.create_model('swin_base_patch4_window7_224').default_cfg)
        # self.backbone = timm.create_model('mit_b0', pretrained=True, features_only=True)
        # 加载预训练的 Segformer 模型 (基于 MIT-B0)
        self.backbone = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
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
resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
# Create the ResNet-FPN model
# fpn_model = ResNetFPN(resnet)

# # # Example input tensor
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size

# # Extract feature pyramids
# features = fpn_model(input_tensor)

# # Print the shapes of the feature maps
# for i, f in enumerate(features):
#     print(f"Feature map {i+1} shape: {f.shape}")
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
        self.decoder = Mamba_Decoder_Pyramid(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            **clean_kwargs
        )

        self.fusion_adapters = nn.ModuleList(
            [HOI_Fusion_Adapter(in_channels=dim, out_channels=2 * dim) for dim in self.encoder.dims]
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
            feature.append(self.fusion_adapters[index](pre_features[index], post_features[index]))
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
    


