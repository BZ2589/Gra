import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from classification.models.vmamba import LayerNorm2d
from changedetection.models.MDP import Mamba_Decoder_Pyramid


class Backbone_Swin(nn.Module):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, target_dims=None, **kwargs):
        super().__init__()
        self.out_indices = out_indices
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            img_size=256
        )
        self.channel_first = True
        swin_channels = [self.backbone.feature_info.channels()[i] for i in out_indices]

        if target_dims is not None:
            self.proj = nn.ModuleList([
                nn.Conv2d(swin_channels[i], target_dims[i], 1)
                for i in range(len(out_indices))
            ])
            self.dims = list(target_dims)
        else:
            self.proj = None
            self.dims = swin_channels
        self.out_indices = list(out_indices)

    def forward(self, x):
        features = self.backbone(x)
        out = []
        for i, idx in enumerate(self.out_indices):
            f = features[idx]
            if self.proj is not None:
                f = f.permute(0, 3, 1, 2).contiguous()
                f = self.proj[i](f)
            out.append(f)
        return out


class SwinPyramid(nn.Module):
    def __init__(self, pretrained=None, **kwargs):
        super(SwinPyramid, self).__init__()
        self.encoder = Backbone_Swin(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            target_dims=[128, 256, 512, 1024],
            **kwargs
        )

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

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = Mamba_Decoder_Pyramid(
            encoder_dims=self.encoder.dims,
            channel_first=False,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_3x3=True,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128 * 2, out_channels=2, kernel_size=1)
        self.ds = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.ds.append(nn.Conv2d(in_channels=128 * 2, out_channels=2, kernel_size=1))

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        feature = []
        for index in range(len(pre_features)):
            pre_f = pre_features[index]
            post_f = post_features[index]
            feature.append(torch.cat([pre_f, post_f], dim=1))
        output, output_ds = self.decoder(feature)
        output = self.main_clf(output)
        for i in range(self.depth - 1):
            output_ds[i] = self.ds[i](output_ds[i])

        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear', align_corners=False)
        for f in range(len(output_ds)):
            output_ds[f] = F.interpolate(output_ds[f], size=pre_data.size()[-2:], mode='bilinear', align_corners=False)
        return output, output_ds
