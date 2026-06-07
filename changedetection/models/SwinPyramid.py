import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from changedetection.models.MDP import Mamba_Decoder_Pyramid


class Backbone_Swin(nn.Module):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, **kwargs):
        super().__init__()
        self.out_indices = out_indices
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True
        )
        self.channel_first = True
        self.dims = [self.backbone.feature_info.channels()[i] for i in out_indices]
        self.out_indices = list(out_indices)

    def forward(self, x):
        features = self.backbone(x)
        return [features[i] for i in self.out_indices]


class SwinPyramid(nn.Module):
    def __init__(self, pretrained=None, **kwargs):
        super(SwinPyramid, self).__init__()
        self.encoder = Backbone_Swin(
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            **kwargs
        )

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=nn.LayerNorm2d,
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
            channel_first=self.encoder.channel_first,
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
            feature.append(torch.cat([pre_features[index], post_features[index]], dim=1))
        output, output_ds = self.decoder(feature)
        output = self.main_clf(output)
        for i in range(self.depth - 1):
            output_ds[i] = self.ds[i](output_ds[i])

        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear', align_corners=False)
        for f in range(len(output_ds)):
            output_ds[f] = F.interpolate(output_ds[f], size=pre_data.size()[-2:], mode='bilinear', align_corners=False)
        return output, output_ds
