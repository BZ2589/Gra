import torch.nn.functional as F
from collections import OrderedDict

from math import exp
# from .utils.CDC import cdcconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine, CALayer
import torch.nn.init as init
import os
import cv2
import numbers
from einops import rearrange
import numpy


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class DenseBlockMscale(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier'):
        super(DenseBlockMscale, self).__init__()
        self.ops = DenseBlock(channel_in, channel_out, init)
        self.fusepool = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc1 = nn.Sequential(nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fuse = nn.Conv2d(3*channel_out,channel_out,1,1,0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops(x1)
        x2 = self.ops(x2)
        x3 = self.ops(x3)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        xattw = self.fusepool(x1+x2+x3)
        xattw1 = self.fc1(xattw)
        xattw2 = self.fc2(xattw)
        xattw3 = self.fc3(xattw)
        # x = x1*xattw1+x2*xattw2+x3*xattw3
        x = self.fuse(torch.cat([x1*xattw1,x2*xattw2,x3*xattw3],1))

        return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlockMscale(channel_in, channel_out, init)
            else:
                return DenseBlockMscale(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None
    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: (z, logdet)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out
    
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class spatialInteraction(nn.Module):
    def __init__(self, channelin, channelout, order=4):
        super(spatialInteraction, self).__init__()
        self.order = order
        self.channelin = channelin
        self.channelout = channelout
        hidden = max(channelout // 4, 16)

        self.reduce = nn.Sequential(nn.Conv2d(channelout, hidden, 1), nn.ReLU())
        self.expand = nn.Sequential(nn.Conv2d(hidden, channelout, 1), nn.ReLU())

        self.reflashFused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
            ) for _ in range(order - 1)
        ])

        self.reflashInfrared = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
            ) for _ in range(order - 1)
        ])

        self.conv_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * hidden, hidden, 1),
                nn.ReLU()
            ) for _ in range(order - 1)
        ])

        self.convf = nn.Sequential(
            nn.Conv2d(order * hidden, hidden, 1)
        )

        self.norm_atten = LayerNorm(hidden, LayerNorm_type='WithBias')

        self.norm_fused = nn.ModuleList([
            LayerNorm(hidden, LayerNorm_type='WithBias')
            for _ in range(order - 1)
        ])

        self.norm_out = nn.GroupNorm(1, hidden)
        nn.init.constant_(self.norm_out.weight, 0.01)
        nn.init.constant_(self.norm_out.bias, 0)

    def _reflash(self, x, k):
        x = self.reflashFused[k](x)
        x = self.norm_fused[k](x + self.conv_fuse[k](torch.cat([x, x], dim=1)))
        return x

    def forward(self, vis, inf, i, j):

        _, C, H, W = vis.size()

        vis_r = self.reduce(vis)
        inf_r = self.reduce(inf)

        vis_fft = torch.fft.rfft2(vis_r.float())
        inf_fft = torch.fft.rfft2(inf_r.float())
        atten = vis_fft * inf_fft
        atten = torch.fft.irfft2(atten, s=(H, W))
        atten = self.norm_atten(atten)
        fused = atten * inf_r

        order_features = [fused]
        infrared_reflash = None

        for k in range(self.order - 1):
            reflash_ir = self.reflashInfrared[k]

            if k == 0:
                fused = self._reflash(fused + vis_r, k)
                infrared_reflash = reflash_ir(inf_r)
            else:
                prev_fused = order_features[-1]
                fused = self._reflash(fused + vis_r, k)
                infrared_reflash = reflash_ir(infrared_reflash)

            fused = fused * infrared_reflash
            order_features.append(fused)

        fused_feat = self.convf(torch.cat(order_features, dim=1))
        fused_feat = self.norm_out(fused_feat)
        fused_feat = self.expand(fused_feat)

        vis_out = fused_feat + vis
        inf_out = fused_feat + inf

        return vis_out, inf_out


class OminiInteraction(nn.Module):
    def __init__(self, channelin, channelout):
        super(OminiInteraction, self).__init__()
        self.reflashFused1 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashFused2 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )

        self.reflashInfrared1 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )
        self.reflashInfrared2 = nn.Sequential(
            nn.Conv2d(channelin, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )



        self.convf = nn.Sequential(
            nn.Conv2d(2 * channelout, channelout, 1)
        )

        self.convout = nn.Sequential(
            nn.Conv2d(channelout, channelout, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channelout, channelout, 3, 1, 1)
        )

        self.convatten = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )

        self.norm1 = LayerNorm(channelout, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(channelout, LayerNorm_type='WithBias')

    def forward(self, vis, inf, i, j):
        B, C, H, W = vis.size()

        vis_fft = torch.fft.rfft2(vis.float())
        inf_fft = torch.fft.rfft2(inf.float())

        atten = vis_fft * inf_fft
        atten = torch.fft.irfft2(atten, s=(H, W))
        atten = self.norm1(atten)


        fused_OneOrderSpa = atten * inf

        theta = vis.float().view(B, C, H * W)
        phi = inf.float().view(B, C, H * W).permute(0, 2, 1)
        g = fused_OneOrderSpa.view(B, C, H * W)

        attention = torch.matmul(theta, phi)
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, g)
        out = out.view(B, C, H, W)
        out = self.convout(out)+vis

        fused_OneOrderSpa = self.reflashFused1(atten)
        fused_OneOrderSpa = self.norm2(fused_OneOrderSpa)
        infraredReflash1 = self.reflashInfrared1(out)
        fused_twoOrderSpa = fused_OneOrderSpa * infraredReflash1

        attention1 = attention.unsqueeze(1)
        attention1 = self.convatten(attention1)
        attention1 = torch.softmax(attention1, dim=-1)
        # print(attention1.shape)
        # print(fused_twoOrderSpa.shape)
        #(b,1,c,c)
        #(b,c,h,w)
        fused_twoOrderSpa = fused_OneOrderSpa.view(B, C, (H*W))
        out1 = torch.matmul(attention1[:,0], fused_twoOrderSpa)
        out1 = out1.view(B, C, H, W)


        fused = self.convf(
            torch.cat([out, out1], dim=1)) + vis

        inf = self.reflashFused2(inf)

        return fused, inf
    

class channelInteraction(nn.Module):
    def __init__(self, channelin, channelout, order=4):
        super(channelInteraction, self).__init__()
        self.order = order
        self.channelout = channelout
        cat_channels = channelin * 2
        hidden = max(channelout // 4, 16)

        self.reduce = nn.Sequential(nn.Conv2d(cat_channels, hidden, 1), nn.ReLU())
        self.expand = nn.Sequential(nn.Conv2d(hidden, channelout, 1), nn.ReLU())

        self.chaAtten = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=True)
        )

        self.reflashChaAtten = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=1, padding=0, bias=True)
            ) for _ in range(order - 1)
        ])

        self.reflashFused = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
            ) for _ in range(order - 1)
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.postprocess = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1, 1, 0)
        )

        self.norm_out = nn.GroupNorm(1, hidden)
        nn.init.constant_(self.norm_out.weight, 0.01)
        nn.init.constant_(self.norm_out.bias, 0)

    def forward(self, vis, inf, i, j):

        vis_cat = torch.cat([vis, inf], 1)
        vis_cat_r = self.reduce(vis_cat)

        chanAtten = self.chaAtten(self.avgpool(vis_cat_r)).softmax(1)
        fused = vis_cat_r * chanAtten

        for k in range(self.order - 1):
            chanAtten = self.reflashChaAtten[k](chanAtten).softmax(1)
            fused = self.reflashFused[k](fused)
            fused = fused * chanAtten

        fused = self.postprocess(fused)
        fused_res = self.norm_out(fused)
        fused_res = self.expand(fused_res)

        vis_out = fused_res + vis
        inf_out = fused_res + inf

        return vis_out, inf_out


class highOrderInteraction(nn.Module):
    def __init__(self, channelin, channelout, order=4):
        super(highOrderInteraction, self).__init__()
        self.spatial = spatialInteraction(channelin, channelout, order=order)
        self.channel = channelInteraction(channelin, channelout, order=order)

    def forward(self, vis_y, inf, i, j):

        vis_spa, inf_spa = self.spatial(vis_y, inf, i, j)

        # 进行通道交互
        vis_cha, inf_cha = self.channel(vis_spa, inf_spa, i, j)
        # vis_cha, inf_cha = self.channel(vis_y, inf, i, j)
        
        return vis_cha, inf_cha
    

# class EdgeBlock(nn.Module):
#     def __init__(self, channelin, channelout):
#         super(EdgeBlock, self).__init__()
#         self.process = nn.Conv2d(channelin,channelout,3,1,1)
#         self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
#             nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
#         self.CDC = cdcconv(channelin, channelout)

#     def forward(self,x):

#         x = self.process(x)
#         out = self.Res(x) + self.CDC(x)

#         return out
import torch.nn.functional as F
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi


# class FeatureExtract(nn.Module):
#     def __init__(self, channelin, channelout):
#         super(FeatureExtract, self).__init__()
#         self.conv = nn.Conv2d(channelin,channelout,1,1,0)
#         self.block1 = EdgeBlock(channelout,channelout)
#         self.block2 = EdgeBlock(channelout, channelout)

#     def forward(self,x):
#         xf = self.conv(x)
#         xf1 = self.block1(xf)
#         xf2 = self.block2(xf1)

#         return xf2


class Net(nn.Module):
    def __init__(self, num_channels=None,base_filter=None,args=None):
        super(Net,self).__init__()

        vis_channels=4
        inf_channels=1
        n_feat=16
        self.vis = nn.Sequential(nn.Conv2d(vis_channels,n_feat,3,1,1),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat))
        self.inf = nn.Sequential(nn.Conv2d(inf_channels,n_feat,3,1,1),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat),HinResBlock(n_feat,n_feat))

        self.interaction1 = OminiInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction2 = OminiInteraction(channelin=n_feat, channelout=n_feat)
        self.interaction3 = OminiInteraction(channelin=n_feat, channelout=n_feat)

        self.postprocess = nn.Sequential(InvBlock(DenseBlock, 3 * n_feat, 3 * n_feat // 2),
                                         nn.Conv2d(3 * n_feat, n_feat, 1, 1, 0))
        
        self.reconstruction = Refine(n_feat, out_channel=vis_channels)

        self.i = 0

    def forward(self, ms,_,pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bilinear')
        vis_y = ms
        inf = pan

        vis_y = self.vis(vis_y)
        inf = self.inf(inf)

        vis_y_feat, inf_feat = self.interaction1(vis_y, inf, self.i, j=1)
        vis_y_feat2, inf_feat2 = self.interaction2(vis_y_feat, inf_feat, self.i, j=2)
        vis_y_feat3, inf_feat3 = self.interaction3(vis_y_feat2, inf_feat2, self.i, j=3)

        fused = self.postprocess(torch.cat([vis_y_feat, vis_y_feat2, vis_y_feat3], 1))

        fused = self.reconstruction(fused)+ms

        self.i += 1

        return fused
