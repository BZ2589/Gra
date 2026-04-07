from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

import re
import torch
import torch.nn as nn

class Backbone_ResNet(nn.Module):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        self.out_indices = out_indices
        
    def forward(self,x):
        pass
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', co_selective_scan=False, **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer, co_selective_scan=co_selective_scan)
        super().__init__(**kwargs)
        self.co_selective_scan = co_selective_scan
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    @staticmethod
    def _remap_ckpt_keys_for_co_selective(k: str) -> str:
        """
        原版 VSSM: layers.{i}.blocks.{j}.*
        Co 结构: layers.{i}.blocks.blocks.{j}.* 且 CoSS2D 内参数在 op.ss2d.*（checkpoint 仍为 op.*）
        """
        k = re.sub(r"^(layers\.\d+\.blocks)\.(\d+)\.", r"\1.blocks.\2.", k)
        if "param_fuse" not in k and ".op.ss2d." not in k and ".op." in k:
            k = k.replace(".op.", ".op.ss2d.", 1)
        return k

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                _ckpt = load_file(ckpt)
                # safetensors usually doesn't have a nested 'model' key for the state dict, it IS the state dict
                # but if it does, we can try to extract it
                state_dict = _ckpt.get(key, _ckpt) if isinstance(_ckpt, dict) else _ckpt
            else:
                # 兼容 PyTorch 2.6
                _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"), weights_only=False)
                state_dict = _ckpt[key] if key in _ckpt else _ckpt
                
            # 重命名键值，以匹配模型的 state_dict
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('vmamba.'):
                    k = k[7:]
                elif k.startswith('model.'):
                    k = k[6:]
                
                if 'dt_projs.weight' in k:
                    k = k.replace('dt_projs.weight', 'dt_projs_weight')
                if 'x_proj.weight' in k:
                    k = k.replace('x_proj.weight', 'x_proj_weight')
                if 'patch_embeddings.projection.0' in k:
                    k = k.replace('patch_embeddings.projection.0', 'patch_embed.0')
                if 'patch_embeddings.projection.1' in k:
                    k = k.replace('patch_embeddings.projection.1', 'patch_embed.2')
                if 'patch_embeddings.projection.3' in k:
                    k = k.replace('patch_embeddings.projection.3', 'patch_embed.5')
                if 'patch_embeddings.projection.4' in k:
                    k = k.replace('patch_embeddings.projection.4', 'patch_embed.7')
                
                # downsample
                if 'downsample.down' in k:
                    k = k.replace('downsample.down', 'downsample.1')
                if 'downsample.norm' in k:
                    k = k.replace('downsample.norm', 'downsample.3')
                
                # Reshape flattened weights from some VMamba implementations
                if 'dt_projs_weight' in k and len(v.shape) == 2:
                    # v is [K * d_inner, dt_rank], we want [K, d_inner, dt_rank] where K=4
                    v = v.view(4, -1, v.shape[-1])
                if 'x_proj_weight' in k and len(v.shape) == 2:
                    # v is [K * N, d_inner], we want [K, N, d_inner] where K=4
                    v = v.view(4, -1, v.shape[-1])
                if 'dt_projs_bias' in k and len(v.shape) == 1:
                    v = v.view(4, -1)

                if getattr(self, "co_selective_scan", False):
                    k = self._remap_ckpt_keys_for_co_selective(k)
                
                new_state_dict[k] = v
            state_dict = new_state_dict
                
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(state_dict, strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, *inputs):
        """单时相：forward(x)。双时相协同扫描：forward(pre, post)，返回 (outs_t1, outs_t2)。"""
        if len(inputs) == 2 and self.co_selective_scan:
            pre_data, post_data = inputs[0], inputs[1]
            x1 = self.patch_embed(pre_data)
            x2 = self.patch_embed(post_data)
            outs1, outs2 = [], []
            for i, layer in enumerate(self.layers):
                x1, x2 = layer.blocks(x1, x2)
                o1, o2 = x1, x2
                x1 = layer.downsample(o1)
                x2 = layer.downsample(o2)
                if i in self.out_indices:
                    nrm = getattr(self, f'outnorm{i}')
                    out1 = nrm(o1)
                    out2 = nrm(o2)
                    if not self.channel_first:
                        out1 = out1.permute(0, 3, 1, 2).contiguous()
                        out2 = out2.permute(0, 3, 1, 2).contiguous()
                    outs1.append(out1)
                    outs2.append(out2)
            if len(self.out_indices) == 0:
                return x1, x2
            return outs1, outs2

        if len(inputs) != 1:
            raise ValueError(
                f"Backbone_VSSM: 期望 1 个输入 (x) 或协同模式下 2 个输入 (pre, post)，收到 {len(inputs)} 个"
            )
        x = inputs[0]

        def layer_forward(l, x_in):
            x_in = l.blocks(x_in)
            y = l.downsample(x_in)
            return x_in, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        if len(self.out_indices) == 0:
            return x

        return outs

