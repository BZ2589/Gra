from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

import torch
import torch.nn as nn

class Backbone_ResNet(nn.Module):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        self.out_indices = out_indices
        
    def forward(self,x):
        pass
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
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
                
                new_state_dict[k] = v
            state_dict = new_state_dict
                
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(state_dict, strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y
        # import pdb
        # pdb.set_trace()
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # import pdb
        # pdb.set_trace()
        if len(self.out_indices) == 0:
            return x
        
        return outs

