import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from changedetection.models.SwinPyramid import SwinPyramid
from changedetection.configs.config import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument("--opts", default=None, nargs='+')
    args = parser.parse_args()
    config = get_config(args)

    print("正在实例化 SwinPyramid 网络架构...")
    model = SwinPyramid(
        pretrained=None,
        hoi_levels=[1, 1, 1, 1],
        hoi_order=3,
        hoi_layers=1,
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        decoder_depths=4,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        drop_rate=0.2,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=False
    ).cuda()
    model.eval()

    dummy_input1 = torch.randn(1, 3, 256, 256).cuda()
    dummy_input2 = torch.randn(1, 3, 256, 256).cuda()

    print("="*40)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params:  {total_params / 1e6:.2f} M")

    from thop import profile
    flops, _ = profile(model, inputs=(dummy_input1, dummy_input2), verbose=False)
    print(f"  FLOPs (thop):   {flops / 1e9:.2f} G")
    print("="*40)

if __name__ == "__main__":
    main()
