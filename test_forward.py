import sys
import os
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from changedetection.configs.config import get_config
from changedetection.models.MambaPyramid import MambaPyramid

def parse_option():
    parser = argparse.ArgumentParser('MambaCD local testing script', add_help=False)
    parser.add_argument('--cfg', type=str, default='changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml', metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='')
    parser.add_argument('--decoder_depths', type=int, default=4)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    args, unparsed = parser.parse_known_args()
    return args

def main():
    print("====== 开始本地环境与前向传播测试 ======")
    args = parse_option()
    
    print(f"正在加载配置: {args.cfg}")
    config = get_config(args)
    
    print("正在初始化 MambaPyramid 模型...")
    # 模拟 train_MambaBCD.py 中的初始化参数
    try:
        model = MambaPyramid(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            decoder_depths=args.decoder_depths,
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
            drop_rate=args.drop_rate,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        print("模型初始化成功！")
    except Exception as e:
        print(f"模型初始化失败，可能缺失依赖或 CUDA 算子未编译: {e}")
        return

    # 检查是否有 CUDA 且算子是否正常工作
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    model.to(device)
    
    print("正在创建测试数据 (Batch Size 2, 3 Channels, 256x256)...")
    # 变化检测通常需要两张图片 (t1, t2)
    x1 = torch.randn(2, 3, 256, 256).to(device)
    x2 = torch.randn(2, 3, 256, 256).to(device)
    
    print("进行前向传播测试...")
    try:
        with torch.no_grad():
            outputs = model(x1, x2)
            
        print(f"前向传播成功！")
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            for i, out in enumerate(outputs):
                if hasattr(out, 'shape'):
                    print(f"输出 {i} shape: {out.shape}")
                else:
                    print(f"输出 {i}: type {type(out)}")
        else:
            print(f"输出 shape: {outputs.shape}")
        print("====== 测试通过，网络结构一切正常！ ======")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"前向传播失败: {e}")
        print("注意: 可能是 kernels/selective_scan 尚未编译，请先编译 C++/CUDA 扩展！")

if __name__ == '__main__':
    main()
