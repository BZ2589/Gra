import torch
from torchvision.models import swin_t
from fvcore.nn import FlopCountAnalysis

def main():
    print("正在实例化 标准Transformer (Swin-T) 基准网络...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化官方的 Swin Transformer Tiny 版本作为原生对照组
    model = swin_t().to(device)
    model.eval()

    # 模拟与Mamba相同的双时相特征提取工作量
    # 变化检测通常需要对两张图像分别提取特征，因此这里将单张图像的计算量乘以2
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    print("="*40)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  原生Transformer 真实参数量 Params:  {total_params / 1e6:.2f} M")

    try:
        with torch.amp.autocast('cuda'):
            flops = FlopCountAnalysis(model, dummy_input)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            
            # 乘以2模拟处理前后两时相图像的总开销
            total_flops = flops.total() * 2
            print(f"  原生Transformer 计算量 FLOPs:      {total_flops / 1e9:.2f} G")
    except Exception as e:
        print(f"计算失败: {e}")
    print("="*40)

if __name__ == "__main__":
    main()