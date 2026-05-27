import os
import sys
import shutil
from PIL import Image


def crop_results(model_dir, crop_size=256):
    """
    将 test_results/{model_dir}/ 下的每张 1024x1024 图片裁剪成 256x256 的小块。

    输出结构：
    test_results/{model_dir}/{sample_name}/cut/{row},{col}/
        T1.png
        T2.png
        label.png
        prediction.png
        color_map.png
    """
    files = ['T1.png', 'T2.png', 'label.png', 'prediction.png', 'color_map.png']

    sample_dirs = sorted([
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d != 'cut'
    ])

    for sample_name in sample_dirs:
        sample_path = os.path.join(model_dir, sample_name)
        cut_base = os.path.join(sample_path, 'cut')

        if os.path.exists(cut_base):
            shutil.rmtree(cut_base)

        img_path = os.path.join(sample_path, files[0])
        if not os.path.exists(img_path):
            print(f"⚠️  {sample_name} 缺少 {files[0]}，跳过")
            continue

        img = Image.open(img_path)
        w, h = img.size
        cols = w // crop_size
        rows = h // crop_size

        print(f"📐 {sample_name}: {w}x{h} -> {rows}x{cols} = {rows * cols} 块")

        for r in range(rows):
            for c in range(cols):
                region = (c * crop_size, r * crop_size, (c + 1) * crop_size, (r + 1) * crop_size)
                out_dir = os.path.join(cut_base, f"{r + 1},{c + 1}")
                os.makedirs(out_dir, exist_ok=True)

                for f in files:
                    src = os.path.join(sample_path, f)
                    if os.path.exists(src):
                        cropped = Image.open(src).crop(region)
                        cropped.save(os.path.join(out_dir, f))

    print(f"\n✅ 全部裁剪完成，结果在 {model_dir}/*/cut/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python cut_results.py <model_dir> [crop_size]")
        print("示例: python cut_results.py test_results/baseline_base_levir_drop=0_levircd_hoi31")
        sys.exit(1)

    model_dir = sys.argv[1]
    crop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    if not os.path.isdir(model_dir):
        print(f"❌ 目录不存在: {model_dir}")
        sys.exit(1)

    crop_results(model_dir, crop_size)
