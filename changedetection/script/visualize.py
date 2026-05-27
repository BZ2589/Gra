import os
import sys
import shutil
# 向上跳两级找到 Gra 目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

from changedetection.configs.config import get_config
from changedetection.models.MambaPyramid import MambaPyramid
from changedetection.utils_func.metrics import Evaluator

# ImageNet 标准化参数（与训练脚本中的 imutils.normalize_img 保持一致）
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]


def normalize_img_torch(img_tensor):
    """
    对 tensor 图像进行 ImageNet 标准化（与训练脚本的 imutils.normalize_img 一致）
    img_tensor: shape (C, H, W), 值域 [0, 1]（ToTensor 后的结果）
    返回: 标准化后的 tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=img_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img_tensor.dtype).view(3, 1, 1)
    # 先将 [0,1] 转为 [0,255]，然后进行标准化
    img_tensor = img_tensor * 255.0
    img_tensor = (img_tensor - mean) / std
    return img_tensor


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, normalize=True):
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.samples = []

        t1_dirs = ['A', 'im1', 't1', 'time1']
        t2_dirs = ['B', 'im2', 't2', 'time2']
        label_dirs = ['label', 'mask']

        self.t1_dir = None
        for d in t1_dirs:
            path = os.path.join(root_dir, d)
            if os.path.exists(path):
                self.t1_dir = path
                break

        self.t2_dir = None
        for d in t2_dirs:
            path = os.path.join(root_dir, d)
            if os.path.exists(path):
                self.t2_dir = path
                break

        self.label_dir = None
        for d in label_dirs:
            path = os.path.join(root_dir, d)
            if os.path.exists(path):
                self.label_dir = path
                break

        # 尝试读取 list/test.txt（如果存在）
        list_file = os.path.join(root_dir, 'list', 'test.txt')
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.samples = [x.strip() for x in f.readlines() if x.strip()]
        elif self.t1_dir:
            for img_name in os.listdir(self.t1_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.samples.append(img_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        t1_path = os.path.join(self.t1_dir, img_name)
        t2_path = os.path.join(self.t2_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        t1 = Image.open(t1_path).convert('RGB')
        t2 = Image.open(t2_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            label = self.transform(label)

        # 对图像进行 ImageNet 标准化（与训练脚本保持一致）
        if self.normalize:
            t1 = normalize_img_torch(t1)
            t2 = normalize_img_torch(t2)

        # 标签 squeeze 到2维 (H, W)，与训练脚本的 ChangeDetectionDatset 对齐
        label = label.squeeze(0)

        return t1, t2, label, img_name


def create_color_map(pred, gt):
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    gt = gt.squeeze().cpu().numpy().astype(np.uint8)

    color_map = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    color_map[(pred == 1) & (gt == 1)] = [255, 255, 255]
    color_map[(pred == 0) & (gt == 0)] = [0, 0, 0]
    color_map[(pred == 1) & (gt == 0)] = [255, 0, 0]
    color_map[(pred == 0) & (gt == 1)] = [0, 255, 0]

    return color_map


def main(pth_path, test_dir, save_dir, hoi_order=3, hoi_levels=None):
    os.makedirs(save_dir, exist_ok=True)

    # 使用与训练脚本相同的数据预处理流程
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = TestDataset(test_dir, transform=transform, normalize=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--pretrained_weight_path', type=str, default=None)
    parser.add_argument('--decoder_depths', type=int, default=4)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    args, _ = parser.parse_known_args()

    config = get_config(args)

    model = MambaPyramid(
        pretrained=args.pretrained_weight_path,
        hoi_order=hoi_order,
        hoi_levels=hoi_levels,
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
        use_checkpoint=False,
    )

    print(f"Loading weights from: {pth_path}")
    # 与训练脚本保持一致的选择性加载方式
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in checkpoint.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {len(model_dict)}/{len(state_dict)} parameters")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded and moved to {device}. Starting inference...")

    to_pil = transforms.ToPILImage()
    evaluator = Evaluator(num_class=2)

    amp_enabled = device.type == 'cuda'
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Visualizations"):
            t1, t2, gt, names = batch
            t1, t2 = t1.to(device), t2.to(device)
            # 标签转为 long 类型（与训练脚本保持一致）
            gt = gt.to(device).long()

            if amp_enabled:
                with torch.amp.autocast('cuda'):
                    pred, _ = model(t1, t2)
            else:
                pred, _ = model(t1, t2)

            # 与训练脚本完全一致的推理逻辑
            pred = pred.float()
            pred_np = pred.data.cpu().numpy()
            pred_binary = np.argmax(pred_np, axis=1)  # (B, H, W)
            gt_np = gt.cpu().numpy()  # (B, H, W)，值为 0 或 1

            for i in range(t1.size(0)):
                name = names[i]
                sample_name = os.path.splitext(name)[0]
                sample_dir = os.path.join(save_dir, sample_name)

                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)
                os.makedirs(sample_dir)

                # 保存图像（需要反标准化才能正确显示）
                t1_img = to_pil(t1[i].clamp(0, 1))
                t1_img.save(os.path.join(sample_dir, 'T1.png'))

                t2_img = to_pil(t2[i].clamp(0, 1))
                t2_img.save(os.path.join(sample_dir, 'T2.png'))

                # 保存标签
                label_img = Image.fromarray((gt_np[i] * 255).astype(np.uint8))
                label_img.save(os.path.join(sample_dir, 'label.png'))

                # 保存预测结果
                pred_img = Image.fromarray((pred_binary[i] * 255).astype(np.uint8))
                pred_img.save(os.path.join(sample_dir, 'prediction.png'))

                # 创建彩色误差图
                pred_tensor = torch.from_numpy(pred_binary[i:i+1]).unsqueeze(0)
                gt_tensor = torch.from_numpy(gt_np[i:i+1]).unsqueeze(0).float()
                color_map = create_color_map(pred_tensor, gt_tensor)
                color_img = Image.fromarray(color_map)
                color_img.save(os.path.join(sample_dir, 'color_map.png'))

                # 评估 - 直接传入 numpy（与训练脚本一致）
                pred_flat = pred_binary[i].flatten()
                gt_flat = gt_np[i].flatten()
                evaluator.add_batch(gt_flat, pred_flat)

    print(f"Done! All images saved to {save_dir}")
    print("\n========== Evaluation Metrics ==========")
    print(f"Recall:    {evaluator.Pixel_Recall_Rate():.4f}")
    print(f"Precision: {evaluator.Pixel_Precision_Rate():.4f}")
    print(f"OA:        {evaluator.Pixel_Accuracy():.4f}")
    print(f"F1:        {evaluator.Pixel_F1_score():.4f}")
    print(f"IoU:       {evaluator.Intersection_over_Union():.4f}")
    print(f"Kappa:     {evaluator.Kappa_coefficient():.4f}")
    print("=========================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--test_dataset_path', type=str, required=True, help='Path to test dataset root')
    parser.add_argument('--result_saved_path', type=str, default='./test_results', help='Root directory for saving results')
    parser.add_argument('--hoi_order', type=int, default=3, help='HOI interaction order (default: 3)')
    parser.add_argument('--hoi_levels', type=int, nargs='*', default=None, help='HOI enable flags for 4 encoder levels')
    args = parser.parse_args()

    model_name = os.path.basename(os.path.dirname(args.resume))
    save_dir = os.path.join(args.result_saved_path, model_name)

    main(args.resume, args.test_dataset_path, save_dir, hoi_order=args.hoi_order, hoi_levels=args.hoi_levels)
