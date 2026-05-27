import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import shutil
import numpy as np
import imageio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from changedetection.configs.config import get_config
from changedetection.datasets.make_data_loader import ChangeDetectionDatset
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.MambaPyramid import MambaPyramid


def find_file(dataset_path, fname, primary_dirs, fallback_dirs):
    """按优先级查找文件"""
    for d in primary_dirs:
        p = os.path.join(dataset_path, d, fname)
        if os.path.exists(p):
            return p
    for d in fallback_dirs:
        p = os.path.join(dataset_path, d, fname)
        if os.path.exists(p):
            return p
    return None


def resolve_test_data_name_list(args):
    """解析测试集文件列表，返回 (name_list, actual_dataset_path)"""
    dataset_path = args.test_dataset_path
    dataset_name = args.dataset

    if dataset_name in ('LEVIR-CD', 'LEVIR-CD+'):
        # 优先用 list 文件（LEVIR-CD256 需要）
        if args.test_data_list_path and os.path.exists(args.test_data_list_path):
            with open(args.test_data_list_path, 'r') as f:
                lines = f.read().strip().split('\n')
            return [l.strip() for l in lines if l.strip()], dataset_path
        for list_path in [
            os.path.join(dataset_path, 'list', 'test.txt'),
            os.path.join(os.path.dirname(dataset_path), 'list', 'test.txt'),
        ]:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                return [l.strip() for l in lines if l.strip()], dataset_path
        # 优先 test/A/（LEVIR-CD-1024）
        test_a = os.path.join(dataset_path, 'test', 'A')
        if os.path.exists(test_a) and os.listdir(test_a):
            return sorted([f for f in os.listdir(test_a) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]), os.path.join(dataset_path, 'test')
        # fallback: 根目录 A/
        a = os.path.join(dataset_path, 'A')
        if os.path.exists(a) and os.listdir(a):
            return sorted([f for f in os.listdir(a) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]), dataset_path
        # 尝试 val/A/
        val_a = os.path.join(dataset_path, 'val', 'A')
        if os.path.exists(val_a) and os.listdir(val_a):
            return sorted([f for f in os.listdir(val_a) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]), os.path.join(dataset_path, 'val')
        raise FileNotFoundError(f'Cannot find test images/list under {dataset_path}')

    elif dataset_name == 'WHU-CD':
        # 优先 test/A/，其次 A/
        test_a = os.path.join(dataset_path, 'test', 'A')
        if os.path.exists(test_a) and os.listdir(test_a):
            return sorted([f for f in os.listdir(test_a) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]), os.path.join(dataset_path, 'test')
        a = os.path.join(dataset_path, 'A')
        if os.path.exists(a) and os.listdir(a):
            return sorted([f for f in os.listdir(a) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]), dataset_path
        raise FileNotFoundError(f'Cannot find test images under {dataset_path}')

    elif dataset_name == 'SYSU':
        # 需要 list 文件
        if args.test_data_list_path and os.path.exists(args.test_data_list_path):
            with open(args.test_data_list_path, 'r') as f:
                lines = f.read().strip().split('\n')
            return [l.strip() for l in lines if l.strip()], dataset_path
        # 尝试自动找 list
        for list_path in [
            os.path.join(dataset_path, 'list', 'test.txt'),
            os.path.join(os.path.dirname(dataset_path), 'list', 'test.txt'),
        ]:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                return [l.strip() for l in lines if l.strip()], dataset_path
        raise FileNotFoundError(f'Cannot find test list for SYSU under {dataset_path}')

    elif dataset_name == 'DSIFN-CD':
        # 优先 list 文件
        if args.test_data_list_path and os.path.exists(args.test_data_list_path):
            with open(args.test_data_list_path, 'r') as f:
                lines = f.read().strip().split('\n')
            return [l.strip() for l in lines if l.strip()], dataset_path
        for list_path in [
            os.path.join(dataset_path, 'list', 'test.txt'),
            os.path.join(os.path.dirname(dataset_path), 'list', 'test.txt'),
        ]:
            if os.path.exists(list_path):
                with open(list_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                return [l.strip() for l in lines if l.strip()], dataset_path
        # fallback: 扫 t1/
        t1 = os.path.join(dataset_path, 't1')
        if os.path.exists(t1) and os.listdir(t1):
            return sorted([f for f in os.listdir(t1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]), dataset_path
        raise FileNotFoundError(f'Cannot find test images/list for DSIFN-CD under {dataset_path}')

    else:
        raise NotImplementedError(f'Unsupported dataset: {dataset_name}')


def find_raw_image(dataset_path, fname, t1=True):
    """按优先级查找原始图片（用于复制 T1/T2.png）"""
    sub = 'T1' if t1 else 'T2'
    a = 'A' if t1 else 'B'
    # 优先级: T1/T2 -> test/T1 -> A -> test/A
    for primary, fallback in [
        (sub, a),
        (os.path.join('test', sub), os.path.join('test', a)),
    ]:
        p = find_file(dataset_path, fname, [primary], [fallback])
        if p:
            return p
    return None


def find_label(dataset_path, fname):
    """查找 label 文件"""
    stem = os.path.splitext(fname)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        for d in ['label', 'mask', 'mask_png']:
            p = os.path.join(dataset_path, d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def save_sample(output_dir, dataset_path, fname, pred_mask, label_mask):
    """保存单个样本的所有输出文件"""
    sample_name = os.path.splitext(fname)[0]
    sample_dir = os.path.join(output_dir, sample_name)

    # 覆盖逻辑：已存在则删除重建
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir)

    # T1.png — 原始时相1
    t1_src = find_raw_image(dataset_path, fname, t1=True)
    if t1_src:
        shutil.copy2(t1_src, os.path.join(sample_dir, 'T1.png'))

    # T2.png — 原始时相2
    t2_src = find_raw_image(dataset_path, fname, t1=False)
    if t2_src:
        shutil.copy2(t2_src, os.path.join(sample_dir, 'T2.png'))

    # label.png — 黑白二值图
    label_src = find_label(dataset_path, fname)
    if label_src:
        label_img = np.array(imageio.imread(label_src))
        if label_img.max() > 1.5:
            label_img = (label_img > 127).astype(np.uint8) * 255
        else:
            label_img = (label_img > 0.5).astype(np.uint8) * 255
        imageio.imwrite(os.path.join(sample_dir, 'label.png'), label_img.astype(np.uint8))
    else:
        # 用模型输入的 label
        label_img = (label_mask > 0.5).astype(np.uint8) * 255
        imageio.imwrite(os.path.join(sample_dir, 'label.png'), label_img.astype(np.uint8))

    # prediction.png — 黑白二值图
    pred_img = (pred_mask > 0.5).astype(np.uint8) * 255
    imageio.imwrite(os.path.join(sample_dir, 'prediction.png'), pred_img.astype(np.uint8))

    # color_map.png — TP白 TN黑 FP红 FN绿
    h, w = pred_mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    pred_bin = pred_mask > 0.5
    label_bin = label_mask > 0.5
    color[pred_bin & label_bin] = [255, 255, 255]       # TP 白
    color[~pred_bin & ~label_bin] = [0, 0, 0]             # TN 黑
    color[pred_bin & ~label_bin] = [255, 0, 0]            # FP 红
    color[~pred_bin & label_bin] = [0, 255, 0]            # FN 绿
    imageio.imwrite(os.path.join(sample_dir, 'color_map.png'), color)


class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = MambaPyramid(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            decoder_depths=args.decoder_depths,
            dims=config.MODEL.VSSM.EMBED_DIM,
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
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

        # 加载 checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume, weights_only=False)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict, strict=False)
            print(f'Loaded checkpoint: {args.resume}')
        else:
            print('WARNING: No checkpoint loaded! Running with random weights.')

        self.deep_model = self.deep_model.cuda()
        self.deep_model.eval()

        # 解析测试数据
        self.test_data_name_list, self.actual_dataset_path = resolve_test_data_name_list(args)
        print(f'Dataset: {args.dataset}')
        print(f'Test path: {self.actual_dataset_path}')
        print(f'Test samples: {len(self.test_data_name_list)}')

        # 输出目录
        self.output_dir = os.path.join(args.result_saved_path, args.dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f'Output dir: {self.output_dir}')

    def infer(self):
        self.evaluator.reset()

        dataset = ChangeDetectionDatset(
            self.args.dataset,
            self.actual_dataset_path,
            self.test_data_name_list,
            256, None, 'test'
        )
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for itera, data in enumerate(tqdm(val_data_loader, ncols=80, desc='Inference')):
                pre_imgs, post_imgs, labels, names = data
                pre_imgs = pre_imgs.cuda().float()
                post_imgs = post_imgs.cuda().float()
                labels = labels.cuda().long()

                output_1, _ = self.deep_model(pre_imgs, post_imgs)
                pred = np.argmax(output_1.data.cpu().numpy(), axis=1)
                labels_np = labels.cpu().numpy()

                self.evaluator.add_batch(labels_np, pred)

                fname = names[0]
                pred_mask = np.squeeze(pred).astype(np.float32)
                label_mask = np.squeeze(labels_np).astype(np.float32)

                save_sample(self.output_dir, self.actual_dataset_path, fname, pred_mask, label_mask)

        # 打印指标
        f1 = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'\nResults:')
        print(f'  Recall:    {rec:.4f}')
        print(f'  Precision: {pre:.4f}')
        print(f'  OA:        {oa:.4f}')
        print(f'  F1:        {f1:.4f}')
        print(f'  IoU:       {iou:.4f}')
        print(f'  Kappa:     {kc:.4f}')
        print(f'\nDone! Results saved to {self.output_dir}')


def main():
    parser = argparse.ArgumentParser(description="Testing on LEVIR-CD/WHU-CD/SYSU/DSIFN-CD")
    parser.add_argument('--cfg', type=str, default='./changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument('--opts', default=None, nargs='+', help='Modify config options by adding KEY VALUE pairs')
    parser.add_argument('--dataset', type=str, required=True, choices=['LEVIR-CD', 'LEVIR-CD+', 'WHU-CD', 'SYSU', 'DSIFN-CD'])
    parser.add_argument('--test_dataset_path', type=str, required=True)
    parser.add_argument('--test_data_list_path', type=str, default=None, help='Path to test.txt (required for SYSU, optional for DSIFN-CD)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pth file')
    parser.add_argument('--pretrained_weight_path', type=str, default=None)
    parser.add_argument('--decoder_depths', type=int, default=4)
    parser.add_argument('--result_saved_path', type=str, default='./test_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iters', type=int, default=800000)

    args = parser.parse_args()
    infer = Inference(args)
    infer.infer()


if __name__ == "__main__":
    main()
