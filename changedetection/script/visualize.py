import os
import sys
# 向上跳两级找到 Gra 目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm  # 加入进度条

from changedetection.configs.config import get_config
from changedetection.models.MambaPyramid import MambaPyramid

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        a_dir = os.path.join(root_dir, 'A')
        if os.path.exists(a_dir):
            for img_name in os.listdir(a_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.samples.append(img_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        t1_path = os.path.join(self.root_dir, 'A', img_name)
        t2_path = os.path.join(self.root_dir, 'B', img_name)
        label_path = os.path.join(self.root_dir, 'label', img_name)
        
        t1 = Image.open(t1_path).convert('RGB')
        t2 = Image.open(t2_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # grayscale
        
        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            # Label also goes through ToTensor, making it [0.0, 1.0]
            label = self.transform(label)
        
        return t1, t2, label, img_name

def create_color_map(pred, gt):
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    gt = gt.squeeze().cpu().numpy().astype(np.uint8)
    
    color_map = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    # TP: white (预测变了，实际也变了)
    color_map[(pred == 1) & (gt == 1)] = [255, 255, 255]  
    # TN: black (预测没变，实际也没变)
    color_map[(pred == 0) & (gt == 0)] = [0, 0, 0]       
    # FP: red (预测变了，实际没变 -> 误检)
    color_map[(pred == 1) & (gt == 0)] = [255, 0, 0]     
    # FN: green (预测没变，实际变了 -> 漏检)
    color_map[(pred == 0) & (gt == 1)] = [0, 255, 0]     
    
    return color_map

def main(pth_path):
    # Extract model_name from pth_path
    model_name = os.path.basename(os.path.dirname(pth_path))
    save_dir = os.path.join('visual_results', model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 【修改点1】：替换为你服务器上的真实测试集物理路径！
    test_dir = '/home/z/dataset/LEVIR-CD-1024/test'
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # ================= 开始配置模型 =================
    parser = argparse.ArgumentParser()
    # 配置文件路径对齐你的训练脚本
    parser.add_argument('--cfg', type=str, default='changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--pretrained_weight_path', type=str, default=None)
    parser.add_argument('--decoder_depths', type=int, default=4)
    parser.add_argument('--drop_rate', type=float, default=0.0) 
    args, _ = parser.parse_known_args()
    
    config = get_config(args)

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
        use_checkpoint=False, 
    )
    # ================= 配置结束 =================
    
    # Load weights
    print(f"Loading weights from: {pth_path}")
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded and moved to {device}. Starting inference...")
    
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        # 【修改点2】：加入 tqdm 进度条，实时显示推理进度
        for batch in tqdm(test_loader, desc="Generating Visualizations"):
            t1, t2, gt, names = batch
            t1, t2, gt = t1.to(device), t2.to(device), gt.to(device)
            
            # 使用自动混合精度(FP16/BF16)防止高阶模块溢出，与你训练时保持一致
            with torch.amp.autocast('cuda'):
                pred, _ = model(t1, t2) 
            
            # 2通道输出，用 argmax 取最大概率通道
            pred_binary = torch.argmax(pred, dim=1, keepdim=True).float()
            
            for i in range(t1.size(0)):
                name = names[i]
                
                # Convert to images
                t1_img = to_pil(t1[i])
                t2_img = to_pil(t2[i])
                gt_img = to_pil(gt[i].repeat(3, 1, 1)) 
                pred_img = to_pil(pred_binary[i].repeat(3, 1, 1)) 
                
                # Create color map
                color_map = create_color_map(pred_binary[i], gt[i])
                color_img = Image.fromarray(color_map)
                
                # Concatenate images horizontally (拼接5张图)
                width, height = t1_img.size
                big_img = Image.new('RGB', (width * 5, height))
                big_img.paste(t1_img, (0, 0))
                big_img.paste(t2_img, (width, 0))
                big_img.paste(gt_img, (2 * width, 0))
                big_img.paste(pred_img, (3 * width, 0))
                big_img.paste(color_img, (4 * width, 0))
                
                # Save
                save_path = os.path.join(save_dir, name)
                big_img.save(save_path)
                
    print(f"Done! All images saved to {save_dir}")

if __name__ == '__main__':
    # 【修改点3】：Linux 环境下的路径，使用正斜杠
    pth_path = 'changedetection/saved_models/LEVIR-CD/baseline_base_levir_drop=0_levircd_base003/90500_model.pth'
    main(pth_path)