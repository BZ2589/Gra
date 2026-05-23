import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import time
from thop import profile
import numpy as np
import imageio
from random import randint
from changedetection.configs.config import get_config
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.MambaPyramid import MambaPyramid

import changedetection.utils_func.lovasz_loss as L


def safe_cross_entropy(logits, labels, ignore_index=255):
    valid_mask = labels.ne(ignore_index)
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
class Trainer(object):
    def __init__(self, args, hoi_order=4, hoi_layers=1):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        print(f"训练批次总数(T_max) = {len(self.train_data_loader)}")
        print(f"HOI_CONFIG (levels) = {args.hoi_levels}")
        print(f"HOI_ORDER (阶数)    = {hoi_order}")
        print(f"HOI_LAYERS (堆叠)   = {hoi_layers}")
        
        # 初始化验证集loader
        val_dataset = ChangeDetectionDatset(args.dataset, args.test_dataset_path, args.test_data_name_list, 256, None, 'test')
        self.val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False)

        self.evaluator = Evaluator(num_class=2)
        log_suffix = args.train_name if args.train_name else str(time.time())
        self.writer = SummaryWriter(log_dir=f"./logs/{self.args.model_type}_{log_suffix}")
        self.deep_model = MambaPyramid(
            pretrained=args.pretrained_weight_path,
            hoi_levels=args.hoi_levels,
            hoi_order=hoi_order,
            hoi_layers=hoi_layers,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            decoder_depths = args.decoder_depths,
            # ===================
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
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            drop_rate = args.drop_rate,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=True,
            ) 
        
        self.deep_model = self.deep_model.cuda()
        
        # Build save path with train_name if provided, else use timestamp
        save_suffix = args.train_name if args.train_name else str(time.time())
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + save_suffix)
        
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size
        torch.random.manual_seed(3407)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            # 添加 weights_only=False 以兼容 PyTorch 2.6 加载 safetensors
            checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        # ==========================================
        # 新增：分层学习率 (Layer-wise Learning Rate)
        # ==========================================
        hoi_params = []
        base_params = []

        # 遍历网络中所有的参数
        for name, param in self.deep_model.named_parameters():
            if "fusion_adapters" in name:
                # 只要参数名字里带有 fusion_adapters（也就是你的高阶交互模块）
                hoi_params.append(param)
            else:
                # 其余的骨干网络 (VMamba Encoder)、Decoder、分类头等
                base_params.append(param)

        # 打印一下参数分组情况（可选，用来在终端确认分配对了）
        print(f"✅ 分层学习率已开启: Base 参数 {len(base_params)} 个 (lr=1e-4), HOI 参数 {len(hoi_params)} 个 (lr=1e-5)")

        # 将分组参数传入优化器
        self.optim = optim.AdamW([
            {'params': base_params, 'lr': 1e-4},  # 皮实的骨干网络用 1e-4 大步流星
            {'params': hoi_params, 'lr': 1e-5}    # 脆弱的高阶模块用 1e-5 谨慎微调
        ], weight_decay=args.weight_decay, fused=True)
        # ==========================================
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optim,               # 优化器
                        T_max=len(self.train_data_loader),             # 学习率下限
                        # 将学习率改成len(self.train_data_loader)而不是一个固定的10000
                    )
        self.scaler = torch.amp.GradScaler('cuda')

    def training(self):
        best_kc = 0.0
        best_iter = 0
        best_round = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        with open(os.path.join(self.model_save_path,'result.txt'),'w') as output:
            output.write(f'best round:{best_round}\n best iter: {best_iter}')
        
        # Initialize CSV for logging validation metrics
        csv_path = os.path.join(self.model_save_path, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            f.write('iter,rec,pre,oa,f1_score,iou,kc\n')

        # === 新增：初始化记录训练 loss 的 CSV ===
        loss_csv_path = os.path.join(self.model_save_path, 'train_loss.csv')
        with open(loss_csv_path, 'w', newline='') as f:
            f.write('iter,ce_loss,ds_loss,final_loss\n')
        # ==========================================

            
        train_enumerator = enumerate(self.train_data_loader)
        pbar = tqdm(range(elem_num), disable=not sys.stdout.isatty())
        start_time = time.time()  # 新增：记录开始时间
        for _ in pbar:
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            self.optim.zero_grad()
            with torch.amp.autocast('cuda'):
                output_1,ds_feature = self.deep_model(pre_change_imgs, post_change_imgs)
                ce_loss_1 = safe_cross_entropy(output_1, labels, ignore_index=255)
                ce_loss_ds = 0
                lovasz_loss = 0
                for index in range(len(ds_feature)):
                    ce_loss_ds += safe_cross_entropy(ds_feature[index], labels, ignore_index=255)*index/4
                    # 强制在计算 lovasz_loss 前转为 float32，避免底层的 torch.dot 在 fp16 下崩溃
                    lovasz_loss += L.lovasz_softmax(F.softmax(ds_feature[index].float(), dim=1), labels, ignore=255)*index/4
                lovasz_loss += L.lovasz_softmax(F.softmax(output_1.float(), dim=1), labels, ignore=255)
                main_loss = ce_loss_1 + 0.75 * lovasz_loss + ce_loss_ds
                final_loss = main_loss
                
            self.writer.add_scalar(tag="ce_loss",scalar_value=ce_loss_1.item(),global_step=itera+1)
            self.writer.add_scalar(tag="ds_loss",scalar_value=ce_loss_ds.item() if isinstance(ce_loss_ds, torch.Tensor) else ce_loss_ds,global_step=itera+1)
            self.writer.add_scalar(tag="final_loss",scalar_value=final_loss.item(),global_step=itera+1)
            
            self.scaler.scale(final_loss).backward()
            # AMP-safe global gradient clipping:
            # unscale -> clip -> step
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), max_norm=0.5)
            self.scaler.step(self.optim)
            self.scaler.update()
            # 他scheduler.step()  # 将学习率调度器的更新放在 optimizer.step() 之后，
            # 确保每次迭代都正确更新学习率
            self.scheduler.step()
            
            if (itera + 1) % 10 == 0:
                pbar.set_postfix({'loss': f'{final_loss.item():.4f}'})

                # === 新增：每 10 轮把当前的所有 Loss 写入 CSV ===
                with open(loss_csv_path, 'a', newline='') as f:
                    ds_loss_val = ce_loss_ds.item() if isinstance(ce_loss_ds, torch.Tensor) else ce_loss_ds
                    f.write(f'{itera + 1},{ce_loss_1.item():.6f},{ds_loss_val:.6f},{final_loss.item():.6f}\n')
                # ==============================================

                if (itera + 1) % 500 == 0:
                    # 新增：计算并打印当前进度和预计剩余时间
                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / (itera + 1)
                    eta_seconds = avg_time_per_iter * (elem_num - (itera + 1))
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    print(f"当前迭代: {itera + 1}/{elem_num}, 预计剩余时间: {eta_str}")

                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation(iter=itera)
                    
                    # Log metrics to CSV
                    with open(csv_path, 'a', newline='') as f:
                        f.write(f'{itera + 1},{rec},{pre},{oa},{f1_score},{iou},{kc}\n')
                        
                    if kc >= best_kc:
                        # 新增逻辑：如果之前已经保存过最优模型，先把旧的删了腾空间
                        if best_iter > 0:
                            old_model_path = os.path.join(self.model_save_path, f'{best_iter}_model.pth')
                            if os.path.exists(old_model_path):
                                os.remove(old_model_path)

                        # 原有逻辑：保存当前最新的最优模型
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_iter = itera + 1
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    print('best round:',best_round)
                    print('best iteration:',best_iter)
                    with open(os.path.join(self.model_save_path,'result.txt'),'w') as output:
                        output.write(f'best round:{best_round}\n best iter: {best_iter}')
                    self.deep_model.train()
        self.writer.close()
        print('The accuracy of the best round is ', best_round)
        print('best iteration:',best_iter)
        print('训练完成')

    def validation(self,iter):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for itera, data in enumerate(self.val_data_loader):
                pre_change_imgs, post_change_imgs, labels, name = data             
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda().float()
                labels = labels.cuda().long()
                
                with torch.amp.autocast('cuda'):
                    output_1,ds_feature = self.deep_model(pre_change_imgs, post_change_imgs)              
                    ce_loss_1 = safe_cross_entropy(output_1, labels, ignore_index=255)
                    ce_loss_ds = 0
                    lovasz_loss = 0
                    for index in range(len(ds_feature)):
                        ce_loss_ds += safe_cross_entropy(ds_feature[index], labels, ignore_index=255)*index/2
                        lovasz_loss += L.lovasz_softmax(F.softmax(ds_feature[index].float(), dim=1), labels, ignore=255)*index/2
                    lovasz_loss += L.lovasz_softmax(F.softmax(output_1.float(), dim=1), labels, ignore=255)
                    main_loss = ce_loss_1 + 0.75 * lovasz_loss + ce_loss_ds
                    final_loss = main_loss
                    
                # self.scheduler.step()
                output_1 = output_1.float() # 确保后续转换为 numpy 时是 float32 类型
                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()
                self.evaluator.add_batch(labels, output_1)
                self.writer.add_scalar(tag="ce_loss_validation",scalar_value=ce_loss_1.item(),global_step=iter+1)
                self.writer.add_scalar(tag="ds_loss_validation",scalar_value=ce_loss_ds.item() if isinstance(ce_loss_ds, torch.Tensor) else ce_loss_ds,global_step=iter+1)
                self.writer.add_scalar(tag="final_loss_validation",scalar_value=final_loss.item(),global_step=iter+1)
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        self.writer.add_scalar(tag="f1-score",scalar_value=f1_score,global_step=iter+1)
        self.writer.add_scalar(tag="kc",scalar_value=kc,global_step=iter+1)
        self.writer.add_scalar(tag="IoU",scalar_value=iou,global_step=iter+1)
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


def main():
    # ============================================================
    # 消融实验配置区：直接修改这里的数字即可切换实验
    #
    # --- HOI 启用开关（4 个数字对应 Encoder 输出的 4 个分辨率层级，从大到小）---
    #   1 = 使用 HOI_Fusion_Adapter（高阶交互融合）
    #   0 = 使用 Bypass_Fusion_Adapter（简单拼接融合）
    #
    # 实验 0 (baseline): 0 0 0 0  — 全部 bypass
    # 实验 1:            0 0 0 1  — 仅最小分辨率用 HOI
    # 实验 2:            0 0 1 1
    # 实验 3:            0 1 1 1
    # 实验 4 (full HOI): 1 1 1 1  — 全部启用 HOI
    HOI_CONFIG = [1, 1, 1, 1]  # <--- 修改这里！

    # --- HOI 交互阶数（1~6，默认 4）---
    HOI_ORDER = 3  # <--- 修改这里！

    # --- HOI 堆叠层数（默认 1）---
    HOI_LAYERS = 1  # <--- 修改这里！
    # ============================================================

    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD/WHU-CD/DSIFN-CD dataset")
    parser.add_argument('--cfg', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'vssm1', 'vssm_base_224.yaml'))
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, help='path to pretrained weights')
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/home/z/dataset/SYSU-CD/train')
    # parser.add_argument('--train_data_list_path', type=str, default='/home/z/dataset/SYSU-CD/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/home/z/dataset/SYSU-CD/test')
    parser.add_argument('--decoder_depths', type=int, default=4)
    # parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    # parser.add_argument('--train_data_name_list', type=list)
    # parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')
    parser.add_argument('--train_name', type=str, default=None, help='name for the training run, used for creating save directory')

    parser.add_argument('--hoi_levels', type=int, nargs='*', default=None,
                        help='HOI enable flags for 4 encoder levels (resolution high->low). '
                             '1=use HOI_Fusion_Adapter, 0=use Bypass_Fusion_Adapter. '
                             'If not provided, uses the hardcoded HOI_CONFIG at the top of main(). '
                              'Example: --hoi_levels 0 0 0 1')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-4)

    args = parser.parse_args()

    if args.hoi_levels is None:
        args.hoi_levels = HOI_CONFIG
    assert len(args.hoi_levels) == 4, f"hoi_levels must have 4 elements, got {len(args.hoi_levels)}: {args.hoi_levels}"

    def image_name_list_from_dir(dir_path):
        return sorted([x for x in os.listdir(dir_path) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    def find_list_file(dataset_path, split):
        search_dirs = [
            os.path.join(dataset_path, 'list'),
            os.path.join(os.path.dirname(dataset_path), 'list'),
            os.path.abspath(os.path.join(dataset_path, '..', 'list')),
        ]
        for d in search_dirs:
            path = os.path.join(d, f'{split}.txt')
            if os.path.exists(path):
                return path
        return None

    def load_name_list(dataset_path, split):
        list_file = find_list_file(dataset_path, split)
        if list_file is None:
            return None
        with open(list_file, 'r') as f:
            return [x.strip() for x in f.read().splitlines() if x.strip()]

    def resolve_split_dir(dataset_path, candidate_names):
        for name in candidate_names:
            dir_path = os.path.join(dataset_path, name)
            if os.path.exists(dir_path):
                return dir_path
        raise FileNotFoundError(f'Cannot resolve split directory under {dataset_path} with candidates {candidate_names}')

    if args.dataset=='LEVIR-CD' or args.dataset=='LEVIR-CD+':
        args.train_data_name_list = os.listdir(os.path.join(args.train_dataset_path,'A'))
        args.test_data_name_list = os.listdir(os.path.join(args.test_dataset_path,'A'))
    if args.dataset=='SYSU':
        args.train_data_name_list = load_name_list(args.train_dataset_path, 'train')
        args.test_data_name_list = load_name_list(args.test_dataset_path, 'test')
        if args.train_data_name_list is None:
            train_dir = resolve_split_dir(args.train_dataset_path, ['time1', 'A'])
            args.train_data_name_list = image_name_list_from_dir(train_dir)
        if args.test_data_name_list is None:
            test_dir = resolve_split_dir(args.test_dataset_path, ['time1', 'A'])
            args.test_data_name_list = image_name_list_from_dir(test_dir)
    if args.dataset=='DSIFN-CD':
        args.train_data_name_list = load_name_list(args.train_dataset_path, 'train')
        args.test_data_name_list = load_name_list(args.test_dataset_path, 'test')
        if args.train_data_name_list is None:
            train_dir = resolve_split_dir(args.train_dataset_path, ['t1', 't11', 'A'])
            args.train_data_name_list = image_name_list_from_dir(train_dir)
        if args.test_data_name_list is None:
            test_dir = resolve_split_dir(args.test_dataset_path, ['t1', 't11', 'A'])
            args.test_data_name_list = image_name_list_from_dir(test_dir)
    if args.dataset == 'WHU-CD':
        args.train_data_name_list = []
        args.test_data_name_list = []
        whu_train_list = find_list_file(args.train_dataset_path, 'train')
        whu_test_list = find_list_file(args.test_dataset_path, 'test')
        if whu_train_list is not None and whu_test_list is not None:
            with open(whu_test_list, 'r') as f_test:
                args.test_data_name_list = [x.strip() for x in f_test.read().splitlines() if x.strip()]
            with open(whu_train_list, 'r') as f_train:
                args.train_data_name_list = [x.strip() for x in f_train.read().splitlines() if x.strip()]
        else:
            args.train_data_name_list = image_name_list_from_dir(os.path.join(args.train_dataset_path, 'A'))
            args.test_data_name_list = image_name_list_from_dir(os.path.join(args.test_dataset_path, 'A'))

    trainer = Trainer(args, hoi_order=HOI_ORDER, hoi_layers=HOI_LAYERS)
    trainer.training()


if __name__ == "__main__":
    main()
