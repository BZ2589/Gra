# MambaPyramid 变化检测训练指南

## 环境配置

- **PyTorch**: 2.10.0
- **CUDA**: 12.8
- **Python**: 3.10+

## 训练

使用 GPU 自动监控训练，脚本会自动扫描空闲 GPU 并启动训练，日志按数据集分别存放。

### 用法

```bash
nohup bash watch_gpu.sh <RUN_NAME> <DATASET> > logs/<RUN_NAME>.log 2>&1 &
```

**参数说明：**
- `RUN_NAME`: 训练名称（用于区分不同实验）
- `DATASET`: 数据集名称（LEVIR-CD / SYSU / WHU-CD）

### 示例

```bash
# LEVIR-CD 训练
nohup bash watch_gpu.sh levir_exp001 LEVIR-CD > logs/levir_exp001.log 2>&1 &

# SYSU 训练
nohup bash watch_gpu.sh sysu_base001 SYSU > logs/sysu_base001.log 2>&1 &

# WHU-CD 训练
nohup bash watch_gpu.sh whu_base001 WHU-CD > logs/whu_base001.log 2>&1 &
```

### 日志位置

| 数据集 | 日志路径 |
|--------|----------|
| LEVIR-CD | `logs/{RUN_NAME}_levir.log` |
| SYSU | `logs/{RUN_NAME}_sysu.log` |
| WHU-CD | `logs/{RUN_NAME}_whu.log` |

### 查看日志

```bash
tail -f logs/levir_exp001.log
```

## 可视化与评估

训练完成后，使用 `visualize.py` 对测试集进行推理和评估：

```bash
python changedetection/script/visualize.py \
    --resume <checkpoint_path> \
    --test_dataset_path <test_dataset_root>
```

**参数说明：**
- `--resume`: 模型 checkpoint 路径（.pth 文件）
- `--test_dataset_path`: 测试集根目录（需包含 A/、B/、label/ 或 time1/、time2/、label/ 子目录）
- `--decoder_depths`: Decoder 深度（默认 4）
- `--drop_rate`: Dropout 率（默认 0.0）

**示例：**

```bash
python changedetection/script/visualize.py \
    --resume changedetection/saved_models/LEVIR-CD/exp001/49000_model.pth \
    --test_dataset_path /home/z/dataset/LEVIR-CD-1024/test
```

**输出：**
- 每个样本独立文件夹，包含 T1.png、T2.png、label.png、prediction.png、color_map.png
- 评估指标：Recall、Precision、OA、F1、IoU、Kappa

### 裁剪结果

将 1024×1024 的结果裁剪成 256×256 小块：

```bash
python cut_results.py test_results/<model_dir> 256
```

## 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 8 | 训练批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--weight_decay` | 4e-4 | 权重衰减系数 |
| `--max_iters` | 800000 | 最大迭代次数 |
| `--crop_size` | 256 | 训练裁剪尺寸 |
| `--decoder_depths` | 4 | Decoder 深度 |

## 数据集支持

| 数据集 | 目录结构 | 说明 |
|--------|----------|------|
| LEVIR-CD | A/ B/ label/ | 1024×1024 遥感变化检测 |
| SYSU | time1/ time2/ label/ | 需配合 list/test.txt 使用 |
| WHU-CD | A/ B/ label/ list/ | 建筑物变化检测，需 list/train.txt |
