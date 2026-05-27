# MambaPyramid 变化检测训练指南

## 环境配置

- **PyTorch**: 2.10.0
- **CUDA**: 12.8
- **Python**: 3.10+

## SSH 连接

```bash
ssh z@10.59.85.10
cd /media/z/1d115f79-c4ea-4c62-9376-98541627908c/LH/Gra
conda activate lh
```

## 训练

使用 GPU 自动监控训练，脚本会自动扫描空闲 GPU 并启动训练，日志按数据集分别存放。

### 用法

```bash
nohup bash watch_gpu.sh <RUN_NAME> <DATASET> > logs/<RUN_NAME>.log 2>&1 &
```

**参数说明：**
- `RUN_NAME`: 训练名称（用于区分不同实验）
- `DATASET`: 数据集名称（LEVIR-CD / LEVIR-CD256 / SYSU / WHU-CD）
- 不指定 GPU_ID，脚本自动扫描 0/1 号显卡

### 示例

```bash
# LEVIR-CD 训练
nohup bash watch_gpu.sh levir_exp001 LEVIR-CD > logs/levir_exp001.log 2>&1 &

# LEVIR-CD256 训练
nohup bash watch_gpu.sh levircd256_exp001 LEVIR-CD256 > logs/levircd256_exp001.log 2>&1 &

# SYSU 训练
nohup bash watch_gpu.sh sysu_exp001 SYSU > logs/sysu_exp001.log 2>&1 &

# WHU-CD 训练
nohup bash watch_gpu.sh whu_exp001 WHU-CD > logs/whu_exp001.log 2>&1 &
```

### 同时训练多个任务

多个任务可以同时运行，脚本会自动分配不同的 GPU：

```bash
nohup bash watch_gpu.sh exp001 LEVIR-CD > logs/exp001_levir.log 2>&1 &
nohup bash watch_gpu.sh exp002 SYSU > logs/exp002_sysu.log 2>&1 &
```

### 日志位置

| 数据集 | 日志路径 |
|--------|----------|
| LEVIR-CD | `logs/{RUN_NAME}_levir.log` |
| LEVIR-CD256 | `logs/{RUN_NAME}_levir256.log` |
| SYSU | `logs/{RUN_NAME}_sysu.log` |
| WHU-CD | `logs/{RUN_NAME}_whu.log` |

### 查看日志

```bash
tail -f logs/levir_exp001.log
```

## 可视化与评估

推理脚本：`changedetection/script/visualize.py`

### 测试命令

```bash
# LEVIR-CD-1024
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/LEVIR-CD/{model_folder}/{iter}_model.pth' \
    --test_dataset_path '/home/z/dataset/LEVIR-CD-1024/test'

# LEVIR-CD256（用 LEVIR-CD-1024 训练的模型直接测试）
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/LEVIR-CD/{model_folder}/{iter}_model.pth' \
    --test_dataset_path '/home/z/dataset/LEVIR-CD256'

# SYSU-CD
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/SYSU/{model_folder}/{iter}_model.pth' \
    --test_dataset_path '/home/z/dataset/SYSU-CD/'

# WHU-CD
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/WHU-CD/{model_folder}/{iter}_model.pth' \
    --test_dataset_path '/home/z/dataset/WHU-CD-256/'
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--resume` | 训练好的 checkpoint `.pth` 文件路径 |
| `--test_dataset_path` | 测试数据根目录 |
| `--result_saved_path` | 输出根目录，默认 `./test_results` |
| `--decoder_depths` | Decoder 深度，默认 `4` |
| `--drop_rate` | Dropout 率，默认 `0.0` |

### 输出目录结构

```
test_results/
  {model_name}/
    {sample_name}/
      T1.png         # 原始时相1图像
      T2.png         # 原始时相2图像
      label.png      # 黑白二值标签（0=无变化，255=有变化）
      prediction.png # 黑白二值预测（0=无变化，255=有变化）
      color_map.png  # 彩色分析图
```

### color_map 颜色含义

| 颜色 | 含义 | 条件 |
|---|---|---|
| 白色 | TP（真正例） | 预测有变化，实际有变化 |
| 黑色 | TN（真负例） | 预测无变化，实际无变化 |
| 红色 | FP（假正例） | 预测有变化，实际无变化（误检） |
| 绿色 | FN（假负例） | 预测无变化，实际有变化（漏检） |

### 裁剪结果

将 1024×1024 的结果裁剪成 256×256 小块：

```bash
python cut_results.py test_results/<model_dir> 256
```

### 数据集目录结构要求

```
# LEVIR-CD-1024（train/test/val 分层）
dataset_root/
  test/
    A/      # 时相1
    B/      # 时相2
    label/  # 标签

# LEVIR-CD256 / SYSU-CD / WHU-CD（A/B/label 根目录 + list 划分）
dataset_root/
  A/
  B/
  label/
  list/
    test.txt
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
| LEVIR-CD | train/A/B/label/, test/A/B/label/ | 1024×1024 遥感变化检测 |
| LEVIR-CD256 | A/B/label/, list/test.txt | 256×256 裁剪版本 |
| SYSU | A/B/label/, list/train.txt, list/test.txt | 图像在根目录，list 文件区分训练/测试 |
| WHU-CD | A/B/label/, list/train.txt, list/test.txt | 图像在根目录，list 文件区分训练/测试 |
