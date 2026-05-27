# MambaPyramid 变化检测训练指南

## 目录结构

```
Gra/
├── changedetection/
│   ├── configs/          # 模型配置文件
│   ├── datasets/         # 数据集加载器
│   ├── models/           # 模型定义（MambaPyramid）
│   ├── script/
│   │   ├── train_MambaBCD.py   # 训练脚本
│   │   ├── infer_MambaBCD.py   # 推理脚本
│   │   ├── visualize.py        # 可视化与评估脚本
│   │   └── run/                # 训练/测试 shell 脚本
│   ├── utils_func/       # 工具函数（评估指标等）
│   └── saved_models/     # 模型保存路径
├── watch_gpu.sh          # GPU 监控自动训练脚本
└── README.md
```

## 环境配置

- **PyTorch**: 2.10.0
- **CUDA**: 12.8
- **Python**: 3.10+

## 训练

### 方式一：直接运行训练脚本

```bash
python changedetection/script/train_MambaBCD.py \
    --dataset LEVIR-CD \
    --train_dataset_path /path/to/LEVIR-CD/train \
    --test_dataset_path /path/to/LEVIR-CD/test \
    --model_param_path changedetection/saved_models \
    --train_name my_experiment
```

### 方式二：使用 shell 脚本（推荐）

在 `changedetection/script/run/` 目录下提供了各数据集的训练脚本：

```bash
# LEVIR-CD
bash changedetection/script/run/train_levircd.sh <RUN_NAME> [GPU_ID]

# SYSU-CD
bash changedetection/script/run/train_sysu.sh <RUN_NAME> [GPU_ID]

# WHU-CD
bash changedetection/script/run/train_whucd.sh <RUN_NAME> [GPU_ID]

# DSIFN-CD
bash changedetection/script/run/train_dsifncd.sh <RUN_NAME> [GPU_ID]
```

**示例：**

```bash
# 在 0 号 GPU 上训练 LEVIR-CD
bash changedetection/script/run/train_levircd.sh levircd_exp001

# 在 1 号 GPU 上训练 SYSU-CD
bash changedetection/script/run/train_sysu.sh sysu_exp001 1
```

### 方式三：GPU 自动监控训练

使用 `watch_gpu.sh` 监控 GPU 空闲状态，自动启动训练：

```bash
nohup bash watch_gpu.sh <RUN_NAME> <DATASET> [GPU_ID] > train.log 2>&1 &
```

**参数说明：**
- `RUN_NAME`: 训练名称
- `DATASET`: 数据集名称（LEVIR-CD / SYSU-CD / WHU-CD / DSIFN-CD）
- `GPU_ID`: 指定 GPU（可选，默认扫描 0/1）

**示例：**

```bash
nohup bash watch_gpu.sh myrun LEVIR-CD > train.log 2>&1 &
```

## 可视化与评估

使用 `visualize.py` 对测试集进行推理、可视化并计算评估指标：

```bash
python changedetection/script/visualize.py \
    --resume <checkpoint_path> \
    --test_dataset_path <test_dataset_root>
```

**参数说明：**
- `--resume`: 模型 checkpoint 路径（.pth 文件）
- `--test_dataset_path`: 测试集根目录（需包含 A/、B/、label/ 子目录）
- `--result_saved_path`: 结果保存路径（默认 ./test_results）
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
- 终端输出评估指标：Recall、Precision、OA、F1、IoU、Kappa

## 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 16 | 训练批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--weight_decay` | 4e-4 | 权重衰减系数 |
| `--max_iters` | 240000 | 最大迭代次数 |
| `--crop_size` | 256 | 训练裁剪尺寸 |
| `--drop_rate` | 0.2 | Dropout 率 |
| `--decoder_depths` | 4 | Decoder 深度 |

## 数据集支持

| 数据集 | 目录结构 | 说明 |
|--------|----------|------|
| LEVIR-CD | A/ B/ label/ | 1024×1024 遥感变化检测 |
| SYSU-CD | time1/ time2/ label/ | 需配合 list/test.txt 使用 |
| WHU-CD | A/ B/ label/ list/ | 建筑物变化检测，需 list/train.txt |
| DSIFN-CD | t1/ t2/ mask/ | 支持多级 mask 目录 |
