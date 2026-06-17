# 毕业论文代码仓库

本仓库为毕业论文代码与实验脚本集合，包含遥感变化检测（changedetection）和图像分类（classification）等模块。

**仓库结构（重点）**
- **changedetection/**: 变化检测代码、训练/推理脚本与模型权重目录。
- **classification/**: 基于 Swin-Transformer 的分类训练代码（改自 Microsoft Swin-Transformer）。
- **requirements.txt**: Python 依赖清单。
- **workflow.md**: 常用训练/验证/监控/推理命令集合（建议优先阅读）。

**环境与依赖**
- 推荐使用 conda 管理环境。

1. 创建并激活环境（示例）：

```
conda create -n lh python=3.10 -y
conda activate lh
```

2. 安装依赖：

```
pip install -r requirements.txt
```

注意：`requirements.txt` 中指定了 `torch==2.0.1`、`torchvision==0.15.2` 等。若你的 GPU / CUDA 版本不同，请根据 PyTorch 官方安装说明调整 `torch` 的安装命令。

**快速开始 — 变化检测（changedetection）**
- 训练脚本位于 [changedetection/script/run](changedetection/script/run)。更多示例和参数见 [workflow.md](workflow.md)。

主要训练命令格式：

```
bash changedetection/script/run/train_<dataset>.sh <RUN_NAME> [GPU_ID]
```

示例：

```
bash changedetection/script/run/train_sysu.sh sysu_base001
bash changedetection/script/run_train_dsifncd.sh dsifn_base001 1
```

- 显卡监控（后台运行）：

```
nohup bash watch_gpu.sh <RUN_NAME> <DATASET> [GPU_ID] > sniper_dual.log 2>&1 &
tail -f sniper_dual.log
```

**推理 / 可视化**
- 推理脚本：`changedetection/script/visualize.py`。示例（从 `workflow.md` 复制并略作整理）：

```
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/LEVIR-CD/.../49000_model.pth' \
    --test_dataset_path '/path/to/LEVIR-CD-1024/test' \
    --hoi_order 3
```

输出结果会保存到 `test_results/` 下，目录结构示例见 [workflow.md](workflow.md) 中的说明。

**数据集目录格式要求**
- 普通（例如 WHU-CD-256 / LEVIR-CD-1024）：

```
dataset_root/
  train/
    A/
    B/
    label/
  val/
  test/
    A/
    B/
    label/
```

- 有些数据集（如 LEVIR-CD256、SYSU-CD、DSIFN-CD）使用根目录 + `list/` 划分，请参考 [workflow.md](workflow.md) 中的具体说明。

**分类模块（classification）**
- 入口脚本：[classification/main.py](classification/main.py)。该部分基于 Microsoft Swin-Transformer 改写。配置通过 `--cfg` 指定配置文件。

示例运行：

```
python classification/main.py --cfg path/to/config.yaml --data-path /path/to/dataset --batch-size 32
```

（具体配置文件与可用选项请参考 `classification` 目录下的配置文件与 `main.py` 的 `--cfg`/`--opts` 参数。）

**常用路径**
- 训练输出与日志：默认保存在各模块的 `output` 或 `changedetection/saved_models/` 下。
- 预训练/保存模型：`changedetection/saved_models/`。
- 推理结果：`test_results/`。

**SSH / 远程运行示例**

```
ssh user@host
cd /path/to/Gra
conda activate lh
nohup bash watch_gpu.sh myrun WHU-CD > sniper_dual.log 2>&1 &
```

**开发与调试建议**
- 先在小数据集或更低分辨率上跑通训练脚本，确认数据路径与依赖无误后再放大规模训练。
- 使用 `--resume` 参数加速调试；使用 `--eval` 仅做验证。

**参考文件**
- 训练与推理命令汇总见 [workflow.md](workflow.md)。
- 依赖清单见 [requirements.txt](requirements.txt)。
- 分类模块入口：[classification/main.py](classification/main.py)。
