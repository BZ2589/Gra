# 训练指南

本指南说明如何运行训练脚本来启动模型训练。当前已支持以下数据集：

* `LEVIR-CD`
* `SYSU-CD`
* `DSIFN-CD`
* `WHU-CD`

## 数据集训练脚本

训练脚本位于 `changedetection/script/run/` 目录下。

| 数据集 | 脚本 | 示例 |
| --- | --- | --- |
| LEVIR-CD | `train_levircd.sh` | `bash changedetection/script/run/train_levircd.sh levircd_base001` |
| LEVIR-CD256 | `train_levircd256.sh` | `bash changedetection/script/run/train_levircd256.sh levircd256_base001` |
| SYSU-CD | `train_sysu.sh` | `bash changedetection/script/run/train_sysu.sh sysu_base001` |
| DSIFN-CD | `train_dsifncd.sh` | `bash changedetection/script/run/train_dsifncd.sh dsifn_base001` |
| WHU-CD | `train_whucd.sh` | `bash changedetection/script/run/train_whucd.sh whucd_base001` |

每个训练脚本接受两个参数：

1. **`RUN_NAME` (必需)**：训练运行名称，用于创建模型保存目录和日志目录。
2. **`GPU_ID` (可选)**：要使用的 GPU 号，默认 `0`。

## 如何运行训练命令

请在项目根目录下执行以下命令来启动训练：

```bash
bash changedetection/script/run/train_sysu.sh <RUN_NAME> [GPU_ID]
```

**示例：**

* 在 `0` 号 GPU 上运行名为 `sysu_base001` 的 SYSU-CD 训练：
```bash
bash changedetection/script/run/train_sysu.sh sysu_base001
```

* 在 `1` 号 GPU 上运行名为 `dsifn_base001` 的 DSIFN-CD 训练：
```bash
bash changedetection/script/run/train_dsifncd.sh dsifn_base001 1
```

* 在 `0` 号 GPU 上运行名为 `whucd_base001` 的 WHU-CD 训练：
```bash
bash changedetection/script/run/train_whucd.sh whucd_base001
```

请确保在运行训练之前，所有必要的依赖项都已安装，并且数据集路径配置正确。
## 启动显卡监控脚本

`watch_gpu.sh` 现在支持选择数据集，命令格式：

```bash
nohup bash watch_gpu.sh <RUN_NAME> <DATASET> [GPU_ID] > sniper_dual.log 2>&1 &
```

**示例：**

```bash
nohup bash watch_gpu.sh myrun WHU-CD > sniper_dual.log 2>&1 &
```
nohup bash watch_gpu.sh whucd001 WHU-CD > sniper_dual.log 2>&1 &

nohup bash watch_gpu.sh levircd_traindata LEVIR-CD > sniper_dual.log 2>&1 &

tail -f sniper_dual.log

如果未指定 `GPU_ID`，脚本会自动扫描显卡并选择空闲显卡。

## SSH 连接
```bash
ssh z@10.59.85.10
cd /media/z/1d115f79-c4ea-4c62-9376-98541627908c/LH/Gra
conda activate lh
```

## 推理测试

推理脚本：`changedetection/script/visualize.py`



### 测试命令

```bash
# LEVIR-CD-1024
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/LEVIR-CD/baseline_base_levir_drop=0_levircd_hoi31/49000_model.pth' \
    --test_dataset_path '/home/z/dataset/LEVIR-CD-1024/test' \
    --hoi_order 3

# LEVIR-CD256
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/LEVIR-CD/baseline_base_levir_drop=0_levircd_hoi31/49000_model.pth' \
    --test_dataset_path '/home/z/dataset/LEVIR-CD256' \
    --hoi_order 3

# WHU-CD
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/WHU-CD/baseline_base_whu-ds-lvl-2layer-101_whucd001/68000_model.pth' \
    --test_dataset_path '/home/z/dataset/WHU-CD-256(tvt)/test' \
    --hoi_order 3

# SYSU-CD
python changedetection/script/visualize.py \
    --resume 'changedetection/saved_models/SYSU/baseline_base_sysu_baseline-res101-nods_sysucd001/35500_model.pth' \
    --test_dataset_path '/home/z/dataset/SYSU-CD/' \
    --hoi_order 3
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--resume` | 训练好的 checkpoint `.pth` 文件路径 |
| `--test_dataset_path` | 测试数据根目录（需包含 A/、B/、label/ 子目录） |
| `--result_saved_path` | 输出根目录，默认 `./test_results` |
| `--hoi_order` | HOI 交互阶数，默认 `3`（需与训练时一致） |
| `--hoi_levels` | HOI 启用层级，4 个数字对应 4 个分辨率层级（1=启用，0=禁用），默认全部启用 |
### 输出目录结构

```
test_results/
  {model_name}/
    {sample_name}/
      T1.png        # 原始时相1图像
      T2.png        # 原始时相2图像
      label.png     # 黑白二值标签（0=无变化，255=有变化）
      prediction.png # 黑白二值预测（0=无变化，255=有变化）
      color_map.png # 彩色分析图
```

### color_map 颜色含义

| 颜色 | 含义 | 条件 |
|---|---|---|
| 白色 | TP（真正例） | 预测有变化，实际有变化 |
| 黑色 | TN（真负例） | 预测无变化，实际无变化 |
| 红色 | FP（假正例） | 预测有变化，实际无变化（误检） |
| 绿色 | FN（假负例） | 预测无变化，实际有变化（漏检） |
### 数据集目录结构要求

```
dataset_root/
  A/      # 时相1
  B/      # 时相2
  label/  # 标签
```

### 数据集目录结构要求

```
# LEVIR-CD-1024 / WHU-CD-256（train/test/val 分层）
dataset_root/
  test/
    A/      # 时相1
    B/      # 时相2
    label/  # 标签

# LEVIR-CD256（A/B/label 根目录 + list 划分）
dataset_root/
  A/
  B/
  label/
  list/
    test.txt

# SYSU-CD（A/B/label 根目录 + list 划分）
dataset_root/
  A/
  B/
  label/
  list/
    test.txt

# DSIFN-CD
dataset_root/
  im1/ 或 t1/  # 时相1
  im2/ 或 t2/  # 时相2
  label/ 或 mask/  # 标签
  list/
    test.txt
```

