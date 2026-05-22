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

tail -f sniper_dual.log

如果未指定 `GPU_ID`，脚本会自动扫描显卡并选择空闲显卡。

## SSH 连接
```bash
ssh z@10.59.85.10
cd /media/z/1d115f79-c4ea-4c62-9376-98541627908c/LH/Gra
```
