# 训练指南

本指南将说明如何运行 `train_levircd.sh` 脚本来启动模型训练。

## 训练脚本 `train_levircd.sh`

`train_levircd.sh` 脚本位于 `changedetection/script/run/` 目录下，用于启动 LEVIR-CD 数据集的训练过程。该脚本接受两个参数：

1.  **`RUN_NAME` (必需)**：训练运行的名称。这个名称将用于创建日志和保存模型的目录，以便于区分不同的训练实验。例如：`levircd_base001`。
2.  **`GPU_ID` (可选)**：指定用于训练的 GPU 设备 ID。如果你有多个 GPU，可以通过这个参数选择使用哪一个。如果未提供此参数，脚本将默认使用 `0` 号 GPU。

### 如何运行训练命令

请在项目根目录下执行以下命令来启动训练：

```bash
bash changedetection/script/run/train_levircd.sh <RUN_NAME> [GPU_ID]
```

**示例：**

*   **在 `0` 号 GPU 上运行名为 `levircd_base001` 的训练：**
    ```bash
bash changedetection/script/run/train_levircd.sh levircd_base001
    ```

*   **在 `1` 号 GPU 上运行名为 `levircd_base002` 的训练：**
    ```bash
bash changedetection/script/run/train_levircd.sh levircd_base002 1
    ```

请确保在运行训练之前，所有必要的依赖项都已安装，并且数据集路径配置正确。
