#!/bin/bash

# 检查是否传入了训练名称参数
if [ -z "$1" ]; then
    echo "❌ 错误：请提供训练名称！"
    echo "💡 用法：bash watch_gpu.sh <训练名称> <DATASET> [GPU_ID]"
    echo "💡 示例：nohup bash watch_gpu.sh myrun WHU-CD > sniper_dual.log 2>&1 &"
    exit 1
fi

TRAIN_NAME=$1
DATASET=${2:-LEVIR-CD}
PREFERRED_GPU_ID=${3:-}
CHECK_INTERVAL=120  # 每 120 秒扫一次

echo "🎯 目标训练名称：$TRAIN_NAME"
echo "⏳ 开始空闲显卡扫描，每 $CHECK_INTERVAL 秒扫一次..."

while true; do
    # 用 for 循环依次检查 0 和 1 号显卡
    if [ -n "$PREFERRED_GPU_ID" ]; then
        GPUS="$PREFERRED_GPU_ID"
    else
        GPUS="0 1"
    fi

    for FREE_GPU_ID in $GPUS; do
        PROCESSES=$(nvidia-smi -i $FREE_GPU_ID --query-compute-apps=name --format=csv,noheader)

        # 判断：如果输出里没有 "python"
        if ! echo "$PROCESSES" | grep -iq "python"; then
            echo "[$(date +'%m-%d %H:%M:%S')] : 🎉 显卡 $FREE_GPU_ID 现已空闲！"
            echo "🚀 正在为您强占显卡 $FREE_GPU_ID 并启动训练: $TRAIN_NAME ($DATASET) ..."

            case "$DATASET" in
                LEVIR-CD|LEVIR-CD+)
                    SCRIPT="changedetection/script/run/train_levircd.sh"
                    ;;
                SYSU|SYSU-CD)
                    SCRIPT="changedetection/script/run/train_sysu.sh"
                    ;;
                DSIFN-CD|DSUFN-CD)
                    SCRIPT="changedetection/script/run/train_dsifncd.sh"
                    ;;
                WHU-CD)
                    SCRIPT="changedetection/script/run/train_whucd.sh"
                    ;;
                *)
                    echo "❌ 错误：不支持的数据集 '$DATASET'。支持 LEVIR-CD、SYSU-CD、DSIFN-CD、WHU-CD。"
                    exit 1
                    ;;
            esac

            bash "$SCRIPT" "$TRAIN_NAME" "$FREE_GPU_ID"

            echo "✅ 训练 [$TRAIN_NAME] 已在 GPU $FREE_GPU_ID 启动，收工。"
            exit 0 # 任务完成，直接结束整个监控脚本
        fi
    done

    echo "[$(date +'%m-%d %H:%M:%S')] 显卡 0 和 1 都还被占着，继续监控..."
    sleep $CHECK_INTERVAL
done