#!/bin/bash

# GPU 自动监控训练脚本
# 用法：nohup bash watch_gpu.sh <RUN_NAME> <DATASET> > logs/<RUN_NAME>.log 2>&1 &
# 示例：nohup bash watch_gpu.sh exp001 LEVIR-CD > logs/exp001_levir.log 2>&1 &

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "❌ 错误：请提供训练名称和数据集！"
    echo "💡 用法：nohup bash watch_gpu.sh <RUN_NAME> <DATASET> > logs/<RUN_NAME>.log 2>&1 &"
    echo "💡 支持数据集：LEVIR-CD, SYSU, WHU-CD"
    echo "💡 示例：nohup bash watch_gpu.sh exp001 LEVIR-CD > logs/exp001_levir.log 2>&1 &"
    exit 1
fi

TRAIN_NAME=$1
DATASET=$2
CHECK_INTERVAL=120  # 每 120 秒扫一次

# 根据数据集选择训练脚本和日志前缀
case "$DATASET" in
    LEVIR-CD|LEVIR-CD+)
        SCRIPT="changedetection/script/run/train_levircd.sh"
        LOG_PREFIX="levir"
        ;;
    SYSU|SYSU-CD)
        SCRIPT="changedetection/script/run/train_sysu.sh"
        LOG_PREFIX="sysu"
        ;;
    WHU-CD)
        SCRIPT="changedetection/script/run/train_whucd.sh"
        LOG_PREFIX="whu"
        ;;
    *)
        echo "❌ 错误：不支持的数据集 '$DATASET'。支持 LEVIR-CD、SYSU、WHU-CD。"
        exit 1
        ;;
esac

# 确保 logs 目录存在
mkdir -p logs
LOG_FILE="logs/${TRAIN_NAME}_${LOG_PREFIX}.log"

echo "🎯 训练名称：$TRAIN_NAME"
echo "📊 数据集：$DATASET"
echo "📝 日志文件：$LOG_FILE"
echo "⏳ 开始空闲显卡扫描，每 $CHECK_INTERVAL 秒扫一次..."

while true; do
    # 自动扫描 0 和 1 号显卡
    for FREE_GPU_ID in 0 1; do
        PROCESSES=$(nvidia-smi -i $FREE_GPU_ID --query-compute-apps=name --format=csv,noheader 2>/dev/null)

        # 判断：如果输出里没有 "python"
        if ! echo "$PROCESSES" | grep -iq "python"; then
            echo "[$(date +'%m-%d %H:%M:%S')] 🎉 显卡 $FREE_GPU_ID 现已空闲！"
            echo "[$(date +'%m-%d %H:%M:%S')] 🚀 在 GPU $FREE_GPU_ID 启动训练: $TRAIN_NAME ($DATASET)"
            echo "[$(date +'%m-%d %H:%M:%S')] 📝 日志输出到: $LOG_FILE"

            # 通过环境变量指定 GPU，日志重定向到各自文件
            CUDA_VISIBLE_DEVICES=$FREE_GPU_ID bash "$SCRIPT" "$TRAIN_NAME" > "$LOG_FILE" 2>&1

            echo "[$(date +'%m-%d %H:%M:%S')] ✅ 训练 [$TRAIN_NAME] 已完成，日志保存在 $LOG_FILE"
            exit 0
        fi
    done

    echo "[$(date +'%m-%d %H:%M:%S')] 显卡繁忙，${CHECK_INTERVAL}秒后重试..."
    sleep $CHECK_INTERVAL
done
