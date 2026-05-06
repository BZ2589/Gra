#!/bin/bash

# 检查是否传入了训练名称参数
if [ -z "$1" ]; then
    echo "❌ 错误：请提供训练名称！"
    echo "💡 用法：bash watch_gpu.sh <训练名称>"
    echo "💡 示例：nohup bash watch_gpu.sh levircd_optimize006 > sniper_dual.log 2>&1 &"
    exit 1
fi

TRAIN_NAME=$1  # 将第一个参数赋值给变量
CHECK_INTERVAL=120  # 每 120 秒扫一次

echo "🎯 目标训练名称：$TRAIN_NAME"
echo "⏳ 开始空闲显卡扫描，每 $CHECK_INTERVAL 秒扫一次..."

while true; do
    # 用 for 循环依次检查 0 和 1 号显卡
    for GPU_ID in 0 1; do
        PROCESSES=$(nvidia-smi -i $GPU_ID --query-compute-apps=name --format=csv,noheader)
        
        # 判断：如果输出里没有 "python"
        if ! echo "$PROCESSES" | grep -iq "python"; then
            echo "[$(date +'%m-%d %H:%M:%S')] : 🎉 显卡 $GPU_ID 现已空闲！"
            echo "🚀 正在为您强占显卡 $GPU_ID 并启动训练: $TRAIN_NAME ..."
            
            # 自动把 $TRAIN_NAME 和 $GPU_ID 传进去
            bash changedetection/script/run/train_levircd.sh $TRAIN_NAME $GPU_ID
            
            echo "✅ 训练 [$TRAIN_NAME] 已在 GPU $GPU_ID 启动，收工。"
            exit 0 # 任务完成，直接结束整个监控脚本
        fi
    done
    
    echo "[$(date +'%m-%d %H:%M:%S')] 显卡 0 和 1 都还被占着，继续监控..."
    sleep $CHECK_INTERVAL
done