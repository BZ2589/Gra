#!/bin/bash

CHECK_INTERVAL=120  # 每 120 秒扫一次

echo "开始空闲显卡扫描，每 $CHECK_INTERVAL 秒扫一次..."

while true; do
    # 用 for 循环依次检查 0 和 1 号显卡
    for GPU_ID in 0 1; do
        PROCESSES=$(nvidia-smi -i $GPU_ID --query-compute-apps=name --format=csv,noheader)
        
        # 判断：如果输出里没有 "python"
        if ! echo "$PROCESSES" | grep -iq "python"; then
            echo "[$(date +'%m-%d %H:%M:%S')] : 显卡 $GPU_ID 现已空闲！"
            echo "正在为您强占显卡 $GPU_ID 并启动训练..."
            
            # 自动把 $GPU_ID 传给你写的 x 的位置
            bash changedetection/script/run/train_levircd.sh levircd_optimize003 $GPU_ID
            
            echo "训练已在 GPU $GPU_ID 启动，收工。"
            exit 0 # 任务完成，直接结束整个监控脚本
        fi
    done
    
    echo "[$(date +'%m-%d %H:%M:%S')] 显卡 0 和 1 都还被占着，继续监控..."
    sleep $CHECK_INTERVAL
done