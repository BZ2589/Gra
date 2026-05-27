#!/bin/bash
# SYSU-CD 训练脚本
# 由 watch_gpu.sh 调用，GPU 由 watch_gpu.sh 通过 CUDA_VISIBLE_DEVICES 设置

RUN_NAME=${1:-'default_run'}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python changedetection/script/train_MambaBCD.py --dataset 'SYSU' \
                                --batch_size 8 \
                                --crop_size 256 \
                                --max_iters 400000 \
                                --model_type baseline_base_sysu \
                                --model_param_path 'changedetection/saved_models' \
                                --train_dataset_path '/home/z/dataset/SYSU-CD' \
                                --test_dataset_path '/home/z/dataset/SYSU-CD' \
                                --cfg './changedetection/configs/vssm1/vssm_base_224.yaml' \
                                --decoder_depths 4 \
                                --pretrained_weight_path './changedetection/pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth' \
                                --train_name "$RUN_NAME"
