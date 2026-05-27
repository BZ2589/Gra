#!/bin/bash
GPU_ID=${2:-0}
RUN_NAME=${1:-'default_run'}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU_ID python changedetection/script/train_MambaBCD.py --dataset 'LEVIR-CD' \
                                --batch_size 8 \
                                --crop_size 256 \
                                --max_iters 800000 \
                                --model_type baseline_base_levir256_drop=0 \
                                --model_param_path 'changedetection/saved_models' \
                                --train_dataset_path '/home/z/dataset/LEVIR-CD256' \
                                --test_dataset_path '/home/z/dataset/LEVIR-CD256' \
                                --decoder_depth 4 \
                                --cfg './changedetection/configs/vssm1/vssm_base_224.yaml' \
                                --pretrained_weight_path './changedetection/pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth' \
                                --train_name "$RUN_NAME"
