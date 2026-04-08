#!/bin/bash
GPU_ID=${2:-0} # Use the second argument as GPU_ID, default to 0 if not provided
RUN_NAME=${1:-'default_run'} # Use the first argument as RUN_NAME, default to 'default_run' if not provided

CUDA_VISIBLE_DEVICES=$GPU_ID python changedetection/script/train_MambaBCD.py --dataset 'LEVIR-CD' \
                                --batch_size 4 \
                                --crop_size 256 \
                                --max_iters 800000 \
                                --model_type baseline_base_levir_drop=0 \
                                --model_param_path 'changedetection/saved_models' \
                                --train_dataset_path '/home/z/dataset/LEVIR-CD-1024/train' \
                                --test_dataset_path '/home/z/dataset/LEVIR-CD-1024/test' \
                                --decoder_depth 4 \
                                --cfg './changedetection/configs/vssm1/vssm_base_224.yaml' \
                                --pretrained_weight_path './changedetection/pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth' \
                                --train_name "$RUN_NAME"