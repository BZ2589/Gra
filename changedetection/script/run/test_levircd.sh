python script/infer_MambaBCD.py  --dataset 'LEVIR-CD' \
                                 --model_type 'baseline_tiny_baseline' \
                                 --test_dataset_path './data/LEVIR-CD/test' \
                                 --decoder_depths 4 \
                                 --if_visible 'gray' \
                                 --cfg './changedetection/configs/vssm1/vssm_small_224.yaml' \
                                 --resume './changedetection/changedetection/saved_models/LEVIR-CD/baseline_tiny_baseline_1725679764.1621282/45500_model.pth' \