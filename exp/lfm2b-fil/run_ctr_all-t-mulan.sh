#!/bin/bash

# 定义要进行实验的模型列表
models=("FM" "DeepFM" "AFM""DCNV2" "MaskNet" "EulerNet" "FinalMLP")

# 定义可用的GPU ID列表
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
model_idx=0

# 遍历模型列表并分配到不同的GPU上运行
for model in "${models[@]}"; do
    gpu_id=${gpus[$((model_idx % num_gpus))]}
    echo "Starting experiment for model: $model on GPU: $gpu_id"
    
    # 在后台运行实验
    CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py \
        --dataset=lfm2b-fil \
        --config_files="configs/lfm2b-fil/all.yaml configs/lfm2b-fil/mulan.yaml configs/lfm2b-fil/text.yaml" \
        --model=$model \
        --task_name=all-t-mulan &
    
    model_idx=$((model_idx + 1))
done

# 等待所有后台任务完成
wait

echo "All experiments finished."