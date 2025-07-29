#!/bin/bash

# 定义要进行实验的模型列表
models=("BPR" "LightGCN" "CDAE" "MultiVAE" "ItemKNN" "RecVAE" "DiffRec") 

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
        --dataset=m4a-seq \
        --config_files="configs/m4a/seq/idonly-bpr.yaml" \
        --model=$model \
        --task_name=id-bpr &
    
    model_idx=$((model_idx + 1))
done

# 等待所有后台任务完成
wait

echo "All experiments finished."