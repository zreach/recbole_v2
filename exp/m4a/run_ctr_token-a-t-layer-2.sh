#!/bin/bash

# 定义要进行实验的模型列表
SCRIPT_DIR=$(dirname "$0")
models=("FM" "DSSM" "WideDeep" "NFM" "DeepFM" "AFM" "xDeepFM" "DCN" "DCNV2" "AutoInt" "MaskNet" "EulerNet" "FinalMLP" "WuKong")

# 定义可用的GPU ID列表
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
model_idx=0

# 遍历模型列表并分配到不同的GPU上运行
for model in "${models[@]}"; do
    gpu_id=${gpus[$((model_idx % num_gpus))]}
    echo "Starting experiment for model: $model on GPU: $gpu_id"
    
    bash "$SCRIPT_DIR/run_m4a-token-a-t-layer.sh" $model $gpu_id &
    
    model_idx=$((model_idx + 1))
done

# 等待所有后台任务完成
wait

echo "All experiments finished."