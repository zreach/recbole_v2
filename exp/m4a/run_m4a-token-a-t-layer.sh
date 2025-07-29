model=${1:-"FM"}
gpu_id=${2:-"0"}

layers=(1 2 3 4 5 6 7 8 9 10 11 mean)

for layer in "${layers[@]}"; do
    echo "Running model: $model, layer: $layer"
    
    # 在后台运行实验
    CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py \
        --dataset=m4a \
        --config_files="configs/m4a/token.yaml configs/m4a/audio.yaml configs/m4a/text.yaml" \
        --model=$model \
        --task_name=token-a-t-$layer \
        --afeat_layer=$layer \
        --epochs=20

done
