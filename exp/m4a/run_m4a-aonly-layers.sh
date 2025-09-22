model=${1:-"FM"}
gpu_id=${2:-"0"}
proj_method=${3:-"linear"}

for layer in {0..11}; do
    CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly.yaml --model=$model --task_name="aonly-layer${layer}" --gpu_id=$gpu_id --proj_method=$proj_method --afeat_layer=$layer
done
