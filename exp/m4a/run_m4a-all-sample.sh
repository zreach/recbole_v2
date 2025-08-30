model=${1:-"FM"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files="configs/m4a/all.yaml configs/m4a/sample.yaml" --model=$model --task_name=all-sample --epochs=2 --gpu_id=$gpu_id