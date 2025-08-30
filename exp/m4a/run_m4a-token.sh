model=${1:-"FM"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/token.yaml --model=$model --task_name=token --gpu_id=$gpu_id