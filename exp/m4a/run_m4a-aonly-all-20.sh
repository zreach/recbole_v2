model=${1:-"FM"}
gpu_id=${2:-"0"}


CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly-all.yaml --model=$model --task_name=aonly-all-20 --gpu_id=$gpu_id --embedding_size=20