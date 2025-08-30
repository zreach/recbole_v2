model=${1:-"BPR"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/topk/idonly.yaml --model=$model --task_name=idonly-topk --gpu_id=3