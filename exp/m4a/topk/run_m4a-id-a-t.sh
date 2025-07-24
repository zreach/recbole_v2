model=${1:-"BPR"}
gpu_id=${2:-0}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/topk/idonly.yaml configs/m4a/topk/audio.yaml configs/m4a/topk/text.yaml" --model=$model --task_name=id-a-t-topk --gpu_id=$gpu_id