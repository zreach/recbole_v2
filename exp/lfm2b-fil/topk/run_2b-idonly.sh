model=${1:-"BPR"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=lfm2b-fil --config_files=configs/lfm2b-fil/topk/idonly.yaml --model=$model --task_name=idonly-topk --gpu_id=$gpu_id