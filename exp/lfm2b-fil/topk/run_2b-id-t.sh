model=${1:-"BPR"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=lfm2b-fil --config_files="configs/lfm2b-fil/topk/idonly.yaml configs/lfm2b-fil/topk/text.yaml" --model=$model --task_name=id-t-topk --gpu_id=$gpu_id