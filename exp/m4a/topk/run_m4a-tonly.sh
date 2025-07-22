model=${1:-"BPR"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/topk/tonly.yaml --model=$model --task_name=tonly-topk