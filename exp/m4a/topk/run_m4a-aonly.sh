model=${1:-"BPR"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/topk/aonly.yaml --model=$model --task_name=aonly-topk --gpu_id=3