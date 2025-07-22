model=${1:-"FM"}

python run_recbole.py --dataset=m4a-fil --config_files=configs/m4a-fil/topk/aonly.yaml --model=$model --task_name=aonly-topk --gpu_id=3