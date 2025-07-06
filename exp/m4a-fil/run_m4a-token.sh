model=${1:-"FM"}

python run_recbole.py --dataset=m4a-fil --config_files=configs/m4a-fil/token.yaml --model=$model --task_name=token --gpu_id=3

# python run_recbole.py --dataset=m4a-fil --config_files=configs/m4a-fil/token.yaml --model=AFN --task_name=token --gpu_id=3