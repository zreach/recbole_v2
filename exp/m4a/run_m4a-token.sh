model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/token.yaml --model=$model --task_name=token --gpu_id=3