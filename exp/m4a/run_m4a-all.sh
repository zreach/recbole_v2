model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/all.yaml --model=$model --task_name=all --gpu_id=3