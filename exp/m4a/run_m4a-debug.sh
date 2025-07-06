model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/debug.yaml --model=$model --task_name=debug --gpu_id=3