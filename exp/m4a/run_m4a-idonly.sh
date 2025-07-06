model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/idonly.yaml --model=$model --task_name=idonly --gpu_id=3