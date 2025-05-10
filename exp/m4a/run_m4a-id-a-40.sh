model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/id-a-40.yaml --model=$model --task_name=id-a-40