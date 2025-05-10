model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/id-a.yaml --model=$model --task_name=id-a