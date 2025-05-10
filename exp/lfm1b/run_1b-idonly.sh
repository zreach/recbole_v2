model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-fil --config_files=configs/lfm1b/idonly.yaml --model=$model --task_name=idonly-1b
