model=${1:-"FM"}

python run_recbole.py --dataset=lfm2b-fil --config_files=configs/lfm2b-fil/idonly.yaml --model=$model --task_name=idonly 