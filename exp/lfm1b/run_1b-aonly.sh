model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-fil --config_files=configs/lfm1b/aonly.yaml --model=$model --task_name=aonly