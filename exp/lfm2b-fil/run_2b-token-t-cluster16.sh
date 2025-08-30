model=${1:-"FM"}

python run_recbole.py --dataset=lfm2b-fil --config_files="configs/lfm2b-fil/token.yaml configs/lfm2b-fil/cluster16.yaml configs/lfm2b-fil/text.yaml" --model=$model --task_name=token-t-cluster16