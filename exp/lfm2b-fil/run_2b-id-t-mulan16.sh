model=${1:-"FM"}

python run_recbole.py --dataset=lfm2b-fil --config_files="configs/lfm2b-fil/idonly.yaml configs/lfm2b-fil/mulan16.yaml configs/lfm2b-fil/text.yaml" --model=$model --task_name=id-t-mulan16