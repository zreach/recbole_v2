model=${1:-"FM"}

python run_recbole.py --dataset=lfm2b-fil --config_files="configs/lfm2b-fil/idonly.yaml configs/lfm2b-fil/audio.yaml" --model=$model --task_name=ida