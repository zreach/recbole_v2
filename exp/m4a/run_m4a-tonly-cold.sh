model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/tonly.yaml configs/settings/cold.yaml" --model=$model --task_name=tonly-cold --gpu_id=3 