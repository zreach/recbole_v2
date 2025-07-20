model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/aonly.yaml configs/settings/cold.yaml" --model=$model --task_name=aonly-cold --gpu_id=3 