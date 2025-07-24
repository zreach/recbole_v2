model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/settings/cold.yaml" --model=$model --task_name=idonly-cold --gpu_id=$gpu_id