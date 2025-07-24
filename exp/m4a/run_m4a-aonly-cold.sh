model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/aonly.yaml configs/settings/cold.yaml" --model=$model --task_name=aonly-cold --gpu_id=$gpu_id