model=${1:-"FM"}
gpu_id=${2:-"0"}
proj_method=${3:-"linear"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly.yaml --model=$model --task_name=aonly --gpu_id=$gpu_id --proj_method=$proj_method