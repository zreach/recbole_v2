model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/a-t.yaml --model=$model --task_name=a-t --gpu_id=$gpu_id