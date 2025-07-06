model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=lfm1b-tiny --config_files=configs/lfm1b-tiny/idonly.yaml --model=$model --task_name=idonly-1b --gpu_id=$gpu_id

