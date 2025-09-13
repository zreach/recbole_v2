model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=Amazon_Books --config_files=configs/books/idonly.yaml --model=$model --task_name=idonly --gpu_id=$gpu_id   