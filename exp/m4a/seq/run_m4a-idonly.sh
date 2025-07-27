model=${1:-"DIN"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a-seq --config_files=configs/m4a/seq/idonly.yaml --model=$model --task_name=idonly --gpu_id=$gpu_id   