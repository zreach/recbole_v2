model=${1:-"DIN"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a-seq --config_files=configs/m4a/seq/idonly-ce.yaml --model=$model --task_name=idonly-ce --gpu_id=$gpu_id   