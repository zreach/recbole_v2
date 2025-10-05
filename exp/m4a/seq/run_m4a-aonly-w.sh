model=${1:-"DIN"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a-seq3 --config_files=configs/m4a/seq/aonly-weighted.yaml --model=$model --task_name=aonly-w --gpu_id=$gpu_id   