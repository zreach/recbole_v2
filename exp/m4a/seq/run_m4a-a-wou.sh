model=${1:-"DIN"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a-seq3 --config_files=configs/m4a/seq/a-wou.yaml --model=$model --task_name=a-wou --gpu_id=$gpu_id   