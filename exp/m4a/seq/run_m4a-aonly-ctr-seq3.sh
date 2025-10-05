model=${1:-"DIN"}
gpu_id=${2:-"0"}
proj_method=${3:-"mlp"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a-seq3 --config_files=configs/m4a/seq/aonly.yaml --model=$model --task_name=aonly-${proj_method} --proj_method=$proj_method --train_batch_size=1024 --learning_rate=0.0005 --eval_batch_size=1024 --gpu_id=$gpu_id   