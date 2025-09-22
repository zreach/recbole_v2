model=${1:-"FM"}
gpu_id=${2:-"0"}
all_mlp_mode=${3:-"shared"}
all_output_mode=${4:-"mean"}


CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly-all.yaml --model=$model --task_name=aonly-all-${all_mlp_mode} --gpu_id=$gpu_id --all_mlp_mode=$all_mlp_mode --all_output_mode=$all_output_mode