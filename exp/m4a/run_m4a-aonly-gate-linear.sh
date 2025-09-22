model=${1:-"FM"}
gpu_id=${2:-"0"}
gate_mode=${3:-"shared"}
gate_input_type=${4:-"mean_feature"}


CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly-gate.yaml --model=$model --task_name=aonly-gate-linear-${gate_mode}-${gate_input_type} --gpu_id=$gpu_id --gate_mode=$gate_mode --gate_input_type=$gate_input_type --wav_mlp_sizes=[]