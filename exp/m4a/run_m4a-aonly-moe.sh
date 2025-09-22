model=${1:-"FM"}
gpu_id=${2:-"0"}
gate_type=${3:-"both"}
topk=${4:-"1"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly-moe.yaml --model=$model --task_name=aonly-moe-${gate_type} --gpu_id=$gpu_id --moe_gate_type=$gate_type --moe_topk=$topk