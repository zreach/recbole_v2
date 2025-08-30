model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/token.yaml configs/m4a/cluster.yaml configs/m4a/text.yaml" --model=$model --task_name=token-t-cluster --gpu_id=$gpu_id