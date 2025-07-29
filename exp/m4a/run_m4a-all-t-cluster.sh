model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/all.yaml configs/m4a/audio.yaml configs/m4a/text.yaml configs/m4a/cluster.yaml" --model=$model --task_name=all-t-cluster --gpu_id=$gpu_id 