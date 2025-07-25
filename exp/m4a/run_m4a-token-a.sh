model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/token.yaml configs/m4a/audio.yaml"--model=$model --task_name=token-a --gpu_id=$gpu_id