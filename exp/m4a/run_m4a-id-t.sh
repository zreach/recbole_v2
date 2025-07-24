model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/audio.yaml configs/m4a/text.yaml" --model=$model --task_name=id-t --gpu_id=$gpu_id