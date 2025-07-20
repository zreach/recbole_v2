model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/audio.yaml configs/m4a/text.yaml" --model=$model --task_name=id-a-t --gpu_id=3