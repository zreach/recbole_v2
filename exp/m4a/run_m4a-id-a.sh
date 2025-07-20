model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/audio.yaml" --model=$model --task_name=id-a --gpu_id=3