model=${1:-"FM"}

CUDA_VISIBLE_DEVICES="3" python run_recbole.py --dataset=m4a --config_files=configs/m4a/muonly.yaml --model=$model --task_name=muonly