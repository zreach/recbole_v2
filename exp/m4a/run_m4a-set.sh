model=${1:-"FM"}

CUDA_VISIBLE_DEVICES="0" python run_recbole.py --dataset=m4a --config_files=configs/m4a/a-token.yaml --model=$model --task_name=a-token &
CUDA_VISIBLE_DEVICES="1" python run_recbole.py --dataset=m4a --config_files=configs/m4a/idonly.yaml --model=$model --task_name=idonly &
CUDA_VISIBLE_DEVICES="2" python run_recbole.py --dataset=m4a --config_files=configs/m4a/aonly.yaml --model=$model --task_name=aonly 