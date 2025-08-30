model=${1:-"FM"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/idonly.yaml configs/lfm1b-fil/audio.yaml" --model=$model --task_name=id-a