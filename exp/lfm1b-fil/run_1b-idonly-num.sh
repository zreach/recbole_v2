model=${1:-"FM"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=${gpu_id} python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/idonly.yaml configs/settings/num.yaml" --model=$model --task_name=idonly-1b-5 --gpu_id=$gpu_id
