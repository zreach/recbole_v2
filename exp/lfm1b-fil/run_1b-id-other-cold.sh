model=${1:-"FM"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=${gpu_id} python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/id-other.yaml configs/settings/cold.yaml" --model=$model --task_name=id-other-cold --gpu_id=$gpu_id
