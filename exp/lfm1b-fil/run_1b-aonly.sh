model=${1:-"FM"}
gpu_id=${2:-"0"}
proj_method=${3:-"linear"}
afeat_layer=${4:-"11"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=lfm1b-fil --config_files=configs/lfm1b-fil/aonly.yaml --model=$model --task_name=aonly-${proj_method}-${afeat_layer} --gpu_id=$gpu_id --proj_method=$proj_method --afeat_layer=$afeat_layer