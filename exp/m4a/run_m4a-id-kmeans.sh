model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/cluster.yaml" --model=$model --task_name=id-k --gpu_id=$gpu_id

# python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/audio.yaml" --model=DeepFM --task_name=id-a