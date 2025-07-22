model=${1:-"FM"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/token.yaml configs/m4a/cluster.yaml" --model=$model --task_name=token-k 

# python run_recbole.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/audio.yaml" --model=DeepFM --task_name=id-a