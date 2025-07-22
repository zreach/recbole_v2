model=${1:-"BPR"}

python run_recbole.py --dataset=m4a --config_files="configs/m4a/topk/idonly.yaml configs/m4a/topk/audio.yaml" --model=$model --task_name=id-a-topk 