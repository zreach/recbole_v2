model=${1:-"BPR"}

python run_recbole.py --dataset=m4a --config_files=configs/m4a/topk/id-a.yaml --model=$model --task_name=id-a-topk --gpu_id=3