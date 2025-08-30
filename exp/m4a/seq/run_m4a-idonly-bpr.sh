model=${1:-"BPR"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a-seq --config_files=configs/m4a/seq/idonly-bpr.yaml --model=$model --task_name=idonly-bpr --gpu_id=$gpu_id   