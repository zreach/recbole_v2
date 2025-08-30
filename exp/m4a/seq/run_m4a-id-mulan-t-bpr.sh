model=${1:-"BPR"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=m4a-seq --config_files="configs/m4a/seq/idonly-bpr.yaml configs/m4a/seq/mulan.yaml configs/m4a/seq/text.yaml" --model=$model --task_name=id-a-t-bpr --gpu_id=$gpu_id   