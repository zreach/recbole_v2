model=${1:-"DIN"}
gpu_id=${2:-"0"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=lfm1b-seq2 --config_files=configs/lfm1b-fil/seq/idonly-ctr-pop.yaml --model=$model --task_name=idonly-ctr-seq2 --gpu_id=$gpu_id   