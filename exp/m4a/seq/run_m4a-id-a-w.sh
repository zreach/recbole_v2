model=${1:-"DIN"}
gpu_id=${2:-"0"}


CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a-seq3 --config_files="configs/m4a/seq/idonly-ctr.yaml configs/m4a/seq/audio.yaml" --model=$model --task_name=id-a-w --gpu_id=$gpu_id --proj_method=mlp --afeat_layer=weighted_sum --train_batch_size=1024 --learning_rate=0.0005 --eval_batch_size=1024