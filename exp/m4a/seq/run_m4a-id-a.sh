model=${1:-"DIN"}
gpu_id=${2:-"0"}
proj_method=${3:-"mlp"}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# afeat_layer=${4:-"11"}

CUDA_VISIBLE_DEVICES=$gpu_id python run_recbole.py --dataset=m4a-seq3 --config_files="configs/m4a/seq/idonly-ctr.yaml configs/m4a/seq/audio.yaml" --model=$model --task_name=id-a-${proj_method} --gpu_id=$gpu_id --proj_method=$proj_method --train_batch_size=1024 --learning_rate=0.0005 --eval_batch_size=1024