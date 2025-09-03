model=${1:-"FM"}
gpu_id=${2:-"0"}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python run_recbole.py --dataset=m4a --config_files="configs/m4a/aonly-cluster.yaml" --model=$model --task_name=aonly-cluster --gpu_id=$gpu_id 