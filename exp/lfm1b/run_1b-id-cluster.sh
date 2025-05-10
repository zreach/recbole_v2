model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-fil --config_files=configs/lfm1b/id-cluster.yaml --model=$model --task_name=id-cluster