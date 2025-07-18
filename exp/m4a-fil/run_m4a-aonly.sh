model=${1:-"FM"}

python run_recbole.py --dataset=m4a-fil --config_files=configs/m4a-fil/aonly.yaml --model=$model --task_name=aonly --gpu_id=3
# python run_recbole.py --dataset=m4a-fil --config_files=configs/m4a-fil/aonly.yaml --model=FM --task_name=aonly --params_file=hyper-layer.test
