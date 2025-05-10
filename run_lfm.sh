model=${1:-"FM"}

python run_recbole.py --model=FM --dataset=lfm-1b --config_files=configs/lfm1b.yaml
