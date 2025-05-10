model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-valid --config_files=configs/lfm1b-val-20-idonly.yaml --model=$model
