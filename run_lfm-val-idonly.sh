model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-valid --config_files=configs/lfm1b-val-idonly.yaml --model=$model
