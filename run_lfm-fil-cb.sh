model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-filtered --config_files=configs/lfm1b-fil-cb.yaml --model=$model
