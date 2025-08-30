CUDA_VISIBLE_DEVICES=0 python run_recbole.py --dataset=lfm1b-fil --config_files=configs/lfm1b-fil/idonly.yaml --model=FFM --task_name=idonly &

CUDA_VISIBLE_DEVICES=1 python run_recbole.py --dataset=lfm1b-fil --config_files=configs/lfm1b-fil/idonly.yaml --model=FiGNN --task_name=idonly &

CUDA_VISIBLE_DEVICES=2 python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/idonly.yaml configs/lfm1b-fil/audio.yaml" --model=FFM --task_name=id-a &

CUDA_VISIBLE_DEVICES=3 python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/idonly.yaml configs/lfm1b-fil/audio.yaml" --model=FiGNN --task_name=id-a 