model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/idonly.yaml configs/settings/cold.yaml" --model=$model --task_name=idonly-1b-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/aonly.yaml configs/settings/cold.yaml" --model=$model --task_name=aonly-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/id-other.yaml configs/settings/cold.yaml" --model=$model --task_name=id-other-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/id-cluster.yaml configs/settings/cold.yaml" --model=$model --task_name=id-cluster-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/clusteronly.yaml configs/settings/cold.yaml" --model=$model --task_name=clusteronly-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/id-token.yaml configs/settings/cold.yaml" --model=$model --task_name=id-token-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/token-cluster.yaml configs/settings/cold.yaml" --model=$model --task_name=token-cluster-cold --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b-fil/other-cluster.yaml configs/settings/cold.yaml" --model=$model --task_name=id-token-cluster-cold --gpu_id=$gpu_id 
