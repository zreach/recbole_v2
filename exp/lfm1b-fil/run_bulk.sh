model=${1:-"FM"}
gpu_id=${2:-"0"}

python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/idonly.yaml configs/settings/num.yaml" --model=$model --task_name=idonly-1b-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/aonly.yaml configs/settings/num.yaml" --model=$model --task_name=aonly-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/id-other.yaml configs/settings/num.yaml" --model=$model --task_name=id-other-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/id-cluster.yaml configs/settings/num.yaml" --model=$model --task_name=id-cluster-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/clusteronly.yaml configs/settings/num.yaml" --model=$model --task_name=clusteronly-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/id-token.yaml configs/settings/num.yaml" --model=$model --task_name=id-token-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/token-cluster.yaml configs/settings/num.yaml" --model=$model --task_name=token-cluster-5 --gpu_id=$gpu_id &
python run_recbole.py --dataset=lfm1b-fil --config_files="configs/lfm1b/other-cluster.yaml configs/settings/num.yaml" --model=$model --task_name=id-token-cluster-5 --gpu_id=$gpu_id 
