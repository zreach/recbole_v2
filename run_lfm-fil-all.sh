model=${1:-"FM"}

python run_recbole.py --dataset=lfm1b-filtered --config_files="configs/lfm1b-fil-all.yaml /user/zhouyz/rec/RecBole-master/RecBole-master/recbole/properties/model/${model}.yaml" --model=$model --task_name=all
