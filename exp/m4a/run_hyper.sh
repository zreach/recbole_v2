model=${1:-"FM"}

CUDA_VISIBLE_DEVICES="2" python run_hyper.py --dataset=m4a --config_files="configs/m4a/idonly.yaml configs/m4a/audio.yaml" --model=$model --task_name=id-a-hyper --params_file=hyper-layer.test