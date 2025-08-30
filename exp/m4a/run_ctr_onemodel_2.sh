model=${1:-"FM"}
gpu_id=${2:-"0"}

SCRIPT_DIR=$(dirname "$0")

CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all-t-mulan16.sh" $model $gpuall


