model=${1:-"FM"}
gpu_id=${2:-"0"}

SCRIPT_DIR=$(dirname "$0")


CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/run_2b-idonly.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/run_2b-id-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/run_2b-id-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/run_2b-id-t-mulan16.sh" $model $gpu_id 

CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/run_2b-token.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/run_2b-token-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/run_2b-token-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/run_2b-token-t-mulan16.sh" $model $gpu_id 

CUDA_VISIBLE_DEVICES=0 bash "$SCRIPT_DIR/run_2b-all.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=1 bash "$SCRIPT_DIR/run_2b-all-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=2 bash "$SCRIPT_DIR/run_2b-all-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=3 bash "$SCRIPT_DIR/run_2b-all-t-mulan16.sh" $model $gpuall


