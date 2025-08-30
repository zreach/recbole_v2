model=${1:-"FM"}
gpu_id=${2:-"0"}

SCRIPT_DIR=$(dirname "$0")


CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-idonly.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-id-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-id-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-id-t-mulan16.sh" $model $gpu_id 

CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-token.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-token-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-token-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-token-t-mulan16.sh" $model $gpu_id 

CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all-a-t.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all-t-cluster16.sh" $model $gpu_id &
CUDA_VISIBLE_DEVICES=$gpu_id bash "$SCRIPT_DIR/run_m4a-all-t-mulan16.sh" $model $gpuall


