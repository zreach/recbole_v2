model=${1:-"FM"}
gpu_id=${2:-"0"}

SCRIPT_DIR=$(dirname "$0")


bash "$SCRIPT_DIR/run_m4a-idonly.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-id-a.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-id-a-t.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-id-t.sh" $model $gpu_id 

bash "$SCRIPT_DIR/run_m4a-token.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-token-a.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-token-a-t.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-token-t.sh" $model $gpu_id 

bash "$SCRIPT_DIR/run_m4a-all.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-all-a.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-all-a-t.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_m4a-all-t.sh" $model $gpu_id



