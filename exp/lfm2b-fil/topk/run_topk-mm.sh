model=${1:-"BPR"}
gpu_id=${2:-"0"}
SCRIPT_DIR=$(dirname "$0")

bash "$SCRIPT_DIR/run_2b-id-a.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_2b-id-t.sh" $model $gpu_id &
bash "$SCRIPT_DIR/run_2b-id-a-t.sh" $model $gpu_id