# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$0")

# bash "$SCRIPT_DIR/run_ctr_id-a.sh"
bash "$SCRIPT_DIR/run_ctr_all.sh"
bash "$SCRIPT_DIR/run_ctr_all-a.sh"
bash "$SCRIPT_DIR/run_ctr_token.sh"
bash "$SCRIPT_DIR/run_ctr_token-a.sh"
# bash "$SCRIPT_DIR/run_ctr_idonly.sh"
