# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$0")

# bash "$SCRIPT_DIR/run_ctr_idonly.sh"
# bash "$SCRIPT_DIR/run_ctr_id-a.sh"
# bash "$SCRIPT_DIR/run_ctr_token.sh"
# bash "$SCRIPT_DIR/run_ctr_token-a-layer.sh"
# bash "$SCRIPT_DIR/run_ctr_token-msclap.sh"
# bash "$SCRIPT_DIR/run_ctr_token-mulan.sh"
# bash "$SCRIPT_DIR/run_ctr_all.sh"
# bash "$SCRIPT_DIR/run_ctr_all-a-layer.sh"
# bash "$SCRIPT_DIR/run_ctr_all-msclap.sh"
# bash "$SCRIPT_DIR/run_ctr_all-mulan.sh"

# bash "$SCRIPT_DIR/run_ctr_id-cluster16.sh"
# bash "$SCRIPT_DIR/run_ctr_token-cluster16.sh"
bash "$SCRIPT_DIR/run_ctr_all.sh"
bash "$SCRIPT_DIR/run_ctr_all-a.sh"
bash "$SCRIPT_DIR/run_ctr_all-cluster16.sh"
bash "$SCRIPT_DIR/run_ctr_all-mulan16.sh"
