# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$0")

# bash "$SCRIPT_DIR/run_ctr_id-a.sh"
# bash "$SCRIPT_DIR/run_ctr_all.sh"
# bash "$SCRIPT_DIR/run_ctr_all-a.sh"
# bash "$SCRIPT_DIR/run_ctr_token-a.sh"
# bash "$SCRIPT_DIR/run_ctr_token.sh"
# bash "$SCRIPT_DIR/run_ctr_idonly.sh"
# bash "$SCRIPT_DIR/run_ctr_id-a-t.sh"
# bash "$SCRIPT_DIR/run_ctr_token-a-t-layer.sh"
# bash "$SCRIPT_DIR/run_ctr_token-msclap-t.sh"
# bash "$SCRIPT_DIR/run_ctr_token-mulan-t.sh"


bash "$SCRIPT_DIR/run_ctr_all.sh"
bash "$SCRIPT_DIR/run_ctr_all-a-t.sh"
bash "$SCRIPT_DIR/run_ctr_all-t-cluster16.sh"
bash "$SCRIPT_DIR/run_ctr_all-t-mulan16.sh"

