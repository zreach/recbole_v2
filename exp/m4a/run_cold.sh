# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$0")

bash "$SCRIPT_DIR/run_ctr_all-cold.sh"
bash "$SCRIPT_DIR/run_ctr_all-t-cluster16-cold.sh"
