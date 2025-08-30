# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$0")

bash "$SCRIPT_DIR/run_ctr_all-t-mulan.sh"
bash "$SCRIPT_DIR/run_ctr_all-t-msclap.sh"
bash "$SCRIPT_DIR/run_ctr_all-t-mfcc.sh"