SCRIPT_DIR=$(dirname "$0")

bash "$SCRIPT_DIR/run_m4a-idonly.sh" BPR 6 &
bash "$SCRIPT_DIR/run_m4a-idonly.sh" LightGCN 2 &
bash "$SCRIPT_DIR/run_m4a-id-a-t.sh" VBPR 3 &
bash "$SCRIPT_DIR/run_m4a-id-a-t.sh" FREEDOM 7 &
bash "$SCRIPT_DIR/run_m4a-id-a-t.sh" LGMRec 6 
