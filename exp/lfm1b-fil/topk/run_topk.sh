SCRIPT_DIR=$(dirname "$0")

bash "$SCRIPT_DIR/run_1b-idonly.sh" BPR 3 &
bash "$SCRIPT_DIR/run_1b-idonly.sh" LightGCN 4 &
bash "$SCRIPT_DIR/run_1b-id-a-t.sh" VBPR 5 &
bash "$SCRIPT_DIR/run_1b-id-a-t.sh" FREEDOM 7 &
bash "$SCRIPT_DIR/run_1b-id-a-t.sh" LGMRec 6 
