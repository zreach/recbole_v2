SCRIPT_DIR=$(dirname "$0")

bash "$SCRIPT_DIR/run_2b-idonly.sh" BPR 6 &
bash "$SCRIPT_DIR/run_2b-idonly.sh" LightGCN 2 &
bash "$SCRIPT_DIR/run_2b-id-a-t.sh" VBPR 3 &
bash "$SCRIPT_DIR/run_2b-id-a-t.sh" FREEDOM 4 &
bash "$SCRIPT_DIR/run_2b-id-a-t.sh" LGMRec 5 
