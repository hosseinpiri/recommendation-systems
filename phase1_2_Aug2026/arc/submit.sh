#!/bin/bash
# Submit chain: prep -> (fit array, boot array, sim array, margin array) -> merge
set -e
cd ~/mind_calib/jobs
PY=~/mind_calib/venv/bin/python
LOG=~/mind_calib/logs
mkdir -p $LOG

N_FIT=$($PY fit_cell.py count)
N_SIM=$($PY sim_cell.py count)
echo "fit cells: $N_FIT, sim cells: $N_SIM"

PREP=$(sbatch --parsable -p cpu2019 -c 4 --mem=24G -t 2:00:00 \
  -o $LOG/prep.log -J mind_prep --wrap "$PY prep.py")
echo "prep: $PREP"

FIT=$(sbatch --parsable --dependency=afterok:$PREP -p cpu2019 -c 2 --mem=14G \
  -t 1:00:00 -a 0-$((N_FIT-1))%400 -o $LOG/fit_%a.log -J mind_fit \
  --wrap "$PY fit_cell.py")
BOOT=$(sbatch --parsable --dependency=afterok:$PREP -p cpu2019 -c 2 --mem=12G \
  -t 2:00:00 -a 0-299%300 -o $LOG/boot_%a.log -J mind_boot \
  --wrap "$PY boot_cell.py")
SIM=$(sbatch --parsable --dependency=afterok:$PREP -p cpu2019 -c 1 --mem=6G \
  -t 3:00:00 -a 0-$((N_SIM-1))%100 -o $LOG/sim_%a.log -J mind_sim \
  --wrap "$PY sim_cell.py")
MARG=$(sbatch --parsable --dependency=afterok:$PREP -p cpu2019 -c 1 --mem=6G \
  -t 3:00:00 -a 0-499%500 -o $LOG/marg_%a.log -J mind_marg \
  --wrap "$PY margin_cell.py")
echo "fit: $FIT boot: $BOOT sim: $SIM marg: $MARG"

MERGE=$(sbatch --parsable \
  --dependency=afterany:$FIT:$BOOT:$SIM:$MARG -p cpu2019 -c 1 --mem=8G \
  -t 0:30:00 -o $LOG/merge.log -J mind_merge --wrap "$PY merge.py")
echo "merge: $MERGE"
