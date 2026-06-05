#!/bin/bash
# Quick status for the loo_a5_dense_narrow pretrain.
cd ~/SSLP
R=loo_a5_dense_narrow
PIDF=runs/$R.pid
echo "=========================================================="
echo " a5 (dense_narrow) pretrain status — $(date +%H:%M:%S)"
echo "=========================================================="
if [ -f "$PIDF" ] && ps -p $(cat $PIDF) >/dev/null 2>&1; then
  echo "state    : RUNNING (pid $(cat $PIDF), elapsed $(ps -p $(cat $PIDF) -o etime= | tr -d " "))"
else
  echo "state    : NOT RUNNING (finished or crashed — see tail below)"
fi
LAST=$(grep -oE "^ep[0-9]+" runs/$R.log | tail -1)
echo "epoch    : ${LAST:-?} of 40"
echo "latest   : $(tail -n1 runs/$R.log)"
echo "ckpts    : $(ls checkpoints/$R 2>/dev/null | tail -2 | tr "\n" " ")"
echo "GPU      : $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader)"
echo "----------------------------------------------------------"
