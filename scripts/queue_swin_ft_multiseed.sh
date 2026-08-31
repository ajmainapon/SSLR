#!/bin/bash
# SwinUNETR FT-vs-FT multi-seed backfill: seeds 1,2,3.
# Existing swin_ft_*_seed0 runs supply seed 0 -> 4 seeds per cell.
#
# Ordered N=20 -> N=50 -> N=100 (cheapest first). The completed four-seed
# comparison gives SSL-vs-random gaps of +17.9%, +0.6%, and -1.5%.
#
# 18 runs. Est ~18.4 h:  N=20 2.9h | N=50 5.5h | N=100 10.0h
set -uo pipefail
ROOT="${SSLR_ROOT:-$HOME/SSLP}"
cd "$ROOT"
source venv/bin/activate
PRETRAIN="$ROOT/checkpoints/model_swinvit.pt"
LOG=runs/queue_swin_ft_multiseed.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] SwinUNETR multi-seed queue starting (PID $$)" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free --format=csv | tee -a $LOG

run_one () {  # $1=N  $2=ssl|random  $3=seed
  local N=$1 INIT=$2 SEED=$3 NAME PRE=""
  NAME="swin_ft_n${N}_${INIT}_seed${SEED}"
  if [ -d "runs/$NAME" ]; then
    echo "[$(date)] SKIP $NAME (exists)" | tee -a $LOG
    return 0
  fi
  [ "$INIT" = "ssl" ] && PRE="--pretrained $PRETRAIN"
  echo "[$(date)] >>> START $NAME" | tee -a $LOG
  python scripts/train_swinunetr.py $PRE --n_train_volumes $N --out runs/$NAME       --slices_root data/slices --labels_root data/labels       --num_classes 11 --require_full_coverage       --mode finetune --epochs 30 --bs 1 --crop_size 96 --iters_per_patient 10       --lr_head 1e-3 --lr_backbone 1e-4 --warmup_epochs 3 --eval_every 5       --seed $SEED 2>&1 | tee runs/$NAME.log
  local RC=${PIPESTATUS[0]}
  if [ $RC -ne 0 ]; then
    echo "[$(date)] !!! FAILED $NAME (rc=$RC) -- continuing" | tee -a $LOG
  else
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
  fi
}

for N in 20 50 100; do
  for SEED in 1 2 3; do
    run_one $N ssl    $SEED
    run_one $N random $SEED
  done
  echo "[$(date)] ===== SwinUNETR N=$N complete (seeds 1-3) =====" | tee -a $LOG
done
echo "[$(date)] === SwinUNETR multi-seed backfill COMPLETE ===" | tee -a $LOG
