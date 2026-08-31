#!/bin/bash
# Fill the one remaining gap in the N=10 linear-probe row: a5 seeds 1-3.
# Existing lin_a5_n10 (seed 0) = 0.1163. ~9.5 min per run, ~30 min total.
set -uo pipefail
ROOT="${SSLR_ROOT:-$HOME/SSLP}"
cd "$ROOT"
LOG=runs/queue_a5_n10_seeds.log
mkdir -p runs
CKPT_A5="$ROOT/checkpoints/loo_a5_dense_narrow/vit_ep039.pt"
echo "[$(date)] a5 N=10 seed backfill starting (PID $$)" | tee -a $LOG
for SEED in 1 2 3; do
  NAME="lin_a5_n10_seed${SEED}"
  if [ -d "runs/$NAME" ]; then echo "[$(date)] SKIP $NAME" | tee -a $LOG; continue; fi
  echo "[$(date)] >>> START $NAME" | tee -a $LOG
  python scripts/train_decoder.py --ckpt "$CKPT_A5"       --slices_root data/slices --labels_root data/labels       --num_classes 11 --require_full_coverage       --head linear --epochs 50 --bs 16 --lr 1e-3 --eval_every 5       --n_train_volumes 10 --seed "$SEED" --out "runs/$NAME" 2>&1 | tee "runs/$NAME.log"
  echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done
echo "[$(date)] === a5 N=10 backfill COMPLETE ===" | tee -a $LOG
