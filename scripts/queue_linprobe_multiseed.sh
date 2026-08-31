#!/bin/bash
# Multi-seed linear-probe backfill: seeds 1,2,3 for every in-dataset probe cell.
# Existing runs (no seed suffix) are seed 0 -- together these give 4 seeds.
#
# Ordered by N priority (50 -> 100 -> 20 -> 10 -> 5) so that if the queue is
# stopped early, the headline operating point is already complete.
#
# 72 runs total. Estimated ~29 h.

set -uo pipefail
ulimit -n 65536
ROOT="${SSLR_ROOT:-$HOME/SSLP}"
cd "$ROOT"

LOG=runs/queue_linprobe_multiseed.log
mkdir -p runs

DECODER_ARGS="--slices_root data/slices --labels_root data/labels               --num_classes 11 --require_full_coverage               --head linear --epochs 50 --bs 16               --lr 1e-3 --eval_every 5"

CKPT_V2="$ROOT/checkpoints/vit_ep039.pt"
CKPT_A1="$ROOT/checkpoints/loo_a1_random_mask/vit_ep039.pt"
CKPT_A3="$ROOT/checkpoints/loo_a3_narrow_k/vit_ep039.pt"
CKPT_A5="$ROOT/checkpoints/loo_a5_dense_narrow/vit_ep039.pt"

echo "============================================================" | tee -a $LOG
echo "[$(date)] multi-seed linear-probe backfill starting (PID $$)" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free --format=csv | tee -a $LOG

run_probe () {
    local NAME=$1; shift
    if [ -d "runs/$NAME" ]; then
        echo "[$(date)] SKIP $NAME (exists)" | tee -a $LOG
        return 0
    fi
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py "$@" --out runs/$NAME 2>&1 | tee runs/$NAME.log
    local RC=${PIPESTATUS[0]}
    if [ $RC -ne 0 ]; then
        echo "[$(date)] !!! FAILED $NAME (rc=$RC) -- continuing" | tee -a $LOG
    else
        echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
    fi
}

for N in 50 100 20 10 5; do
  for SEED in 1 2 3; do
    run_probe "lin_v2_ssl_n${N}_seed${SEED}"     --ckpt $CKPT_V2 $DECODER_ARGS --n_train_volumes $N --seed $SEED
    run_probe "lin_loo_a1_n${N}_seed${SEED}"     --ckpt $CKPT_A1 $DECODER_ARGS --n_train_volumes $N --seed $SEED
    run_probe "lin_loo_a3_n${N}_seed${SEED}"     --ckpt $CKPT_A3 $DECODER_ARGS --n_train_volumes $N --seed $SEED
    run_probe "lin_v2_random_n${N}_seed${SEED}"  $DECODER_ARGS --n_train_volumes $N --seed $SEED
    if [ "$N" != "10" ]; then
      run_probe "lin_a5_n${N}_seed${SEED}"       --ckpt $CKPT_A5 $DECODER_ARGS --n_train_volumes $N --seed $SEED
    fi
  done
  echo "[$(date)] ===== N=$N complete (all backbones, seeds 1-3) =====" | tee -a $LOG
done

echo "[$(date)] === multi-seed linear-probe backfill COMPLETE ===" | tee -a $LOG
