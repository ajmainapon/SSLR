#!/bin/bash
# SwinUNETR linear-probe baseline — 4 runs covering N=50 and N=100, SSL and random init.
# Same val set as ViT pipeline → apples-to-apples comparison.
#
# Per-run timing at iters_per_patient=10, epochs=25:
#   N=50 SSL/random:   12,500 iters → ~4.5 hr (incl. ~5 evals × 15 min)
#   N=100 SSL/random:  25,000 iters → ~9 hr
#   Queue total: ~27 hr
#
# After completion, compare:
#   v2 SSL linear probe N=50 = 0.1249  vs  SwinUNETR linear probe N=50 = ???
#   v2 SSL linear probe N=100 = 0.1492 vs  SwinUNETR linear probe N=100 = ???

set -eo pipefail   # pipefail catches python crashes through tee — DO NOT REMOVE.
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_swin_linprobe.log' ERR

PRETRAIN=~/SSLP/checkpoints/model_swinvit.pt
LOG=runs/queue_swin_linprobe.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — SwinUNETR linear probe" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

# Common args (everything except --pretrained, --n_train_volumes, --out)
COMMON="--slices_root data/slices --labels_root data/labels \
        --num_classes 11 --require_full_coverage \
        --mode linear \
        --epochs 25 --bs 1 --crop_size 96 --iters_per_patient 10 \
        --lr 1e-3 --warmup_epochs 3 --eval_every 5"

# ------------------------------------------------------------------
# Run 1/4: SSL pretrained @ N=50  (~4.5 hr)
# ------------------------------------------------------------------
NAME="swin_lin_n50_ssl"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --pretrained $PRETRAIN \
    --n_train_volumes 50 \
    --out runs/$NAME \
    $COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

# ------------------------------------------------------------------
# Run 2/4: random init @ N=50  (~4.5 hr)
# ------------------------------------------------------------------
NAME="swin_lin_n50_random"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --n_train_volumes 50 \
    --out runs/$NAME \
    $COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

# ------------------------------------------------------------------
# Run 3/4: SSL pretrained @ N=100  (~9 hr)
# ------------------------------------------------------------------
NAME="swin_lin_n100_ssl"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --pretrained $PRETRAIN \
    --n_train_volumes 100 \
    --out runs/$NAME \
    $COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

# ------------------------------------------------------------------
# Run 4/4: random init @ N=100  (~9 hr)
# ------------------------------------------------------------------
NAME="swin_lin_n100_random"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --n_train_volumes 100 \
    --out runs/$NAME \
    $COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "============================================================" | tee -a $LOG
echo "[$(date)] ALL SWINUNETR RUNS COMPLETE — summary:" | tee -a $LOG
echo "============================================================" | tee -a $LOG
for d in runs/swin_lin_n50_ssl runs/swin_lin_n50_random \
         runs/swin_lin_n100_ssl runs/swin_lin_n100_random; do
    if [ -f "$d/log.json" ]; then
        best=$(python3 -c "
import json
log = json.load(open('$d/log.json'))
peak = max(log, key=lambda r: r['val_fg'])
print(f\"{peak['val_fg']:.4f} at ep{peak['ep']:03d}\")
")
        printf "  %-30s best val_fg=%s\n" "$d" "$best" | tee -a $LOG
    fi
done
echo "[$(date)] queue done" | tee -a $LOG
