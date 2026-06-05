#!/bin/bash
# Recovery queue for the SwinUNETR linear probe — Runs 3 & 4 only.
# Originally was a 4-run queue; Runs 1 & 2 (N=50) completed before the
# 2026-05-17 03:00 power outage. This script picks up where the original
# `queue_swin_linprobe.sh` died.
#
# Per-run estimate (based on observed Run 2 timing of ~3h 20m at N=50):
#   N=100 SSL/random:  ~6–8 hr each (2x data than N=50)
#   Queue total:       ~13–16 hr
#
# IMPORTANT: do NOT run this until the N=50 results have been verified
# (see verification command in chat). If Runs 1 or 2 are invalid, use the
# full queue_swin_linprobe.sh instead.

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_swin_linprobe_recovery.log' ERR

PRETRAIN=~/SSLP/checkpoints/model_swinvit.pt
LOG=runs/queue_swin_linprobe_recovery.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — SwinUNETR recovery (N=100 only)" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

COMMON="--slices_root data/slices --labels_root data/labels \
        --num_classes 11 --require_full_coverage \
        --mode linear \
        --epochs 25 --bs 1 --crop_size 96 --iters_per_patient 10 \
        --lr 1e-3 --warmup_epochs 3 --eval_every 5"

# ------------------------------------------------------------------
# Run 1/2 (resumes original Run 3): SSL pretrained @ N=100  (~6–8 hr)
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
# Run 2/2 (resumes original Run 4): random init @ N=100  (~6–8 hr)
# ------------------------------------------------------------------
NAME="swin_lin_n100_random"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --n_train_volumes 100 \
    --out runs/$NAME \
    $COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

# ------------------------------------------------------------------
# Summary across ALL 4 runs (the original 2 + these 2)
# ------------------------------------------------------------------
echo "============================================================" | tee -a $LOG
echo "[$(date)] FULL SWINUNETR COMPARISON — all 4 runs:" | tee -a $LOG
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
echo "[$(date)] recovery queue done" | tee -a $LOG
