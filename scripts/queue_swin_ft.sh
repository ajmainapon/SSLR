#!/bin/bash
# SwinUNETR FT-vs-FT matched comparison (WACV, user decision 2026-06-04).
# Replaces the apples-to-oranges "our FT vs their LP" with a matched protocol:
# fine-tune SwinUNETR END-TO-END under the SAME recipe as the A3 fine-tune
#   epochs=30, lr_head=1e-3, lr_backbone=1e-4, backbone_wd=0.05, warmup=3,
#   crop=96, bs=1, --require_full_coverage, same 52-patient val split.
# (Only unmatched axis vs A3: no LLRD on SwinUNETR's 2-group optimizer -- an
#  architecture difference, caveated in the paper. Within-SwinUNETR SSL-vs-Random
#  is perfectly matched: identical script, only --pretrained differs.)
#
# Single seed (seed 0) first, cheapest N first, so the N=20 run validates the FT
# path + gives real timing before the expensive N=100 runs. Extra seeds decided
# after timing is observed.
set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO"' ERR

cd "$HOME/SSLP"
source venv/bin/activate
PRETRAIN="$HOME/SSLP/checkpoints/model_swinvit.pt"
LOG=runs/queue_swin_ft.log
mkdir -p runs
echo "============================================================" | tee -a $LOG
echo "[$(date)] SwinUNETR FT-vs-FT queue starting (PID $$)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

COMMON="--slices_root data/slices --labels_root data/labels \
        --num_classes 11 --require_full_coverage \
        --mode finetune --epochs 30 --bs 1 --crop_size 96 --iters_per_patient 10 \
        --lr_head 1e-3 --lr_backbone 1e-4 --warmup_epochs 3 --eval_every 5 --seed 0"

run_one () {  # $1=N  $2=ssl|random
  local N=$1 INIT=$2 NAME PRE=""
  NAME="swin_ft_n${N}_${INIT}_seed0"
  [ "$INIT" = "ssl" ] && PRE="--pretrained $PRETRAIN"
  echo "[$(date)] >>> START $NAME" | tee -a $LOG
  python scripts/train_swinunetr.py $PRE --n_train_volumes $N --out runs/$NAME $COMMON \
      2>&1 | tee runs/$NAME.log
  echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
}

run_one 20  ssl
run_one 20  random
run_one 50  ssl
run_one 50  random
run_one 100 ssl
run_one 100 random

echo "============================================================" | tee -a $LOG
echo "[$(date)] SwinUNETR FT-vs-FT — summary (best val_fg):" | tee -a $LOG
for d in runs/swin_ft_n20_ssl_seed0 runs/swin_ft_n20_random_seed0 \
         runs/swin_ft_n50_ssl_seed0 runs/swin_ft_n50_random_seed0 \
         runs/swin_ft_n100_ssl_seed0 runs/swin_ft_n100_random_seed0; do
  if [ -f "$d/log.json" ]; then
    best=$(python3 -c "import json;l=json.load(open('$d/log.json'));p=max(l,key=lambda r:r['val_fg']);print(f\"{p['val_fg']:.4f} ep{p['ep']:03d}\")")
    printf "  %-34s best val_fg=%s\n" "$d" "$best" | tee -a $LOG
  fi
done
echo "[$(date)] queue done" | tee -a $LOG
