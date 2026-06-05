#!/bin/bash
# MRI cross-modality linear probe — MULTI-SEED extension (seeds 1,2,3).
# Single-seed (seed 0) already done: lin_mri_{a3,random}_n{20,50,100}.
#
# IMPORTANT (verified 2026-06-04): only 39 MR train patients pass full 7-class
# coverage. Therefore:
#   - N=20  is a GENUINE data-efficiency point (20/39 subset; varies by seed).
#   - N=50  is CAPPED to all 39 volumes (same SET every seed; variance = init/optim only).
#   - N=100 is IDENTICAL to N=50 (same 39-volume set, overlap 39/39) -> NOT RUN (pure duplicate).
# So this queue runs seeds 1,2,3 x {a3 ssl, random} x {N=20, N=50} = 12 runs.
# Faithful replication of seed-0 protocol (queue_mri_linprobe.sh): frozen A3
# backbone, linear head, epochs=50, bs=8, lr=1e-3, wd=1e-4, num_classes=8,
# --require_full_coverage, AMOS-MR (IDs 0501-0600), 7-class overlap.
set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_mri_multiseed.log' ERR

cd "$HOME/SSLP"
source venv/bin/activate
ulimit -n 65536

A3_CKPT=$HOME/SSLP/checkpoints/loo_a3_narrow_k/vit_ep039.pt
LOG=runs/queue_mri_multiseed.log
mkdir -p runs
echo "============================================================" | tee -a $LOG
echo "[$(date)] MRI multi-seed (seeds 1-3) starting (PID $$)" | tee -a $LOG
echo "[$(date)] backbone $A3_CKPT (frozen); N in {20,50}; skip N=100 (==N=50)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

COMMON="--slices_root data_mri/slices --labels_root data_mri/labels \
        --num_classes 8 --require_full_coverage \
        --head linear --epochs 50 --bs 8 --lr 1e-3 --wd 1e-4 --eval_every 5"

run_one () {  # $1=N  $2=ssl|random  $3=seed
  local N=$1 INIT=$2 S=$3 NAME PRE=""
  NAME="lin_mri_${INIT/ssl/a3}_n${N}_seed${S}"
  [ "$INIT" = "ssl" ] && PRE="--ckpt $A3_CKPT"
  echo "[$(date)] >>> START $NAME" | tee -a $LOG
  python scripts/train_decoder.py $PRE --n_train_volumes $N --seed $S --out runs/$NAME $COMMON \
      2>&1 | tee runs/$NAME.log
  echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
}

for S in 1 2 3; do
  run_one 20 ssl    $S
  run_one 20 random $S
  run_one 50 ssl    $S
  run_one 50 random $S
done

echo "============================================================" | tee -a $LOG
echo "[$(date)] MRI multi-seed COMPLETE — 4-seed (0..3) summary:" | tee -a $LOG
echo "============================================================" | tee -a $LOG
python3 <<'PYEOF' | tee -a $LOG
import json, os, statistics as st
R = os.path.expanduser("~/SSLP/runs")
def best(p):
    try:    return max(e["val_fg"] for e in json.load(open(p)) if "val_fg" in e)
    except: return None
for N in (20, 50):
    cell = {}
    for init in ("a3", "random"):
        vals = []
        for s in (0, 1, 2, 3):
            name = f"lin_mri_{init}_n{N}" if s == 0 else f"lin_mri_{init}_n{N}_seed{s}"
            v = best(f"{R}/{name}/log.json")
            if v is not None: vals.append(v)
        cell[init] = vals
        if vals:
            m = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"N={N:<3} {init:<7} n={len(vals)} mean={m:.4f} std={sd:.4f}  vals={[round(x,4) for x in vals]}")
    if cell.get("a3") and cell.get("random"):
        ma, mr = st.mean(cell["a3"]), st.mean(cell["random"])
        print(f"  -> N={N}: A3 vs Random  mean {ma:.4f} vs {mr:.4f}  ({100*(ma-mr)/mr:+.1f}% rel)\n")
PYEOF
echo "[$(date)] queue done" | tee -a $LOG
