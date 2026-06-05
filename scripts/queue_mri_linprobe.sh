#!/bin/bash
# MRI cross-modality linear probe — A3 SSL (CT-pretrained) vs Random init.
#
# THE BIG QUESTION: do CT-pretrained SSL features transfer to a DIFFERENT
# MODALITY (MR) at all? Modality shift is much larger than dataset shift.
#
# Acceptance criterion per CLAUDE.md WACV R1 plan:
#   CT->MRI linear probe at N=50 must show SSL > Random by >=10% relative.
#   If yes -> include MRI section in paper.
#   If no  -> ship CT-only (don't oversell).
#
# Total: 6 runs (3 N x {A3 SSL, Random}, single seed).
# Wall-clock: ~12 min/run * 6 = ~75 min.
#
# Prereqs (verify before launching):
#   - data_mri/slices/{train,val}/   populated  (preprocess_amos_mri.py)
#   - data_mri/labels/{train,val}/   populated  (prepare_labels_amos_mri.py)
#   - checkpoints/loo_a3_narrow_k/vit_ep039.pt   (A3 backbone)

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_mri_linprobe.log' ERR

source $HOME/SSLP/venv/bin/activate

ulimit -n 65536

A3_CKPT=$HOME/SSLP/checkpoints/loo_a3_narrow_k/vit_ep039.pt
LOG=runs/queue_mri_linprobe.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — MRI cross-modality linear probe" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "Backbone: $A3_CKPT (A3 SSL, CT-pretrained, frozen)" | tee -a $LOG
echo "Target: AMOS22 MR (IDs 0501-0600, 7-class abdominal overlap)" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

# AMOS-MR config: 7-class overlap (same as AMOS-CT), --num_classes 8.
COMMON="--slices_root data_mri/slices --labels_root data_mri/labels \
        --num_classes 8 --require_full_coverage \
        --head linear \
        --epochs 50 --bs 8 \
        --lr 1e-3 --wd 1e-4 \
        --eval_every 5"

# ===================================================================
# A3 SSL probes — 3 N values
# ===================================================================
for N in 20 50 100; do
    NAME="lin_mri_a3_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --ckpt $A3_CKPT \
        --n_train_volumes $N \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

# ===================================================================
# Random init probes — 3 N values (no --ckpt)
# ===================================================================
for N in 20 50 100; do
    NAME="lin_mri_random_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --n_train_volumes $N \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

# ===================================================================
# Decision-criterion summary
# ===================================================================
echo "============================================================" | tee -a $LOG
echo "[$(date)] MRI CROSS-MODALITY LINEAR PROBE COMPLETE" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os
runs_dir = os.path.expanduser("~/SSLP/runs")

def best(p):
    try:
        log = json.load(open(p))
        return max((e["val_fg"] for e in log if "val_fg" in e), default=None)
    except: return None

print(f"\n{'cell':<22} {'val_fg':<8}")
print("-" * 50)
results = {}
for n in (20, 50, 100):
    v_a3  = best(f"{runs_dir}/lin_mri_a3_n{n}/log.json")
    v_rnd = best(f"{runs_dir}/lin_mri_random_n{n}/log.json")
    results[n] = (v_a3, v_rnd)
    print(f"lin_mri_a3_n{n:<10} {v_a3:.4f}" if v_a3 else f"lin_mri_a3_n{n:<10} MISSING")
    print(f"lin_mri_random_n{n:<6} {v_rnd:.4f}" if v_rnd else f"lin_mri_random_n{n:<6} MISSING")
    if v_a3 and v_rnd:
        rel = 100 * (v_a3 - v_rnd) / v_rnd
        print(f"  -> A3 vs Random N={n}: +{v_a3-v_rnd:.4f}  ({rel:+.1f}%)")
    print()

# Decision criterion: CT->MRI N=50 must show SSL > Random by >=10% relative.
v_a3, v_rnd = results.get(50, (None, None))
print("=" * 60)
if v_a3 and v_rnd:
    rel = 100 * (v_a3 - v_rnd) / v_rnd
    if rel >= 10:
        print(f"DECISION: INCLUDE MRI in paper (N=50 gap = {rel:+.1f}% >= 10%)")
    else:
        print(f"DECISION: SHIP CT-ONLY (N=50 gap = {rel:+.1f}% < 10%)")
        print(f"          MRI features did not transfer well enough.")
else:
    print("DECISION: cannot evaluate (missing N=50 runs)")
print("=" * 60)
PYEOF

echo "[$(date)] queue done" | tee -a $LOG
