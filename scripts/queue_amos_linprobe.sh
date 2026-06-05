#!/bin/bash
# AMOS-CT cross-dataset linear probe — A3 SSL vs Random init at N=20/50/100.
#
# Purpose: convert "single dataset" limitation (Limitation #4 in paper_draft)
# into evidence that the A3 SSL features generalize beyond TotalSegmentator.
# This is a 7-class probe (background + liver, spleen, kid_L, kid_R, stomach,
# pancreas, aorta) — AMOS is abdominal-only, so lungs+heart are excluded.
#
# Prereq (run once):
#   python scripts/preprocess_amos.py --split train
#   python scripts/preprocess_amos.py --split val
#   python scripts/prepare_labels_amos.py --split train
#   python scripts/prepare_labels_amos.py --split val
#
# Compute budget: 6 runs × ~15 min each = ~1.5 hr total. Linear probes only
# (fine-tune sweep deferred to queue_amos_finetune.sh; ~10x more compute).

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_amos_linprobe.log' ERR

ulimit -n 65536

# Adjust if your checkpoint paths differ
A3_CKPT=$HOME/SSLP/checkpoints/loo_a3_narrow_k/vit_ep039.pt
LOG=runs/queue_amos_linprobe.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — AMOS cross-dataset linear probe" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "Backbone: $A3_CKPT (A3 SSL — random masking + narrow slice gap)" | tee -a $LOG
echo "Dataset: AMOS22 Task-1 (CT only, abdominal, 7-class overlap)" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

# Common args for the AMOS probes.
# NOTE: --num_classes 8  (background + 7 organs, NOT 11 as in TotalSeg).
COMMON="--slices_root data_amos/slices --labels_root data_amos/labels \
        --num_classes 8 --require_full_coverage \
        --head linear \
        --epochs 50 --bs 8 \
        --lr 1e-3 --wd 1e-4 \
        --eval_every 5"

# ===================================================================
# A3 SSL probes — 3 N values
# ===================================================================
for N in 20 50 100; do
    NAME="lin_amos_a3_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --ckpt $A3_CKPT \
        --n_train_volumes $N \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

# ===================================================================
# Random init probes — 3 N values (matched protocol, no --ckpt)
# ===================================================================
for N in 20 50 100; do
    NAME="lin_amos_random_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --n_train_volumes $N \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

# ===================================================================
# Summary
# ===================================================================
echo "============================================================" | tee -a $LOG
echo "[$(date)] ALL AMOS LINEAR-PROBE RUNS COMPLETE" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os

def best(p):
    if not os.path.exists(p): return None
    log = json.load(open(p))
    vals = [e['val_fg'] for e in log if 'val_fg' in e]
    return max(vals) if vals else None

print(f"\n{'cell':<28} {'val_fg':<8}")
print("-" * 40)
for n in (20, 50, 100):
    for tag in ('a3', 'random'):
        p = f"runs/lin_amos_{tag}_n{n}/log.json"
        v = best(p)
        s = f"{v:.4f}" if v is not None else "MISSING"
        print(f"  lin_amos_{tag}_n{n:<3}            {s}")
    a3 = best(f"runs/lin_amos_a3_n{n}/log.json")
    rd = best(f"runs/lin_amos_random_n{n}/log.json")
    if a3 and rd:
        rel = 100 * (a3 - rd) / rd
        print(f"  -> A3 vs Random at N={n}: {a3-rd:+.4f}  ({rel:+.1f}%)")
    print()

print("\nReporting note for paper §Exp-cross-dataset:")
print("  AMOS = 7-class probe (lungs+heart absent), N=20/50/100, single seed,")
print("  same A3 backbone as TotalSeg headline, same linear-probe protocol.")
print("  Compare against TotalSeg linear-probe (Table tab:data-efficiency):")
print("    TotalSeg A3 SSL @ N=50:  0.2077 ;  AMOS A3 SSL @ N=50:  <see above>")
print("    TotalSeg A3 SSL @ N=100: 0.2159 ;  AMOS A3 SSL @ N=100: <see above>")
PYEOF

echo "[$(date)] queue done" | tee -a $LOG
