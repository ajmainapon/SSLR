#!/bin/bash
# Phase-1 combined queue:
#   (a) AMOS multi-seed extension: 4 seeds × 3 N × {A3 SSL, Random} = 24 runs
#       Reuses existing seed=0 runs (lin_amos_a3_n*, lin_amos_random_n*),
#       only launches seeds 1,2,3 → 18 NEW runs.
#   (b) SwinUNETR N=20: adds a third row to paper Table 7 (currently N=50,100).
#       SSL + Random, single seed → 2 NEW runs.
#
# Total: 20 new runs.
# Wall-clock estimate:
#   AMOS linear probe: ~12 min/run × 18 = ~3.6 hr
#   SwinUNETR N=20:    ~1.5 hr/run × 2  = ~3 hr   (Swin N=50 took ~3.3 hr)
#   GRAND TOTAL:                          ~6-7 hr
#
# Prereqs (verify before launching):
#   - data_amos/slices/{train,val}/  populated  (Phase 0)
#   - data_amos/labels/{train,val}/  populated  (Phase 0)
#   - data/slices/, data/labels/     populated  (TotalSeg, for SwinUNETR)
#   - checkpoints/loo_a3_narrow_k/vit_ep039.pt   (A3 backbone)
#   - checkpoints/model_swinvit.pt               (SwinUNETR pretrained)

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_phase1.log' ERR

ulimit -n 65536

A3_CKPT=$HOME/SSLP/checkpoints/loo_a3_narrow_k/vit_ep039.pt
SWIN_CKPT=$HOME/SSLP/checkpoints/model_swinvit.pt
LOG=runs/queue_phase1.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] PHASE-1 queue starting (PID $$)" | tee -a $LOG
echo "  (a) AMOS multi-seed (3 seeds × 3 N × {SSL, Random}) = 18 runs" | tee -a $LOG
echo "  (b) SwinUNETR N=20 (SSL + Random)                   = 2 runs"  | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "A3 backbone: $A3_CKPT" | tee -a $LOG
echo "Swin pretrained: $SWIN_CKPT" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

# ===================================================================
# (a) AMOS multi-seed extension
# ===================================================================
echo "" | tee -a $LOG
echo "===== STAGE (a): AMOS multi-seed =====" | tee -a $LOG

AMOS_COMMON="--slices_root data_amos/slices --labels_root data_amos/labels \
             --num_classes 8 --require_full_coverage \
             --head linear \
             --epochs 50 --bs 8 \
             --lr 1e-3 --wd 1e-4 \
             --eval_every 5"

for N in 20 50 100; do
    for SEED in 1 2 3; do
        NAME="lin_amos_a3_n${N}_seed${SEED}"
        echo "[$(date)] >>> START $NAME" | tee -a $LOG
        python scripts/train_decoder.py \
            --ckpt $A3_CKPT \
            --n_train_volumes $N \
            --seed $SEED \
            $AMOS_COMMON \
            --out runs/$NAME 2>&1 | tee runs/$NAME.log
        echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
    done
done

for N in 20 50 100; do
    for SEED in 1 2 3; do
        NAME="lin_amos_random_n${N}_seed${SEED}"
        echo "[$(date)] >>> START $NAME" | tee -a $LOG
        python scripts/train_decoder.py \
            --n_train_volumes $N \
            --seed $SEED \
            $AMOS_COMMON \
            --out runs/$NAME 2>&1 | tee runs/$NAME.log
        echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
    done
done

echo "[$(date)] === STAGE (a) AMOS multi-seed COMPLETE ===" | tee -a $LOG

# ===================================================================
# (b) SwinUNETR N=20 — third row for Table 7
# ===================================================================
echo "" | tee -a $LOG
echo "===== STAGE (b): SwinUNETR N=20 =====" | tee -a $LOG

SWIN_COMMON="--slices_root data/slices --labels_root data/labels \
             --num_classes 11 --require_full_coverage \
             --mode linear \
             --epochs 25 --bs 1 --crop_size 96 --iters_per_patient 10 \
             --lr 1e-3 --warmup_epochs 3 --eval_every 5"

NAME="swin_lin_n20_ssl"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --pretrained $SWIN_CKPT \
    --n_train_volumes 20 \
    --out runs/$NAME \
    $SWIN_COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

NAME="swin_lin_n20_random"
echo "[$(date)] >>> START $NAME" | tee -a $LOG
python scripts/train_swinunetr.py \
    --n_train_volumes 20 \
    --out runs/$NAME \
    $SWIN_COMMON 2>&1 | tee runs/$NAME.log
echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

echo "[$(date)] === STAGE (b) SwinUNETR N=20 COMPLETE ===" | tee -a $LOG

# ===================================================================
# Summary
# ===================================================================
echo "============================================================" | tee -a $LOG
echo "[$(date)] PHASE-1 ALL RUNS COMPLETE" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os, statistics as st
runs_dir = os.path.expanduser("~/SSLP/runs")

def best(p):
    try:
        log = json.load(open(p))
        return max((e["val_fg"] for e in log if "val_fg" in e), default=None)
    except: return None

print("\n=== AMOS multi-seed (4 seeds per cell, including pre-existing seed 0) ===")
print(f"{'cell':<25} {'mean':<8} {'std':<8} {'n':<3}  seeds")
for n in (20, 50, 100):
    for tag in ("a3", "random"):
        vals = []
        v0 = best(f"{runs_dir}/lin_amos_{tag}_n{n}/log.json")
        if v0 is not None: vals.append(v0)
        for s in (1, 2, 3):
            v = best(f"{runs_dir}/lin_amos_{tag}_n{n}_seed{s}/log.json")
            if v is not None: vals.append(v)
        if vals:
            m = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            label = f"AMOS-{tag} N={n}"
            print(f"{label:<25} {m:.4f}  {sd:.4f}  {len(vals)}    {[f'{v:.4f}' for v in vals]}")

print("\n=== SwinUNETR N=20 (new) + existing N=50, N=100 ===")
print(f"{'cell':<25} {'val_fg':<8}")
for n in (20, 50, 100):
    for tag in ("ssl", "random"):
        v = best(f"{runs_dir}/swin_lin_n{n}_{tag}/log.json")
        label = f"swin_lin_n{n}_{tag}"
        if v is not None:
            print(f"{label:<25} {v:.4f}")
        else:
            print(f"{label:<25} MISSING")
PYEOF

echo "[$(date)] phase-1 queue done" | tee -a $LOG
