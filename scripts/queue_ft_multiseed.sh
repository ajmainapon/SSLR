#!/bin/bash
# Multi-seed confirmation at the headline operating point (lr_backbone=1e-4).
# 12 runs total: 3 new seeds × 2 inits × 2 N values.
# seed=0 results already exist from previous queues — combined with these 3
# gives 4-seed statistics per cell.
#
# Order: N=100 first (paper-critical), N=50 second. Within each N, seeds and
# inits are interleaved so that early termination still yields matched pairs.
#
# Expected wall-clock: ~25 hours.

set -eo pipefail   # pipefail is CRITICAL — without it, `python ... | tee` swallows python's exit code,
                   # and a killed python (via pkill -9) would silently advance to the next loop iteration.
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_ft_multiseed.log' ERR

CKPT=~/SSLP/checkpoints/vit_ep039.pt
LOG=runs/queue_ft_multiseed.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — multi-seed @ lr=1e-4"     | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu \
           --format=csv | tee -a $LOG

# ===================================================================
# N=100 BLOCK — paper-critical
# 6 runs × ~2.7 hr = ~16 hr
# ===================================================================

for SEED in 1 2 3; do

    # --- SSL @ N=100, seed=$SEED ---
    NAME="ft_v2_ssl_n100_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME"  | tee -a $LOG
    python scripts/train_finetune.py \
        --ckpt $CKPT \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 100 --require_full_coverage \
        --head conv --epochs 30 --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint \
        --seed $SEED \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME"  | tee -a $LOG

    # --- random @ N=100, seed=$SEED ---
    NAME="ft_v2_random_n100_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME"  | tee -a $LOG
    python scripts/train_finetune.py \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 100 --require_full_coverage \
        --head conv --epochs 30 --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint \
        --seed $SEED \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME"  | tee -a $LOG

done

echo "[$(date)] === N=100 multi-seed block complete ===" | tee -a $LOG

# ===================================================================
# N=50 BLOCK — Tier 1
# 6 runs × ~1.4 hr = ~8.4 hr
# ===================================================================

for SEED in 1 2 3; do

    # --- SSL @ N=50, seed=$SEED ---
    NAME="ft_v2_ssl_n50_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME"  | tee -a $LOG
    python scripts/train_finetune.py \
        --ckpt $CKPT \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \
        --head conv --epochs 30 --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint \
        --seed $SEED \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME"  | tee -a $LOG

    # --- random @ N=50, seed=$SEED ---
    NAME="ft_v2_random_n50_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME"  | tee -a $LOG
    python scripts/train_finetune.py \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \
        --head conv --epochs 30 --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint \
        --seed $SEED \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME"  | tee -a $LOG

done

echo "[$(date)] === N=50 multi-seed block complete ===" | tee -a $LOG

# ===================================================================
# Summary
# ===================================================================
echo "============================================================" | tee -a $LOG
echo "[$(date)] ALL MULTI-SEED RUNS COMPLETE"                       | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os, statistics as st
runs_dir = os.path.expanduser("~/SSLP/runs")

def best_val(path):
    log = json.load(open(path))
    peak = max(log, key=lambda r: r["val_fg"])
    return peak["val_fg"], peak["ep"]

cells = {
    ("ssl",    100): ["ft_v2_ssl_n100_bblr1e-4"]    + [f"ft_v2_ssl_n100_bblr1e-4_seed{s}"    for s in (1,2,3)],
    ("random", 100): ["ft_v2_random_n100_bblr1e-4"] + [f"ft_v2_random_n100_bblr1e-4_seed{s}" for s in (1,2,3)],
    ("ssl",    50):  ["ft_v2_ssl_n50_bblr1e-4"]     + [f"ft_v2_ssl_n50_bblr1e-4_seed{s}"     for s in (1,2,3)],
    ("random", 50):  ["ft_v2_random_n50_bblr1e-4"]  + [f"ft_v2_random_n50_bblr1e-4_seed{s}"  for s in (1,2,3)],
}

print(f"\n{'cell':<22} {'n':<3} {'mean':<8} {'std':<8} {'min':<8} {'max':<8}  seeds")
print("-" * 80)
for (init, n), runs in cells.items():
    vals = []
    for r in runs:
        p = f"{runs_dir}/{r}/log.json"
        if os.path.exists(p):
            v, _ = best_val(p)
            vals.append(v)
    if vals:
        mean = sum(vals)/len(vals)
        std = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{init} N={n:<3}            {len(vals):<3} {mean:.4f}  {std:.4f}  {min(vals):.4f}  {max(vals):.4f}  {[f'{v:.4f}' for v in vals]}")

print("\nSSL vs Random:")
for n in (100, 50):
    ssl_vals = [best_val(f"{runs_dir}/{r}/log.json")[0] for r in cells[("ssl", n)] if os.path.exists(f"{runs_dir}/{r}/log.json")]
    rnd_vals = [best_val(f"{runs_dir}/{r}/log.json")[0] for r in cells[("random", n)] if os.path.exists(f"{runs_dir}/{r}/log.json")]
    if ssl_vals and rnd_vals:
        d = sum(ssl_vals)/len(ssl_vals) - sum(rnd_vals)/len(rnd_vals)
        print(f"  N={n}: SSL mean={sum(ssl_vals)/len(ssl_vals):.4f}  Random mean={sum(rnd_vals)/len(rnd_vals):.4f}  Delta={d:+.4f}  ({100*d/(sum(rnd_vals)/len(rnd_vals)):+.2f}%)")
PYEOF

echo "[$(date)] queue done" | tee -a $LOG
