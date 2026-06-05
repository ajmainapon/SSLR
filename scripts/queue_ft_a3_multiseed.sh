#!/bin/bash
# Multi-seed fine-tune sweep using LOO A3 backbone (narrow k=3-7, block mask 0.4).
# A3 outperformed v2 in linear probe by 45-124% — this rerun makes A3 the new
# headline SSL backbone for the WACV paper.
#
# 12 runs total: 3 N values × 4 seeds × 1 init (random ft already done with v2 queue).
# Expected wall-clock: ~20 hr.
#
# After completion, compare A3-SSL ft (mean ± std) vs existing Random ft (mean ± std).
# Expected: A3-SSL beats Random by significantly more than v2-SSL did, given A3's
# stronger linear-probe features.

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_ft_a3_multiseed.log' ERR

ulimit -n 65536

CKPT=/home/rmedu-04/SSLP/checkpoints/loo_a3_narrow_k/vit_ep039.pt
LOG=runs/queue_ft_a3_multiseed.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — A3 multi-seed fine-tune" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "Backbone: $CKPT" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

COMMON="--slices_root data/slices --labels_root data/labels \
        --num_classes 11 --require_full_coverage \
        --head conv --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint"

# ===================================================================
# N=20 — 4 seeds × ~30 min = ~2 hr
# ===================================================================
for SEED in 0 1 2 3; do
    NAME="ft_a3_ssl_n20_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_finetune.py \
        --ckpt $CKPT \
        --n_train_volumes 20 \
        --epochs 25 \
        --seed $SEED \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

echo "[$(date)] === N=20 block complete ===" | tee -a $LOG

# ===================================================================
# N=50 — 4 seeds × ~1.5 hr = ~6 hr
# ===================================================================
for SEED in 0 1 2 3; do
    NAME="ft_a3_ssl_n50_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_finetune.py \
        --ckpt $CKPT \
        --n_train_volumes 50 \
        --epochs 30 \
        --seed $SEED \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

echo "[$(date)] === N=50 block complete ===" | tee -a $LOG

# ===================================================================
# N=100 — 4 seeds × ~3 hr = ~12 hr
# ===================================================================
for SEED in 0 1 2 3; do
    NAME="ft_a3_ssl_n100_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_finetune.py \
        --ckpt $CKPT \
        --n_train_volumes 100 \
        --epochs 40 \
        --seed $SEED \
        $COMMON \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

echo "[$(date)] === N=100 block complete ===" | tee -a $LOG

# ===================================================================
# Summary comparison
# ===================================================================
echo "============================================================" | tee -a $LOG
echo "[$(date)] ALL A3 MULTI-SEED RUNS COMPLETE" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os, statistics as st
runs_dir = os.path.expanduser("~/SSLP/runs")

def best(p):
    log = json.load(open(p))
    return max(log, key=lambda r: r["val_fg"])["val_fg"]

print(f"\n{'cell':<22} {'n':<3} {'mean':<8} {'std':<8} seeds")
print("-" * 70)
for n in (20, 50, 100):
    # A3 SSL ft (new)
    vals_a3 = []
    for s in (0,1,2,3):
        p = f"{runs_dir}/ft_a3_ssl_n{n}_bblr1e-4_seed{s}/log.json"
        if os.path.exists(p):
            vals_a3.append(best(p))
    if vals_a3:
        mean = sum(vals_a3)/len(vals_a3)
        std = st.stdev(vals_a3) if len(vals_a3) > 1 else 0.0
        print(f"A3 SSL N={n:<3}            {len(vals_a3):<3} {mean:.4f}  {std:.4f}  {[f'{v:.4f}' for v in vals_a3]}")

    # Existing Random ft
    vals_rnd = []
    rnd_runs = [f"ft_v2_random_n{n}_bblr1e-4"] + [f"ft_v2_random_n{n}_bblr1e-4_seed{s}" for s in (1,2,3)]
    for r in rnd_runs:
        p = f"{runs_dir}/{r}/log.json"
        if os.path.exists(p):
            vals_rnd.append(best(p))
    if vals_rnd:
        mean = sum(vals_rnd)/len(vals_rnd)
        std = st.stdev(vals_rnd) if len(vals_rnd) > 1 else 0.0
        print(f"Random N={n:<3}            {len(vals_rnd):<3} {mean:.4f}  {std:.4f}  {[f'{v:.4f}' for v in vals_rnd]}")

    if vals_a3 and vals_rnd:
        d = sum(vals_a3)/len(vals_a3) - sum(vals_rnd)/len(vals_rnd)
        rel = 100*d/(sum(vals_rnd)/len(vals_rnd))
        print(f"  → A3 SSL vs Random Delta = {d:+.4f}  ({rel:+.2f}%)")
    print()
PYEOF

echo "[$(date)] queue done" | tee -a $LOG
