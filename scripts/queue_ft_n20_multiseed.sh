#!/bin/bash
# N=20 multi-seed at lr=1e-4 — completes the cross-N seed coverage.
# seed=0 N=20 is already done (single-seed). This adds seeds 1, 2, 3.
# 6 runs total × ~30 min = ~3 GPU-hours.

set -eo pipefail   # pipefail is CRITICAL — without it, `python ... | tee` swallows
                   # the python exit code on failure and the loop happily advances.
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_ft_n20_multiseed.log' ERR

CKPT=~/SSLP/checkpoints/vit_ep039.pt
LOG=runs/queue_ft_n20_multiseed.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — N=20 multi-seed @ lr=1e-4" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

for SEED in 1 2 3; do

    # SSL @ N=20, seed=$SEED
    NAME="ft_v2_ssl_n20_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_finetune.py \
        --ckpt $CKPT \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 20 --require_full_coverage \
        --head conv --epochs 25 --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint \
        --seed $SEED \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

    # random @ N=20, seed=$SEED
    NAME="ft_v2_random_n20_bblr1e-4_seed${SEED}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_finetune.py \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 20 --require_full_coverage \
        --head conv --epochs 25 --bs 8 \
        --lr_head 1e-3 --lr_backbone 1e-4 --llrd 0.75 \
        --warmup_epochs 3 --grad_checkpoint \
        --seed $SEED \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG

done

echo "============================================================" | tee -a $LOG
echo "[$(date)] N=20 multi-seed COMPLETE — summary:" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os, statistics as st
runs_dir = os.path.expanduser("~/SSLP/runs")

cells = {
    ("ssl",    20): ["ft_v2_ssl_n20_bblr1e-4"]    + [f"ft_v2_ssl_n20_bblr1e-4_seed{s}"    for s in (1,2,3)],
    ("random", 20): ["ft_v2_random_n20_bblr1e-4"] + [f"ft_v2_random_n20_bblr1e-4_seed{s}" for s in (1,2,3)],
}

print(f"\n{'cell':<22} {'n':<3} {'mean':<8} {'std':<8} seeds")
print("-" * 70)
for (init, n), runs in cells.items():
    vals = []
    for r in runs:
        p = f"{runs_dir}/{r}/log.json"
        if os.path.exists(p):
            log = json.load(open(p))
            peak = max(log, key=lambda r: r["val_fg"])
            vals.append(peak["val_fg"])
    if vals:
        mean = sum(vals) / len(vals)
        std = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{init} N={n:<3}            {len(vals):<3} {mean:.4f}  {std:.4f}  {[f'{v:.4f}' for v in vals]}")

print("\nN=20 SSL vs Random:")
for n in (20,):
    ssl_vals = []
    rnd_vals = []
    for r in cells[("ssl", n)]:
        p = f"{runs_dir}/{r}/log.json"
        if os.path.exists(p):
            ssl_vals.append(max(json.load(open(p)), key=lambda r: r["val_fg"])["val_fg"])
    for r in cells[("random", n)]:
        p = f"{runs_dir}/{r}/log.json"
        if os.path.exists(p):
            rnd_vals.append(max(json.load(open(p)), key=lambda r: r["val_fg"])["val_fg"])
    if ssl_vals and rnd_vals:
        d = sum(ssl_vals)/len(ssl_vals) - sum(rnd_vals)/len(rnd_vals)
        print(f"  SSL mean={sum(ssl_vals)/len(ssl_vals):.4f}  Random mean={sum(rnd_vals)/len(rnd_vals):.4f}  Delta={d:+.4f}  ({100*d/(sum(rnd_vals)/len(rnd_vals)):+.2f}%)")
PYEOF

echo "[$(date)] queue done" | tee -a $LOG
