#!/bin/bash
# LOO linear-probe sweep — generates the §5.3 ablation table.
# Two backbones (A1, A3) × five N values = 10 linear probes.
# Each probe = frozen ViT + 1x1 conv LinearSegHead, 50 epochs, --require_full_coverage.
# Mirrors the v2 baseline lin_v2_ssl_n* protocol for direct comparison.
#
# Expected wall-clock per probe:
#   N=5    ~10 min     N=10   ~15 min    N=20   ~20 min
#   N=50   ~30 min     N=100  ~45 min
# Total: ~2 hr per backbone × 2 backbones = ~4 hr

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_loo_linprobe.log' ERR

ulimit -n 65536

LOG=runs/queue_loo_linprobe.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — LOO linear-probe sweep" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

# Shared decoder hyperparams — identical to v2 baseline (lin_v2_ssl_n*)
DECODER_ARGS="--slices_root data/slices --labels_root data/labels \
              --num_classes 11 --require_full_coverage \
              --head linear --epochs 50 --bs 16 \
              --lr 1e-3 --eval_every 5"

# ===================================================================
# A1 backbone (random per-patch masking) × N ∈ {5,10,20,50,100}
# ===================================================================
CKPT_A1=/home/rmedu-04/SSLP/checkpoints/loo_a1_random_mask/vit_ep039.pt

for N in 5 10 20 50 100; do
    NAME="lin_loo_a1_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --ckpt $CKPT_A1 \
        $DECODER_ARGS \
        --n_train_volumes $N \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

echo "[$(date)] === LOO A1 linear-probe sweep complete ===" | tee -a $LOG

# ===================================================================
# A3 backbone (narrow k=3-7) × N ∈ {5,10,20,50,100}
# ===================================================================
CKPT_A3=/home/rmedu-04/SSLP/checkpoints/loo_a3_narrow_k/vit_ep039.pt

for N in 5 10 20 50 100; do
    NAME="lin_loo_a3_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --ckpt $CKPT_A3 \
        $DECODER_ARGS \
        --n_train_volumes $N \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

echo "[$(date)] === LOO A3 linear-probe sweep complete ===" | tee -a $LOG

# ===================================================================
# Summary — ablation comparison table
# ===================================================================
echo "============================================================" | tee -a $LOG
echo "[$(date)] ALL LOO LINEAR-PROBE RUNS COMPLETE" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF' | tee -a $LOG
import json, os
runs_dir = os.path.expanduser("~/SSLP/runs")

def best(path):
    log = json.load(open(path))
    peak = max(log, key=lambda r: r["val_fg"])
    return peak["val_fg"], peak["ep"]

print(f"\n{'N':<5} {'v2 SSL':<10} {'A1':<10} {'A3':<10} {'v2-A1':<10} {'v2-A3':<10}")
print("-" * 70)
for n in (5, 10, 20, 50, 100):
    v2 = best(f"{runs_dir}/lin_v2_ssl_n{n}/log.json") if os.path.exists(f"{runs_dir}/lin_v2_ssl_n{n}/log.json") else (None, None)
    a1 = best(f"{runs_dir}/lin_loo_a1_n{n}/log.json")
    a3 = best(f"{runs_dir}/lin_loo_a3_n{n}/log.json")
    v2s = f"{v2[0]:.4f}" if v2[0] is not None else "  ----  "
    delta_a1 = f"{v2[0]-a1[0]:+.4f}" if v2[0] is not None else "  ----  "
    delta_a3 = f"{v2[0]-a3[0]:+.4f}" if v2[0] is not None else "  ----  "
    print(f"{n:<5} {v2s:<10} {a1[0]:.4f}     {a3[0]:.4f}     {delta_a1:<10} {delta_a3:<10}")
PYEOF

echo "[$(date)] queue done" | tee -a $LOG
