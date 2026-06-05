#!/bin/bash
# a5 (loo_a5_dense_narrow = random 0.75 masking + narrow k=3-7) linear-probe sweep.
# The "combined inversion": A1 masking + A3 gap. Tests whether inverting BOTH
# I-JEPA choices beats the single-factor A3 backbone.
# Frozen ViT + 1x1 conv LinearSegHead, 50 ep, --require_full_coverage.
# Mirrors scripts/queue_loo_linprobe.sh byte-for-byte (decoder args identical).
set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_a5_linprobe.log' ERR
ulimit -n 65536
source ~/SSLP/venv/bin/activate
LOG=runs/queue_a5_linprobe.log
mkdir -p runs
echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — a5 linear-probe sweep" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

DECODER_ARGS="--slices_root data/slices --labels_root data/labels \
              --num_classes 11 --require_full_coverage \
              --head linear --epochs 50 --bs 16 \
              --lr 1e-3 --eval_every 5"
CKPT_A5=/home/rmedu-04/SSLP/checkpoints/loo_a5_dense_narrow/vit_ep039.pt

for N in 5 10 20 50 100; do
    NAME="lin_a5_n${N}"
    echo "[$(date)] >>> START $NAME" | tee -a $LOG
    python scripts/train_decoder.py \
        --ckpt $CKPT_A5 \
        $DECODER_ARGS \
        --n_train_volumes $N \
        --out runs/$NAME 2>&1 | tee runs/$NAME.log
    echo "[$(date)] <<< DONE  $NAME" | tee -a $LOG
done

echo "============================================================" | tee -a $LOG
echo "[$(date)] ALL a5 LINEAR-PROBE RUNS COMPLETE" | tee -a $LOG
echo "============================================================" | tee -a $LOG

python3 <<'PYEOF'
import json, os
rd = os.path.expanduser("~/SSLP/runs")
def best(p):
    if not os.path.exists(p): return None
    log = json.load(open(p))
    return max(r["val_fg"] for r in log)
print(f"\n{'N':<5} {'v2 SSL':<9} {'A3':<9} {'a5(comb)':<9} {'a5-A3':<10} {'a5 vs A3 %':<10}")
print("-"*62)
for n in (5,10,20,50,100):
    v2 = best(f"{rd}/lin_v2_ssl_n{n}/log.json")
    a3 = best(f"{rd}/lin_loo_a3_n{n}/log.json")
    a5 = best(f"{rd}/lin_a5_n{n}/log.json")
    v2s = f"{v2:.4f}" if v2 else "  ----"
    a3s = f"{a3:.4f}" if a3 else "  ----"
    a5s = f"{a5:.4f}" if a5 else "  ----"
    d   = f"{a5-a3:+.4f}" if (a5 and a3) else "  ----"
    pct = f"{100*(a5-a3)/a3:+.1f}%" if (a5 and a3) else "  ----"
    print(f"{n:<5} {v2s:<9} {a3s:<9} {a5s:<9} {d:<10} {pct:<10}")
print("\n(a5 = random-0.75 masking + narrow k=3-7 = BOTH inversions; A3 = gap-only.)")
print("If a5-A3 > 0 at most N: the combined recipe wins -> stronger thesis.")
PYEOF
echo "[$(date)] summary printed above" | tee -a $LOG
