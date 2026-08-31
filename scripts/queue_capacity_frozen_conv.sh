#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[$(date -Is)] FAILED at line $LINENO" | tee -a "$QUEUE_LOG"' ERR

ROOT="${SSLR_ROOT:-$HOME/SSLP}"
PYTHON="$ROOT/venv/bin/python"
QUEUE_LOG="$ROOT/runs/queue_capacity_frozen_conv.log"
CKPT="$ROOT/checkpoints/loo_a3_narrow_k/vit_ep039.pt"

cd "$ROOT"
mkdir -p runs

if pgrep -af 'scripts/train_(decoder|finetune)\.py' >/dev/null; then
  echo "A downstream training process is already running; refusing to overlap." >&2
  exit 1
fi

echo "[$(date -Is)] frozen+Conv capacity-control queue starting" | tee -a "$QUEUE_LOG"
nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu --format=csv,noheader | tee -a "$QUEUE_LOG"

COMMON=(
  --slices_root data/slices
  --labels_root data/labels
  --num_classes 11
  --n_train_volumes 50
  --require_full_coverage
  --head conv
  --epochs 50
  --bs 16
  --lr 1e-3
  --wd 1e-4
  --eval_every 5
  --workers 2
)

for seed in 0 1 2 3; do
  for initialization in ssl random; do
    run_name="conv_a3_${initialization}_n50_seed${seed}"
    run_dir="$ROOT/runs/$run_name"
    run_log="$ROOT/runs/${run_name}.log"
    if [[ -e "$run_dir" ]]; then
      echo "Refusing to overwrite existing $run_dir" >&2
      exit 1
    fi

    echo "[$(date -Is)] START $run_name" | tee -a "$QUEUE_LOG"
    command=(
      "$PYTHON" scripts/train_decoder.py
      "${COMMON[@]}"
      --seed "$seed"
      --out "$run_dir"
    )
    if [[ "$initialization" == ssl ]]; then
      command+=(--ckpt "$CKPT")
    fi
    "${command[@]}" 2>&1 | tee "$run_log"
    echo "[$(date -Is)] DONE $run_name" | tee -a "$QUEUE_LOG"
  done
done

echo "[$(date -Is)] ALL 8 FROZEN+CONV RUNS COMPLETE" | tee -a "$QUEUE_LOG"
