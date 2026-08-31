#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[$(date -Is)] FAILED at line $LINENO" | tee -a "$QUEUE_LOG"' ERR

ROOT="${SSLR_ROOT:-$HOME/SSLP}"
PYTHON="$ROOT/venv/bin/python"
QUEUE_LOG="$ROOT/runs/queue_capacity_frozen_conv_n20.log"
CKPT="$ROOT/checkpoints/loo_a3_narrow_k/vit_ep039.pt"

cd "$ROOT"
mkdir -p runs

if pgrep -af 'scripts/train_(decoder|finetune)\.py' >/dev/null; then
  echo "A downstream training process is already running; refusing to overlap." >&2
  exit 1
fi

echo "[$(date -Is)] N=20 frozen+Conv capacity-control queue starting" | tee -a "$QUEUE_LOG"
nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu --format=csv,noheader | tee -a "$QUEUE_LOG"

COMMON=(
  --slices_root data/slices
  --labels_root data/labels
  --num_classes 11
  --n_train_volumes 20
  --head conv
  --epochs 50
  --bs 16
  --lr 1e-3
  --wd 1e-4
  --eval_every 5
  --workers 2
)

for seed in 0 1 2 3; do
  patient_file="$ROOT/runs/ft_a3_ssl_n20_bblr1e-4_seed${seed}/train_volumes.json"
  mapfile -t patients < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$patient_file")
  if [[ "${#patients[@]}" != 20 ]]; then
    echo "Expected 20 patients for seed $seed, found ${#patients[@]}" >&2
    exit 1
  fi

  for initialization in ssl random; do
    run_name="conv_a3_${initialization}_n20_seed${seed}"
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
      --patients "${patients[@]}"
      --out "$run_dir"
    )
    if [[ "$initialization" == ssl ]]; then
      command+=(--ckpt "$CKPT")
    fi
    "${command[@]}" 2>&1 | tee "$run_log"
    echo "[$(date -Is)] DONE $run_name" | tee -a "$QUEUE_LOG"
  done
done

echo "[$(date -Is)] ALL 8 N=20 FROZEN+CONV RUNS COMPLETE" | tee -a "$QUEUE_LOG"
