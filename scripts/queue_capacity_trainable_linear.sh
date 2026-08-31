#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[$(date -Is)] FAILED at line $LINENO" | tee -a "$QUEUE_LOG"' ERR

ROOT="${SSLR_ROOT:-$HOME/SSLP_capacity}"
PYTHON="$ROOT/venv/bin/python"
QUEUE_LOG="$ROOT/runs/queue_capacity_trainable_linear.log"
CKPT="$ROOT/checkpoints/loo_a3_narrow_k/vit_ep039.pt"

cd "$ROOT"
mkdir -p runs

if pgrep -af 'scripts/train_(decoder|finetune)\.py' >/dev/null; then
  echo "A downstream training process is already running; refusing to overlap." >&2
  exit 1
fi

echo "[$(date -Is)] trainable+linear capacity-control queue starting" | tee -a "$QUEUE_LOG"
nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu --format=csv,noheader | tee -a "$QUEUE_LOG"

COMMON=(
  --slices_root data/slices
  --labels_root data/labels
  --num_classes 11
  --n_train_volumes 50
  --head linear
  --epochs 30
  --bs 8
  --lr_head 1e-3
  --lr_backbone 1e-4
  --head_wd 1e-4
  --backbone_wd 0.05
  --llrd 0.75
  --warmup_epochs 3
  --grad_checkpoint
  --eval_every 2
  --workers 2
)

for seed in 0 1 2 3; do
  patient_file="$ROOT/patient_lists/seed${seed}.json"
  mapfile -t patients < <("$PYTHON" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$patient_file")
  if [[ "${#patients[@]}" != 50 ]]; then
    echo "Expected 50 patients for seed $seed, found ${#patients[@]}" >&2
    exit 1
  fi

  for initialization in ssl random; do
    run_name="ftlin_a3_${initialization}_n50_seed${seed}"
    run_dir="$ROOT/runs/$run_name"
    run_log="$ROOT/runs/${run_name}.log"
    if [[ -e "$run_dir" ]]; then
      echo "Refusing to overwrite existing $run_dir" >&2
      exit 1
    fi

    echo "[$(date -Is)] START $run_name" | tee -a "$QUEUE_LOG"
    command=(
      "$PYTHON" scripts/train_finetune.py
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

echo "[$(date -Is)] ALL 8 TRAINABLE+LINEAR RUNS COMPLETE" | tee -a "$QUEUE_LOG"
