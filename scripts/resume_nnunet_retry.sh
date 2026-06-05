#!/bin/bash
# Resume nnU-Net 2D training to completion (epoch 900 -> 1000) with a self-heal
# retry loop. A transient CUDA device-side assert killed the run once at epoch
# 910 on already-converged weights (consumer 3090 Ti, no ECC). Each --c resume
# picks up from checkpoint_latest, so up to 4 attempts ride through transient
# faults without manual intervention. NOT set -e: we inspect exit codes.
cd "$HOME/SSLP"
source venv/bin/activate

export nnUNet_raw="$HOME/SSLP/nnUNet_raw"
export nnUNet_preprocessed="$HOME/SSLP/nnUNet_preprocessed"
export nnUNet_results="$HOME/SSLP/nnUNet_results"
export nnUNet_n_proc_DA=6

FINAL="nnUNet_results/Dataset001_Organ/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth"

for attempt in 1 2 3 4; do
  if [ -f "$FINAL" ]; then
    echo "[$(date)] checkpoint_final.pth already present -- training complete."
    break
  fi
  echo "[$(date)] ===== resume attempt $attempt: nnUNetv2_train 1 2d all --c ====="
  nnUNetv2_train 1 2d all --c
  rc=$?
  echo "[$(date)] attempt $attempt exited rc=$rc"
  if [ -f "$FINAL" ]; then
    echo "[$(date)] DONE: checkpoint_final.pth written after attempt $attempt."
    break
  fi
  echo "[$(date)] no final checkpoint yet; will retry after brief GPU settle."
  sleep 15
done

if [ -f "$FINAL" ]; then
  echo "[$(date)] DONE nnUNet train (completed)"
else
  echo "[$(date)] GAVE UP after retries -- final checkpoint missing"
fi
