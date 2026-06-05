#!/bin/bash
# Resume nnU-Net 2D supervised baseline (Dataset001_Organ, default 1000-epoch
# trainer) from the epoch-150 checkpoint. The original run OOM-killed at epoch
# 181 because the default 12 data-augmentation workers exceed the 30 GB host
# RAM. Fix: cap nnUNet_n_proc_DA=6. Canonical default config otherwise, so the
# resulting Dice stays the standard "nnU-Net default" supervised reference and
# the "interrupted at epoch 181" caveat can be dropped once it completes.
set -e
cd "$HOME/SSLP"
source venv/bin/activate

export nnUNet_raw="$HOME/SSLP/nnUNet_raw"
export nnUNet_preprocessed="$HOME/SSLP/nnUNet_preprocessed"
export nnUNet_results="$HOME/SSLP/nnUNet_results"
export nnUNet_n_proc_DA=6

echo "[$(date)] START resume: nnUNetv2_train 1 2d all --c  (n_proc_DA=$nnUNet_n_proc_DA)"
nnUNetv2_train 1 2d all --c
echo "[$(date)] DONE nnUNet train (exit $?)"
