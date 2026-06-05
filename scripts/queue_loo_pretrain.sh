#!/bin/bash
# Leave-one-out ablation pretrains.
# A1 = random per-patch masking (vs v2's multi-block) — diagnoses masking choice
# A3 = narrow k_range=(3,7)   (vs v2's (8,20))       — diagnoses slice-gap choice
#
# Each pretrain is 40 epochs ≈ 4 GPU-days at standard v2 settings.
# Total queue: ~8 GPU-days. Plan accordingly.
#
# IMPORTANT: this only runs the PRETRAIN. After completion, launch the
# downstream linear-probe sweep separately using each LOO checkpoint:
#   python scripts/train_decoder.py --ckpt checkpoints/loo_a1_random_mask/vit_ep039.pt ...
#   python scripts/train_decoder.py --ckpt checkpoints/loo_a3_narrow_k/vit_ep039.pt ...
#
# requires the patched pretrain.py with --out_dir, --data_root, --k_range, --no_augment

set -eo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO" | tee -a runs/queue_loo_pretrain.log' ERR

# Belt-and-suspenders against "Too many open files" errors. Default Linux
# soft limit is 1024; SliceTriplet + 4 DataLoader workers + tensor sharing
# can blow past this. 65536 is safe for the duration of an 8-day queue.
ulimit -n 65536

LOG=runs/queue_loo_pretrain.log
mkdir -p runs

echo "============================================================" | tee -a $LOG
echo "[$(date)] queue starting (PID $$) — LOO pretrain A1 + A3" | tee -a $LOG
echo "============================================================" | tee -a $LOG
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv | tee -a $LOG

# ------------------------------------------------------------------
# A1: random per-patch masking (v1's masking scheme) — ~4 GPU-days
# ------------------------------------------------------------------
echo "[$(date)] >>> START loo_a1_random_mask (random masking, k=8-20, with aug)" | tee -a $LOG
python -m src.train.pretrain \
    --epochs 40 \
    --mask_mode random \
    --mask_ratio 0.75 \
    --data_root data/slices/train \
    --k_range 8 20 \
    --out_dir checkpoints/loo_a1_random_mask 2>&1 | tee runs/loo_a1_random_mask.log
echo "[$(date)] <<< DONE  loo_a1_random_mask" | tee -a $LOG

# ------------------------------------------------------------------
# A3: narrow slice gap k=(3,7) (v1's gap) — ~4 GPU-days
# ------------------------------------------------------------------
echo "[$(date)] >>> START loo_a3_narrow_k (block mask 0.4, k=3-7, with aug)" | tee -a $LOG
python -m src.train.pretrain \
    --epochs 40 \
    --mask_mode block \
    --mask_ratio 0.4 \
    --n_blocks 4 \
    --data_root data/slices/train \
    --k_range 3 7 \
    --out_dir checkpoints/loo_a3_narrow_k 2>&1 | tee runs/loo_a3_narrow_k.log
echo "[$(date)] <<< DONE  loo_a3_narrow_k" | tee -a $LOG

echo "============================================================" | tee -a $LOG
echo "[$(date)] BOTH LOO PRETRAINS COMPLETE" | tee -a $LOG
echo "  - checkpoints/loo_a1_random_mask/vit_ep039.pt" | tee -a $LOG
echo "  - checkpoints/loo_a3_narrow_k/vit_ep039.pt" | tee -a $LOG
echo "" | tee -a $LOG
echo "Next: run linear probe sweeps with each checkpoint." | tee -a $LOG
echo "  python scripts/train_decoder.py --ckpt checkpoints/loo_a1_random_mask/vit_ep039.pt \\" | tee -a $LOG
echo "      --slices_root data/slices --labels_root data/labels --num_classes 11 \\" | tee -a $LOG
echo "      --n_train_volumes 50 --require_full_coverage --head linear --epochs 50 \\" | tee -a $LOG
echo "      --out runs/lin_loo_a1_n50" | tee -a $LOG
echo "[$(date)] queue done" | tee -a $LOG
