#!/bin/bash
# nnU-Net v2 baseline at N=50 — supervised ceiling for the v2 SSL paper.
#
# Uses the SAME 50 training patients as runs/lin_v2_ssl_n50, predicts on the
# same val patients used by train_decoder.py. Train in 2D mode for speed and
# resolution-matching with our 2.5D evaluation.
#
# Run from ~/SSLP. Assumes venv is activated.
#
# Cost estimate: ~3-5 hr for plan_and_preprocess + 2D training on RTX 3090 Ti.

set -e

DATASET_ID=1
DATASET_DIR_NAME="Dataset001_Organ"
PROJECT_ROOT="$HOME/SSLP"
TRAIN_VOLS_JSON="runs/lin_v2_ssl_n50/train_volumes.json"

# nnU-Net v2 wants these env vars; export them in the current shell.
export nnUNet_raw="$PROJECT_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Step 0 — install nnU-Net v2 if needed (idempotent)
if ! python -c "import nnunetv2" 2>/dev/null; then
    echo "[install] pip install nnunetv2"
    pip install nnunetv2
fi

# Step 1 — convert SSLP layout into nnU-Net Dataset001 format
DATASET_DIR="$nnUNet_raw/$DATASET_DIR_NAME"
if [ ! -f "$DATASET_DIR/dataset.json" ]; then
    echo "[convert] building $DATASET_DIR_NAME..."
    python scripts/make_nnunet_dataset.py \
        --train_volumes_json "$TRAIN_VOLS_JSON" \
        --val_labels_dir data/labels/val \
        --ts_root totalsegmentator \
        --out "$DATASET_DIR"
else
    echo "[skip] $DATASET_DIR/dataset.json exists -- not re-converting"
fi

# Step 2 — disk + integrity check before launching
echo "[check] free space (need ~20 GB for preprocessed):"
df -h "$nnUNet_preprocessed" | tail -1

# Step 3 — plan + preprocess (fingerprint, target spacing, normalization, etc.)
echo "[step 3] nnUNetv2_plan_and_preprocess -d $DATASET_ID"
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

# Step 4 — train 2D on all 50 patients (no cross-validation)
echo "[step 4] nnUNetv2_train $DATASET_ID 2d all"
nnUNetv2_train $DATASET_ID 2d all

# Step 5 — predict on val (test) set
PREDICT_OUT="$PROJECT_ROOT/predictions/nnunet_2d_n50"
mkdir -p "$PREDICT_OUT"
echo "[step 5] nnUNetv2_predict -> $PREDICT_OUT"
nnUNetv2_predict \
    -i "$DATASET_DIR/imagesTs" \
    -o "$PREDICT_OUT" \
    -d $DATASET_ID -c 2d -f all

echo ""
echo "[done] nnU-Net 2D supervised baseline complete."
echo "  Predictions: $PREDICT_OUT"
echo "  Next: write scripts/eval_nnunet_predictions.py to resize predictions"
echo "        to 224x224 and compute per-class Dice against data/labels/val/."
