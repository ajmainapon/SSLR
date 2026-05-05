# SSLR — Zero-Shot Medical World Models

Self-supervised representation learning for medical image segmentation. Adapts ideas
from Zero-shot World Models (Stanford, 2026) from temporal video prediction to
spatial-volumetric continuity in CT scans, with the goal of segmenting organs from
unlabelled data.

## Approach

A Vision Transformer (ViT-B/16) is pretrained on unlabelled CT volumes using a
Joint-Embedding Predictive Architecture (I-JEPA style): given a context slice and a
target slice from the same volume, the model predicts the embeddings of masked
target patches. The hypothesis is that learning anatomical continuity across slices
yields features that transfer to downstream organ segmentation.

## Phases

1. **Self-supervised pretraining** — JEPA on unlabelled CT slice pairs.
2. **Few-shot decoder** — frozen backbone + lightweight segmentation head trained on
   a small set of labelled volumes (linear probe and conv head variants).
3. **Zero-shot causal masking** (planned) — extract organ masks by perturbing latent
   patches and measuring the response of surrounding patches.

## Data

TotalSegmentator (≈1,200 CT volumes, 100+ organ labels). Volumes are HU-windowed
to soft tissue, downscaled to 224×224 in-plane, and stored as per-volume `uint8`
stacks for fast streaming.

## Layout

```
src/
  data/         dataset + preprocessing
  models/       ViT-JEPA backbone, segmentation heads
  train/        pretraining loop
scripts/
  prepare_labels.py   fuse organ NIfTIs into class-indexed masks
  train_decoder.py    few-shot head training
  sanity_check.py     attention / similarity visualizations
  check_*.py          mask alignment and organ overlay diagnostics
preview.py            single-volume NIfTI inspector
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch timm nibabel scipy pandas tqdm matplotlib
```

Expects TotalSegmentator extracted under `totalsegmentator/` and preprocessed slice
stacks under `data/slices/{train,val,test}/`.

## Training

```bash
# Phase 1 — self-supervised pretraining
python -m src.train.pretrain

# Phase 2 — few-shot decoder on a frozen backbone
python scripts/train_decoder.py --ckpt checkpoints/vit_epXXX.pt \
    --slices_root data/slices --labels_root data/labels \
    --num_classes 11 --n_train_volumes 20 --head linear \
    --out runs/probe
```

## Hardware

Trained on a single NVIDIA RTX 3090 Ti (24 GB) with bfloat16 mixed precision and
gradient accumulation. Authored on Apple Silicon.

## Status

Active research. Phase 1 self-supervised pretraining, Phase 2 few-shot
linear-probe evaluation across N ∈ {5, 10, 20, 50, 100}, and a fully-supervised
nnU-Net 2D reference baseline are all complete; pretrained features outperform
random initialization on the standard linear-probe protocol with a monotonically
growing gap, and a label-efficient framing positions the SSL features against
the supervised ceiling. Phase 3 (zero-shot causal masking) is a planned
methods extension. Detailed experimental notes and results are tracked in
the project notes alongside this repository.

## References

* Aw et al., *Zero-shot World Models Are Developmentally Efficient Learners*,
  Stanford University, 2026.
* Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture* (I-JEPA), 2023.
* Wasserthal et al., *TotalSegmentator: Robust Segmentation of 104 Anatomic
  Structures in CT Images*, 2023.
