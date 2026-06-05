"""Preprocess AMOS22 MR volumes (IDs 0501-0600) into the same uint8 slice
format used by our TotalSegmentator and AMOS-CT pipelines.

CROSS-MODALITY CAVEAT (important — flag in paper):
    Unlike CT, MR has no fixed HU scale. Each volume's intensities depend on
    scanner, sequence, and acquisition parameters. We use per-volume
    1st–99th percentile windowing followed by [0,255] normalisation as the
    standard MR-prep convention. The output uint8 format matches CT byte-for-
    byte so the A3 backbone (trained on CT) can be applied directly without
    architecture changes.

Output layout (mirror of data_amos/slices/):
    data_mri/slices/train/{pid}.npy   uint8 (Z, 224, 224)
    data_mri/slices/val/{pid}.npy     uint8 (Z, 224, 224)

The HU equivalent (now percentile windowing) differs from CT; everything else
(resize, dtype, transpose) is identical to preprocess_amos.py so the SSL
backbone sees only the *modality* shift, not an additional pipeline shift.

Usage on the RTX box:
    cd ~/SSLP
    python scripts/preprocess_amos_mri.py --split train
    python scripts/preprocess_amos_mri.py --split val
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants -- spatial layout must match preprocess_amos.py (CT) byte-for-byte
# ---------------------------------------------------------------------------
TARGET_HW = 224
PCT_LO = 1.0
PCT_HI = 99.0

# AMOS22 MR IDs run 0501..0600. Same official split scheme as CT:
#   --split train -> imagesTr/, labelsTr/   (~40 MR volumes after coverage filter)
#   --split val   -> imagesVa/, labelsVa/   (~20 MR volumes)
SPLIT_TO_DIR = {"train": "imagesTr", "val": "imagesVa"}
MIN_MR_ID = 501
MAX_MR_ID = 600

ROOT = Path("amos")
OUT  = Path("data_mri/slices")


def mri_to_volume(mr_path: Path) -> np.ndarray:
    """Load MR NIfTI, per-volume percentile window, normalize to uint8, resize
    HW to 224, transpose to (Z, 224, 224). Returns uint8 array.

    Per-volume percentile windowing handles the lack of an absolute MR scale.
    Same as standard MR preprocessing in MONAI / Models Genesis / SwinUNETR.
    """
    nii = nib.load(str(mr_path))
    vol = nii.get_fdata().astype(np.float32)            # (H, W, Z)
    H, W, Z = vol.shape

    # Per-volume 1-99 percentile windowing
    lo, hi = np.percentile(vol, (PCT_LO, PCT_HI))
    vol = np.clip(vol, lo, hi)
    if hi > lo:
        vol = (vol - lo) / (hi - lo)                    # [0, 1]
    else:
        vol = np.zeros_like(vol)
    vol = (vol * 255.0).astype(np.uint8)                # uint8

    # Resize HW to 224 (preserve Z), bilinear — matches CT pipeline
    if H != TARGET_HW or W != TARGET_HW:
        vol = zoom(vol, (TARGET_HW / H, TARGET_HW / W, 1.0), order=1)
        vol = vol.astype(np.uint8)

    return np.ascontiguousarray(vol.transpose(2, 0, 1))   # (Z, 224, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="AMOS22 root dir (with imagesTr/, imagesVa/)")
    ap.add_argument("--out",  default=str(OUT),  help="Output slices root")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = root / SPLIT_TO_DIR[args.split]
    if not images_dir.exists():
        raise FileNotFoundError(f"AMOS {SPLIT_TO_DIR[args.split]}/ not found at {images_dir}")

    todo = []
    all_files = sorted(images_dir.glob("amos_*.nii.gz"))
    for mr_path in all_files:
        pid = mr_path.name.replace(".nii.gz", "")        # amos_0501
        try:
            i = int(pid.split("_")[1])
        except ValueError:
            continue
        if i < MIN_MR_ID or i > MAX_MR_ID:               # MR-only window
            continue
        out_path = out_root / f"{pid}.npy"
        if not out_path.exists():
            todo.append((pid, mr_path, out_path))

    print(f"[{args.split}] {len(todo)} MR volumes to process "
          f"(from {len(all_files)} files in {SPLIT_TO_DIR[args.split]}/, "
          f"MR ID window {MIN_MR_ID}-{MAX_MR_ID})")

    for pid, mr_path, out_path in tqdm(todo):
        try:
            vol = mri_to_volume(mr_path)
            np.save(out_path, vol)
            print(f"[done] {pid} shape={vol.shape}")
        except Exception as e:
            print(f"[FAIL] {pid}: {e}")


if __name__ == "__main__":
    main()
