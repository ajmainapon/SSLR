"""Preprocess AMOS22 Task-1 CT volumes into the same uint8 slice format
used by our TotalSegmentator pipeline.

Output layout (mirror of data/slices/):
    data_amos/slices/train/{pid}.npy   uint8 (Z, 224, 224)
    data_amos/slices/val/{pid}.npy     uint8 (Z, 224, 224)

The HU window, resize, dtype, and array transpose MUST match the existing
preprocess.py used for TotalSegmentator -- otherwise the SSL backbone sees
a covariate shift on top of the dataset shift, and the cross-dataset claim
breaks. (See CLAUDE.md: "Data alignment is fragile" -- this is the same lesson.)

Usage on the RTX box:
    # First, download AMOS22 Task-1 raw archive into ~/SSLP/amos/:
    #   https://amos22.grand-challenge.org/   (Task 1 = CT only)
    # Expected layout after extraction:
    #   ~/SSLP/amos/imagesTr/amos_{0001..0500}.nii.gz   (CT volumes)
    #   ~/SSLP/amos/labelsTr/amos_{0001..0500}.nii.gz   (multi-organ masks)
    # AMOS naming convention: IDs 0001-0500 are CT; 0501-0600 are MRI (skip).

    cd ~/SSLP
    python SSLR/scripts/preprocess_amos.py --split train
    python SSLR/scripts/preprocess_amos.py --split val
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants -- MUST match src/data/preprocess.py for TotalSegmentator
# ---------------------------------------------------------------------------
TARGET_HW = 224
HU_LO = -1000
HU_HI = 400          # soft-tissue window (matches preprocess.py)

# AMOS22 Task-1 official split: ~200 labeled CT in imagesTr/, ~100 in imagesVa/.
# IDs are NON-contiguous (test split takes some IDs). We use AMOS's own split:
#   --split train -> imagesTr/, labelsTr/
#   --split val   -> imagesVa/, labelsVa/
# Inside each folder we glob all amos_*.nii.gz with ID <= 500 (CT-only; >=501 is MRI).
SPLIT_TO_DIR = {"train": "imagesTr", "val": "imagesVa"}
MAX_CT_ID = 500

# Paths -- override with CLI if your layout differs
ROOT = Path("amos")                          # input
OUT  = Path("data_amos/slices")              # output


def amos_to_volume(ct_path: Path) -> np.ndarray:
    """Load AMOS CT NIfTI, HU window, normalize to uint8, resize HW to 224,
    transpose to (Z, 224, 224). Returns uint8 array.

    NOTE: AMOS CT volumes have CT IDs only (no canonicalize), same as our
    TotalSegmentator pipeline. Do NOT call nib.as_closest_canonical().
    """
    nii = nib.load(str(ct_path))
    vol = nii.get_fdata().astype(np.float32)        # (H, W, Z) in HU
    H, W, Z = vol.shape

    # HU window -> [0, 1] -> uint8
    vol = np.clip(vol, HU_LO, HU_HI)
    vol = (vol - HU_LO) / (HU_HI - HU_LO)           # [0, 1]
    vol = (vol * 255.0).astype(np.uint8)            # uint8

    # Resize HW to 224, preserve Z (matches TotalSeg pipeline exactly)
    if H != TARGET_HW or W != TARGET_HW:
        vol = zoom(vol, (TARGET_HW / H, TARGET_HW / W, 1.0), order=1)
        vol = vol.astype(np.uint8)

    # Final shape (Z, 224, 224) — same as TotalSeg slices
    return np.ascontiguousarray(vol.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="AMOS22 root dir (contains imagesTr/)")
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
    for ct_path in all_files:
        pid = ct_path.name.replace(".nii.gz", "")              # amos_0001
        try:
            i = int(pid.split("_")[1])
        except ValueError:
            continue
        if i > MAX_CT_ID:                                       # skip MRI (>=501)
            continue
        out_path = out_root / f"{pid}.npy"
        if not out_path.exists():
            todo.append((pid, ct_path, out_path))

    print(f"[{args.split}] {len(todo)} volumes to process "
          f"(from {len(all_files)} files in {SPLIT_TO_DIR[args.split]}/, "
          f"after skipping MRI IDs > {MAX_CT_ID} and already-done)")

    for pid, ct_path, out_path in tqdm(todo):
        try:
            vol = amos_to_volume(ct_path)
            np.save(out_path, vol)
            print(f"[done] {pid} shape={vol.shape}")
        except Exception as e:
            print(f"[FAIL] {pid}: {e}")


if __name__ == "__main__":
    main()
