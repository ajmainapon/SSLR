"""Prepare AMOS22 MR labels (IDs 0501-0600) in the same 7-class subset used
by prepare_labels_amos.py.

CROSS-MODALITY NOTE:
    AMOS22 MR labels use the SAME class scheme as AMOS22 CT labels
    (per the official AMOS22 spec). So the same AMOS_TO_OURS remap applies
    byte-for-byte. Only the intensity processing differs (MR has no HU scale,
    handled in preprocess_amos_mri.py).

Output layout (mirror of data_mri/slices/):
    data_mri/labels/train/{pid}.npy   uint8 (Z, 224, 224)  class-indexed
    data_mri/labels/val/{pid}.npy     uint8 (Z, 224, 224)  class-indexed

Class IDs (must match --num_classes 8 at probe time):
    0 = background
    1 = liver,   2 = spleen,   3 = kidney_L,   4 = kidney_R
    5 = stomach, 6 = pancreas, 7 = aorta

Usage on the RTX box:
    cd ~/SSLP
    python scripts/prepare_labels_amos_mri.py --split train
    python scripts/prepare_labels_amos_mri.py --split val
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------------------------------------------------------------------
TARGET_HW = 224

# AMOS22 official label IDs -> our 7-class output IDs.
# Same mapping as prepare_labels_amos.py (CT) since AMOS22 uses one label
# scheme across modalities. Anything not in dict -> background.
AMOS_TO_OURS = {
    6:  1,   # liver
    1:  2,   # spleen
    3:  3,   # kidney_left
    2:  4,   # kidney_right
    7:  5,   # stomach
    10: 6,   # pancreas
    8:  7,   # aorta
}

SPLIT_TO_IMG_DIR = {"train": "imagesTr", "val": "imagesVa"}
SPLIT_TO_LBL_DIR = {"train": "labelsTr", "val": "labelsVa"}
MIN_MR_ID = 501
MAX_MR_ID = 600

ROOT = Path("amos")
OUT  = Path("data_mri/labels")


def amos_to_label_volume(label_path: Path, ref_HWZ: tuple) -> np.ndarray:
    """Read AMOS MR label NIfTI, remap to our 7-class IDs, resize HW to 224,
    transpose to (Z, 224, 224). Returns uint8 array.
    """
    H, W, Z = ref_HWZ
    raw = nib.load(str(label_path)).get_fdata().astype(np.uint8)
    if raw.shape != (H, W, Z):
        raw = zoom(raw, (H / raw.shape[0], W / raw.shape[1], Z / raw.shape[2]), order=0)
        raw = raw.astype(np.uint8)

    fused = np.zeros_like(raw, dtype=np.uint8)
    for amos_id, our_id in AMOS_TO_OURS.items():
        fused[raw == amos_id] = our_id

    if H != TARGET_HW or W != TARGET_HW:
        fused = zoom(fused, (TARGET_HW / H, TARGET_HW / W, 1.0), order=0)
        fused = fused.astype(np.uint8)

    return np.ascontiguousarray(fused.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out",  default=str(OUT))
    ap.add_argument("--split", required=True, choices=["train", "val"])
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = root / SPLIT_TO_IMG_DIR[args.split]
    labels_dir = root / SPLIT_TO_LBL_DIR[args.split]
    if not labels_dir.exists():
        raise FileNotFoundError(f"AMOS {SPLIT_TO_LBL_DIR[args.split]}/ not found at {labels_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"AMOS {SPLIT_TO_IMG_DIR[args.split]}/ not found at {images_dir}")

    todo = []
    all_files = sorted(labels_dir.glob("amos_*.nii.gz"))
    for lbl_path in all_files:
        pid = lbl_path.name.replace(".nii.gz", "")
        try:
            i = int(pid.split("_")[1])
        except ValueError:
            continue
        if i < MIN_MR_ID or i > MAX_MR_ID:
            continue
        ct_path  = images_dir / f"{pid}.nii.gz"
        out_path = out_root / f"{pid}.npy"
        if ct_path.exists() and not out_path.exists():
            todo.append((pid, ct_path, lbl_path, out_path))

    print(f"[{args.split}] {len(todo)} MR volumes to process "
          f"(MR ID window {MIN_MR_ID}-{MAX_MR_ID})")
    print(f"AMOS -> our 7-class remap: {AMOS_TO_OURS}")

    for pid, ct_path, lbl_path, out_path in tqdm(todo):
        try:
            H, W, Z = nib.load(str(ct_path)).shape
            lbl = amos_to_label_volume(lbl_path, (H, W, Z))
            np.save(out_path, lbl)
            fg = (lbl > 0).mean()
            present = sorted(set(np.unique(lbl).tolist()) - {0})
            print(f"[done] {pid} shape={lbl.shape} fg_frac={fg:.3f} classes_present={present}")
        except Exception as e:
            print(f"[FAIL] {pid}: {e}")


if __name__ == "__main__":
    main()
