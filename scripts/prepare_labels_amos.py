"""Prepare AMOS22 Task-1 labels in the 7-class subset that overlaps with
our TotalSegmentator 10-organ scheme.

CROSS-DATASET CAVEAT (important, document in paper):
    AMOS22 is abdominal-only. Of our 10 evaluated organs on TotalSegmentator,
    only 7 are present in AMOS:
        liver, spleen, kidney_L, kidney_R, stomach, pancreas, aorta.
    Missing in AMOS:  upper-lobe lung L/R, heart  (thoracic; not in AMOS FoV).
    The AMOS cross-dataset experiment is therefore a 7-class probe (background
    + 7 organs) and is reported alongside (not replacing) the 10-class
    TotalSeg headline.

Output layout (mirror of data/labels/):
    data_amos/labels/train/{pid}.npy   uint8 (Z, 224, 224)  class-indexed
    data_amos/labels/val/{pid}.npy     uint8 (Z, 224, 224)  class-indexed

Class IDs in the output (must match --num_classes 8 at probe time):
    0 = background
    1 = liver
    2 = spleen
    3 = kidney_L
    4 = kidney_R
    5 = stomach
    6 = pancreas
    7 = aorta

Usage on the RTX box:
    cd ~/SSLP
    python SSLR/scripts/prepare_labels_amos.py --split train
    python SSLR/scripts/prepare_labels_amos.py --split val
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants -- MUST match preprocess_amos.py for spatial alignment
# ---------------------------------------------------------------------------
TARGET_HW = 224

# AMOS22 Task-1 official label IDs -> our 7-class output IDs.
# (AMOS IDs not in this dict are dropped to background.)
#
# AMOS22 spec: https://amos22.grand-challenge.org/data/
#   1=spleen, 2=right kidney, 3=left kidney, 4=gallbladder, 5=esophagus,
#   6=liver, 7=stomach, 8=aorta, 9=postcava, 10=pancreas,
#   11=right adrenal, 12=left adrenal, 13=duodenum, 14=bladder, 15=prostate/uterus
AMOS_TO_OURS = {
    6:  1,   # liver
    1:  2,   # spleen
    3:  3,   # kidney_left
    2:  4,   # kidney_right
    7:  5,   # stomach
    10: 6,   # pancreas
    8:  7,   # aorta
}

# Mirror preprocess_amos.py: use AMOS official split, glob actual files,
# skip MRI IDs (>= 501).
SPLIT_TO_IMG_DIR = {"train": "imagesTr", "val": "imagesVa"}
SPLIT_TO_LBL_DIR = {"train": "labelsTr", "val": "labelsVa"}
MAX_CT_ID = 500

ROOT = Path("amos")                 # input
OUT  = Path("data_amos/labels")     # output


def amos_to_label_volume(label_path: Path, ref_HWZ: tuple) -> np.ndarray:
    """Read AMOS label NIfTI, remap to our 7-class IDs, resize HW to 224,
    transpose to (Z, 224, 224). Returns uint8 array.

    The reference shape from the CT volume is used to ensure label and slice
    arrays align byte-for-byte (the prepare_labels.py pattern).
    """
    H, W, Z = ref_HWZ
    raw = nib.load(str(label_path)).get_fdata().astype(np.uint8)
    if raw.shape != (H, W, Z):
        raw = zoom(raw, (H / raw.shape[0], W / raw.shape[1], Z / raw.shape[2]), order=0)
        raw = raw.astype(np.uint8)

    # Remap AMOS IDs -> our class IDs (everything not in the dict -> 0)
    fused = np.zeros_like(raw, dtype=np.uint8)
    for amos_id, our_id in AMOS_TO_OURS.items():
        fused[raw == amos_id] = our_id

    # Resize HW to 224, preserve Z (order=0 for class labels — same as prepare_labels.py)
    if H != TARGET_HW or W != TARGET_HW:
        fused = zoom(fused, (TARGET_HW / H, TARGET_HW / W, 1.0), order=0)
        fused = fused.astype(np.uint8)

    return np.ascontiguousarray(fused.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="AMOS22 root (contains imagesTr/ and labelsTr/)")
    ap.add_argument("--out",  default=str(OUT),  help="Output labels root")
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
        pid = lbl_path.name.replace(".nii.gz", "")              # amos_0001
        try:
            i = int(pid.split("_")[1])
        except ValueError:
            continue
        if i > MAX_CT_ID:                                       # skip MRI (>=501)
            continue
        ct_path  = images_dir / f"{pid}.nii.gz"
        out_path = out_root / f"{pid}.npy"
        if ct_path.exists() and not out_path.exists():
            todo.append((pid, ct_path, lbl_path, out_path))

    print(f"[{args.split}] {len(todo)} volumes to process "
          f"(from {len(all_files)} files in {SPLIT_TO_LBL_DIR[args.split]}/, "
          f"after skipping MRI IDs > {MAX_CT_ID} and already-done)")
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
