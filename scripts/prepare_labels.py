import argparse, numpy as np, nibabel as nib, pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm

# MUST match src/data/preprocess.py
ROOT = Path("totalsegmentator")
OUT  = Path("data/labels")
META = ROOT / "meta.csv"
TARGET_HW = 224

# class 0 = background; classes 1..K below
ORGANS = [
    "liver",
    "spleen",
    "kidney_left",
    "kidney_right",
    "stomach",
    "pancreas",
    "lung_upper_lobe_left",
    "lung_upper_lobe_right",
    "heart",
    "aorta",
]

def main(split):
    meta = pd.read_csv(META, sep=";")
    meta.columns = [c.strip().lstrip("\ufeff") for c in meta.columns]
    out_root = OUT / split
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [r for _, r in meta.iterrows() if r["split"] == split]
    for row in tqdm(rows):
        pid = row["image_id"]
        seg_dir = ROOT / pid / "segmentations"
        ct_path = ROOT / pid / "ct.nii.gz"
        if not seg_dir.exists() or not ct_path.exists():
            continue
        out_path = out_root / f"{pid}.npy"
        if out_path.exists():
            continue
        # Reference shape from CT — same load as preprocess.py (NO canonicalize)
        H, W, Z = nib.load(ct_path).shape
        fused = np.zeros((H, W, Z), dtype=np.uint8)
        for k, organ in enumerate(ORGANS, start=1):
            p = seg_dir / f"{organ}.nii.gz"
            if not p.exists():
                continue
            m = nib.load(p).get_fdata().astype(np.uint8)
            if m.shape != (H, W, Z):
                m = zoom(m, (H/m.shape[0], W/m.shape[1], Z/m.shape[2]), order=0)
            fused[m > 0] = k
        # Identical pipeline to preprocess.py: resize HW to 224, keep Z, transpose to (Z, H, W)
        fused = zoom(fused, (TARGET_HW/H, TARGET_HW/W, 1.0), order=0)
        fused = np.ascontiguousarray(fused.transpose(2, 0, 1))
        np.save(out_path, fused)
        print(f"[done] {pid} shape={fused.shape} fg_frac={(fused>0).mean():.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    main(ap.parse_args().split)
