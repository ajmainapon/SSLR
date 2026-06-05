import nibabel as nib, numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm

ROOT = Path("totalsegmentator")
OUT  = Path("data/slices")
META = ROOT / "meta.csv"
HU_MIN, HU_MAX = -1000, 400
TARGET_HW = 224

def window_to_u8(x):
    x = np.clip(x, HU_MIN, HU_MAX)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)
    return (x * 255.0).astype(np.uint8)

def resize_vol(vol, size=TARGET_HW):
    h, w, z = vol.shape
    return zoom(vol, (size/h, size/w, 1.0), order=1)

def main():
    meta = pd.read_csv(META, sep=";")
    meta.columns = [c.strip().lstrip("\ufeff") for c in meta.columns]
    OUT.mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        pid, split = row["image_id"], row["split"]
        ct_path = ROOT / pid / "ct.nii.gz"
        if not ct_path.exists(): continue
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pid}.npy"
        if out_path.exists(): continue
        vol = nib.load(ct_path).get_fdata().astype(np.float32)
        vol = resize_vol(vol)
        vol = window_to_u8(vol)
        np.save(out_path, np.ascontiguousarray(vol.transpose(2, 0, 1)))

if __name__ == "__main__":
    main()
