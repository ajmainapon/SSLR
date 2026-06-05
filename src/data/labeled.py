import numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset


class LabeledSlices(Dataset):
    """
    Few-shot segmentation dataset, mirrors Phase 1's flat layout.

    Reads:
        {slices_root}/{pid}.npy   uint8 [Z, 224, 224]   image stack 0..255
        {labels_root}/{pid}.npy   uint8 [Z, 224, 224]   class index 0..K

    Returns (img [3,H,W] float32 in [0,1], mask [H,W] int64).
    Normalization matches src/data/dataset.py exactly (vol/255.0).
    """
    def __init__(self, slices_root, labels_root, patients=None, min_fg_frac=0.0):
        self.slices_root = Path(slices_root)
        self.labels_root = Path(labels_root)

        files = sorted(self.slices_root.glob("*.npy"))
        if patients is not None:
            keep = set(patients)
            files = [f for f in files if f.stem in keep]

        self._img_mm  = {}
        self._mask_mm = {}
        self.index = []

        for f in files:
            mask_f = self.labels_root / f.name
            if not mask_f.exists():
                continue
            n = np.load(f, mmap_mode="r").shape[0]
            mn = np.load(mask_f, mmap_mode="r").shape[0]
            if n != mn:
                print(f"[warn] {f.stem}: img Z={n} != mask Z={mn}, skipping")
                continue
            if min_fg_frac > 0:
                m = np.load(mask_f, mmap_mode="r")
                fg = (m > 0).reshape(n, -1).mean(axis=1)
                valid = np.where(fg >= min_fg_frac)[0].tolist()
            else:
                valid = list(range(n))
            for z in valid:
                self.index.append((f, mask_f, int(z)))

    def _img(self, f):
        if f not in self._img_mm:
            self._img_mm[f] = np.load(f, mmap_mode="r")
        return self._img_mm[f]

    def _mask(self, f):
        if f not in self._mask_mm:
            self._mask_mm[f] = np.load(f, mmap_mode="r")
        return self._mask_mm[f]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        f, mask_f, z = self.index[i]
        img  = self._img(f)[z].astype(np.float32) / 255.0
        mask = self._mask(mask_f)[z].astype(np.int64)
        # Defensive: clip to valid class range so a corrupt pixel
        # can't crash CUDA loss kernel asynchronously many epochs later.
        np.clip(mask, 0, 10, out=mask)
        img  = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask
