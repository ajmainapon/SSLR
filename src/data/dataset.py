import numpy as np, torch, random
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset


class SliceTriplet(Dataset):
    """Yields (S_{z-k}, S_z) slice pairs from per-volume uint8 stacks.

    k_range widened from (3, 7) to (8, 20): at ~3 mm slice spacing, the v1
    context slice was 9-21 mm from target -- close enough that the predictor
    could solve the JEPA loss by copying local texture. Wider k forces it
    to reason about anatomical continuity instead.

    augment=True (default): paired hflip + paired random crop-resize
    + per-slice intensity jitter. Without these, JEPA-family models learn
    position-conditioned and photometric shortcuts.
    """

    def __init__(self, root="data/slices/train", k_range=(8, 20),
                 augment=True, hw=224,
                 crop_scale=(0.85, 1.0), intensity_jitter=0.05):
        self.root = Path(root)
        self.k_range = k_range
        self.augment = augment
        self.hw = hw
        self.crop_scale = crop_scale
        self.intensity_jitter = intensity_jitter
        self.files = sorted(self.root.glob("*.npy"))
        self._mm = {}
        self.index = []
        for f in self.files:
            n = np.load(f, mmap_mode="r").shape[0]     # (Z,224,224)
            for i in range(n):
                self.index.append((f, i, n))

    def _mmap(self, f):
        if f not in self._mm:
            self._mm[f] = np.load(f, mmap_mode="r")
        return self._mm[f]

    def __len__(self): return len(self.index)

    def _pick_zc(self, z, n):
        k = random.randint(*self.k_range)
        valid = [zc for zc in (z - k, z + k) if 0 <= zc < n]
        if valid:
            return random.choice(valid)
        # Volume thinner than k_range: shrink k to whatever fits, never return z.
        k = max(z, n - 1 - z, 1)
        return z - k if z - k >= 0 else z + k

    def _crop_resize_pair(self, ctx, tgt):
        H, W = ctx.shape
        s = random.uniform(*self.crop_scale)
        ch = int(round(H * s)); cw = int(round(W * s))
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        ctx = ctx[top:top + ch, left:left + cw]
        tgt = tgt[top:top + ch, left:left + cw]
        if (ch, cw) != (self.hw, self.hw):
            stack = torch.from_numpy(np.stack([ctx, tgt]).copy()) \
                         .float().unsqueeze(1)
            stack = F.interpolate(stack, size=(self.hw, self.hw),
                                  mode="bilinear", align_corners=False)
            ctx, tgt = stack[0, 0].numpy(), stack[1, 0].numpy()
        return ctx, tgt

    def __getitem__(self, idx):
        f, z, n = self.index[idx]
        zc = self._pick_zc(z, n)
        vol = self._mmap(f)
        ctx = vol[zc].astype(np.float32) / 255.0
        tgt = vol[z ].astype(np.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                ctx = ctx[:, ::-1].copy()
                tgt = tgt[:, ::-1].copy()
            ctx, tgt = self._crop_resize_pair(ctx, tgt)
            j = self.intensity_jitter
            ctx = np.clip(ctx + random.uniform(-j, j), 0.0, 1.0)
            tgt = np.clip(tgt + random.uniform(-j, j), 0.0, 1.0)

        ctx = torch.from_numpy(ctx).unsqueeze(0).repeat(3, 1, 1)
        tgt = torch.from_numpy(tgt).unsqueeze(0).repeat(3, 1, 1)
        return ctx, tgt
