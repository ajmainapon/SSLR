"""Verify image/mask alignment by overlay for a few high-fg patients."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("alignment_check"); out_dir.mkdir(exist_ok=True)

# Find all patients with both image and label, in either split
all_pairs = []
for split in ("train", "val"):
    slices_dir = Path("data/slices") / split
    labels_dir = Path("data/labels") / split
    if not (slices_dir.exists() and labels_dir.exists()):
        continue
    for img_f in sorted(slices_dir.glob("*.npy")):
        msk_f = labels_dir / img_f.name
        if msk_f.exists():
            all_pairs.append((img_f, msk_f, split))

print(f"Found {len(all_pairs)} (image,mask) pairs total")
if not all_pairs:
    print("Nothing to check. Did prepare_labels.py run?"); exit(1)

# Score each by total foreground voxels in the volume
scored = []
for img_f, msk_f, split in all_pairs[:200]:        # cap at 200 to keep it fast
    msk = np.load(msk_f, mmap_mode="r")
    total_fg = int((msk > 0).sum())
    scored.append((total_fg, img_f, msk_f, split))
scored.sort(key=lambda x: -x[0])

# Take the top 5 most-foreground patients
for total_fg, img_f, msk_f, split in scored[:5]:
    pid = img_f.stem
    img = np.load(img_f, mmap_mode="r")
    msk = np.load(msk_f, mmap_mode="r")
    fg_per_z = (msk > 0).reshape(msk.shape[0], -1).sum(axis=1)
    z = int(fg_per_z.argmax())

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img[z], cmap="gray"); ax[0].set_title(f"{pid}/{split} z={z}  CT")
    ax[1].imshow(img[z], cmap="gray")
    ax[1].imshow(np.ma.masked_where(msk[z] == 0, msk[z]),
                 cmap="tab10", alpha=0.5, vmin=0, vmax=10)
    ax[1].set_title(f"{pid}/{split} z={z}  mask overlay")
    for a in ax: a.axis("off")
    plt.tight_layout()
    out = out_dir / f"{pid}_{split}_z{z}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight"); plt.close()
    print(f"[saved] {out.name}  total_fg_vox={total_fg}")
