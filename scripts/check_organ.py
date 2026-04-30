import argparse, numpy as np, nibabel as nib, matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

ap = argparse.ArgumentParser()
ap.add_argument("--ts_root", default="totalsegmentator")
ap.add_argument("--slices_root", default="data/slices/train")
ap.add_argument("--organ", default="liver")
ap.add_argument("--out", default="organ_check")
ap.add_argument("--patients", nargs="+", default=["s1106","s0655","s0839","s1002"])
args = ap.parse_args()

Path(args.out).mkdir(exist_ok=True)
for pid in args.patients:
    img_path = Path(args.slices_root) / f"{pid}.npy"
    nii_path = Path(args.ts_root) / pid / "segmentations" / f"{args.organ}.nii.gz"
    if not img_path.exists() or not nii_path.exists():
        print(f"[skip] {pid}: img={img_path.exists()} nii={nii_path.exists()}"); continue
    img = np.load(img_path, mmap_mode="r")             # (Z, 224, 224) uint8
    m   = nib.load(str(nii_path)).get_fdata().astype(np.uint8)  # (H, W, Z) raw
    H, W, Z0 = m.shape
    m = zoom(m, (224/H, 224/W, 1.0), order=0)          # (224, 224, Z0)
    m = np.ascontiguousarray(m.transpose(2, 0, 1))     # (Z0, 224, 224)
    if m.shape[0] != img.shape[0]:
        print(f"[warn] {pid}: Z mismatch img={img.shape[0]} mask={m.shape[0]}")
        continue
    fg = m.reshape(m.shape[0], -1).sum(1)
    if fg.max() == 0:
        print(f"[skip] {pid}: organ empty"); continue
    z = int(np.argmax(fg))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img[z], cmap="gray"); ax[0].set_title(f"{pid} z={z}")
    ax[1].imshow(m[z], cmap="Reds"); ax[1].set_title(f"{args.organ} mask")
    ax[2].imshow(img[z], cmap="gray"); ax[2].imshow(m[z], cmap="Reds", alpha=0.45); ax[2].set_title("overlay")
    for a in ax: a.axis("off")
    out = Path(args.out)/f"{pid}_{args.organ}_z{z:03d}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"[ok] {out}  argmax_z={z}/{m.shape[0]}  fg={fg[z]}")
