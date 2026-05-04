"""Evaluate nnU-Net predictions at SSLR's 224x224 val resolution.

Reads predictions/nnunet_2d_n50/{pid}.nii.gz (full-resolution NIfTI from
nnU-Net's predict step), applies the IDENTICAL HW resize + transpose as
prepare_labels.py / preprocess.py, and computes per-class Dice against
data/labels/val/{pid}.npy.

Produces:
  - stdout summary (per-organ Dice + headline mean foreground Dice)
  - runs/<out_run>/{log.json,args.json} in the same schema as
    train_decoder.py so make_figures.py can pick it up.

This is the "supervised ceiling" number that frames the v2 SSL N=50/N=100
linear-probe results in the paper.
"""
import argparse, json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm


ORGANS = [
    "liver", "spleen", "kidney_L", "kidney_R", "stomach", "pancreas",
    "lung_uL", "lung_uR", "heart", "aorta",
]


def resize_pred_to_labels(pred_3d, target_hw=224):
    """Mirror prepare_labels.py: nib array (H, W, Z) -> zoom HW only ->
    transpose (2, 0, 1) -> (Z, 224, 224) uint8."""
    H, W, Z = pred_3d.shape
    if (H, W) != (target_hw, target_hw):
        pred_3d = zoom(pred_3d, (target_hw / H, target_hw / W, 1.0), order=0)
    return np.ascontiguousarray(pred_3d.transpose(2, 0, 1)).astype(np.uint8)


def per_class_dice_accum(pred, target, sums, counts, C):
    """Per-class Dice across the volume, accumulated into sums/counts.
    Mirrors train_decoder.py's per_class_dice but operates on full volumes."""
    for c in range(C):
        p = (pred == c)
        t = (target == c)
        denom = p.sum() + t.sum()
        if denom > 0:
            sums[c] += 2 * (p & t).sum() / denom
            counts[c] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",   default="predictions/nnunet_2d_n50",
                    help="nnU-Net predict output dir of *.nii.gz files")
    ap.add_argument("--val_labels_dir", default="data/labels/val")
    ap.add_argument("--num_classes",   type=int, default=11)
    ap.add_argument("--out_run",       default="runs/nnunet_2d_n50",
                    help="Output dir for log.json + args.json (train_decoder.py compatible)")
    ap.add_argument("--n_train_volumes", type=int, default=50,
                    help="Just for the args.json record")
    args = ap.parse_args()

    pred_dir = Path(args.predictions)
    lbl_dir  = Path(args.val_labels_dir)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise SystemExit(f"No .nii.gz predictions in {pred_dir}")

    print(f"[eval] {len(pred_files)} predictions, val labels in {lbl_dir}")

    sums   = np.zeros(args.num_classes)
    counts = np.zeros(args.num_classes)
    used, skipped = 0, []

    for pf in tqdm(pred_files, desc="eval"):
        pid = pf.name.replace(".nii.gz", "")
        lbl_p = lbl_dir / f"{pid}.npy"
        if not lbl_p.exists():
            skipped.append((pid, "no_label"))
            continue
        pred_3d = nib.load(str(pf)).get_fdata().astype(np.uint8)
        pred = resize_pred_to_labels(pred_3d)
        target = np.load(lbl_p)
        if pred.shape != target.shape:
            print(f"[warn] {pid}: pred {pred.shape} != label {target.shape} -- skipping")
            skipped.append((pid, f"shape_mismatch_{pred.shape}_vs_{target.shape}"))
            continue
        per_class_dice_accum(pred, target, sums, counts, args.num_classes)
        used += 1

    per_class = (sums / counts.clip(min=1)).tolist()
    fg = float(np.mean(per_class[1:]))

    # Stdout summary
    print(f"\n[eval] used {used}/{len(pred_files)} predictions, skipped {len(skipped)}")
    if skipped:
        print(f"[eval] skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    print(f"\n=== nnU-Net 2D supervised at N={args.n_train_volumes} ===")
    print(f"Mean foreground Dice: {fg:.4f}")
    print(f"\nPer-organ:")
    for i, organ in enumerate(ORGANS):
        print(f"  {organ:12s}  {per_class[i+1]:.4f}")

    # Save in train_decoder.py's log.json schema so make_figures.py picks it up
    out = Path(args.out_run); out.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "ep":     0,
        "loss":   0.0,
        "train_fg": fg,
        "val_fg":   fg,
        "train_per_class": per_class,
        "val_per_class":   per_class,
    }
    (out / "log.json").write_text(json.dumps([log_entry], indent=2))
    (out / "args.json").write_text(json.dumps({
        "head":            "nnunet_2d",
        "ckpt":            "supervised",  # truthy so make_figures labels it "SSL"-equivalent
        "n_train_volumes": args.n_train_volumes,
        "num_classes":     args.num_classes,
        "predictions":     str(pred_dir),
    }, indent=2))
    print(f"\n[wrote] {out}/log.json")
    print(f"[wrote] {out}/args.json")
    print(f"\n[next] regenerate figures with the supervised baseline:")
    print(f"  python scripts/make_figures.py --runs_root runs --out figures_v2 --era v2")
    print(f"  python scripts/per_organ_table.py \\")
    print(f"      --ssl_run runs/lin_v2_ssl_n{args.n_train_volumes} \\")
    print(f"      --random_run {out} \\")
    print(f"      --n {args.n_train_volumes} --out figures_v2/per_organ_vs_nnunet")


if __name__ == "__main__":
    main()
