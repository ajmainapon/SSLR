"""Convert the SSLP TotalSegmentator layout into nnU-Net v2 raw format.

Builds a Dataset001_Organ/ directory with the SAME 50 training patients used
for the v2 SSL N=50 linear probe (read from runs/lin_v2_ssl_n50/train_volumes.json)
and the same val patient set used by train_decoder.py (read from data/labels/val/).

  imagesTr/{pid}_0000.nii.gz   CT volume (raw, original resolution)
  labelsTr/{pid}.nii.gz        Multi-class organ mask (10 organs + bg)
  imagesTs/{pid}_0000.nii.gz   Val CT volumes (for nnU-Net inference)
  labelsTs/{pid}.nii.gz        Val masks (for our own eval, not used by nnU-Net)
  dataset.json                 nnU-Net v2 schema

Source data layout expected:
  totalsegmentator/{pid}/ct.nii.gz
  totalsegmentator/{pid}/segmentations/{organ}.nii.gz   (one file per organ)

Multi-class label fusion is identical to scripts/prepare_labels.py: organ k gets
class index k (1..10), background = 0. No HU windowing or resampling here --
nnU-Net does its own preprocessing.

Usage:
  python scripts/make_nnunet_dataset.py \\
      --train_volumes_json runs/lin_v2_ssl_n50/train_volumes.json \\
      --val_labels_dir data/labels/val \\
      --ts_root totalsegmentator \\
      --out nnUNet_raw/Dataset001_Organ
"""
import argparse, json, os, shutil
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm


ORGAN_NIFTI_NAMES = [
    "liver", "spleen", "kidney_left", "kidney_right",
    "stomach", "pancreas", "lung_upper_lobe_left",
    "lung_upper_lobe_right", "heart", "aorta",
]

LABEL_MAP = {"background": 0}
for k, name in enumerate(ORGAN_NIFTI_NAMES, start=1):
    LABEL_MAP[name] = k


def fuse_organs(seg_dir, ref_shape):
    """Fuse per-organ NIfTI masks into a single multi-class label volume.
    Returns array of shape ref_shape with values in {0..len(ORGAN_NIFTI_NAMES)}."""
    fused = np.zeros(ref_shape, dtype=np.uint8)
    for k, organ in enumerate(ORGAN_NIFTI_NAMES, start=1):
        p = seg_dir / f"{organ}.nii.gz"
        if not p.exists():
            continue
        m = nib.load(str(p)).get_fdata().astype(np.uint8)
        if m.shape != ref_shape:
            print(f"[warn] {organ}: shape mismatch {m.shape} vs ct {ref_shape} -- skipping")
            continue
        fused[m > 0] = k
    return fused


def write_pair(pid, ts_root, img_out_dir, lbl_out_dir, link_mode="symlink"):
    """Place CT in img_out_dir and write fused label NIfTI to lbl_out_dir.

    link_mode:
      "symlink" (default) -- link to the source CT, near-zero disk cost.
      "copy"              -- duplicate the CT (safer for archival use).
    Returns True if successful."""
    ct_p = (ts_root / pid / "ct.nii.gz").resolve()
    seg_dir = ts_root / pid / "segmentations"
    if not (ct_p.exists() and seg_dir.exists()):
        print(f"[skip] {pid}: missing ct or segmentations dir")
        return False
    ct_nib = nib.load(str(ct_p))
    fused = fuse_organs(seg_dir, ct_nib.shape)
    if fused.sum() == 0:
        print(f"[skip] {pid}: empty fused label")
        return False
    # Place CT (symlink by default; saves ~50-100 MB per patient)
    img_dst = img_out_dir / f"{pid}_0000.nii.gz"
    if img_dst.exists() or img_dst.is_symlink():
        img_dst.unlink()
    if link_mode == "symlink":
        os.symlink(ct_p, img_dst)
    else:
        shutil.copy(ct_p, img_dst)
    # Write fused label with the CT's affine + header
    lbl_img = nib.Nifti1Image(fused, ct_nib.affine, ct_nib.header)
    lbl_img.set_data_dtype(np.uint8)
    nib.save(lbl_img, str(lbl_out_dir / f"{pid}.nii.gz"))
    return True


def derive_val_pids(val_labels_dir):
    """Get val patient IDs from data/labels/val/{pid}.npy filenames."""
    return sorted(p.stem for p in Path(val_labels_dir).glob("*.npy")
                  if p.stem != "_audit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_volumes_json", required=True,
                    help="Path to runs/lin_v2_ssl_n50/train_volumes.json")
    ap.add_argument("--val_labels_dir",     default="data/labels/val")
    ap.add_argument("--ts_root",            default="totalsegmentator")
    ap.add_argument("--out",                required=True,
                    help="e.g. nnUNet_raw/Dataset001_Organ")
    ap.add_argument("--max_val", type=int, default=None,
                    help="Optional cap on val patients (useful for quick smoke test)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy source CTs into nnUNet_raw instead of symlinking. "
                         "Default is symlink (saves ~15-25 GB).")
    args = ap.parse_args()
    link_mode = "copy" if args.copy else "symlink"

    train_pids = json.loads(Path(args.train_volumes_json).read_text())
    val_pids = derive_val_pids(args.val_labels_dir)
    if args.max_val:
        val_pids = val_pids[:args.max_val]

    print(f"[plan] {len(train_pids)} train + {len(val_pids)} val patients")
    print(f"[plan] train: {train_pids[:5]}{'...' if len(train_pids) > 5 else ''}")
    print(f"[plan] val:   {val_pids[:5]}{'...' if len(val_pids) > 5 else ''}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    img_tr, lbl_tr = out / "imagesTr", out / "labelsTr"
    img_ts, lbl_ts = out / "imagesTs", out / "labelsTs"
    for d in (img_tr, lbl_tr, img_ts, lbl_ts):
        d.mkdir(exist_ok=True)

    ts_root = Path(args.ts_root)
    n_tr_ok = 0
    for pid in tqdm(train_pids, desc="train"):
        if write_pair(pid, ts_root, img_tr, lbl_tr, link_mode=link_mode):
            n_tr_ok += 1
    n_ts_ok = 0
    for pid in tqdm(val_pids, desc="val"):
        if write_pair(pid, ts_root, img_ts, lbl_ts, link_mode=link_mode):
            n_ts_ok += 1

    # Write nnU-Net v2 dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels":        LABEL_MAP,
        "numTraining":   n_tr_ok,
        "file_ending":   ".nii.gz",
        "description":   "TotalSegmentator subset matched to v2 SSL N=50 linear-probe split",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"\n[done] wrote {n_tr_ok}/{len(train_pids)} train + {n_ts_ok}/{len(val_pids)} val patients to {out}")
    print(f"[done] dataset.json: {len(LABEL_MAP)-1} foreground classes")
    print(f"\nNext: set env vars and run nnU-Net:")
    print(f"  export nnUNet_raw='{Path.cwd() / Path(args.out).parent}'")
    print(f"  export nnUNet_preprocessed='{Path.cwd() / 'nnUNet_preprocessed'}'")
    print(f"  export nnUNet_results='{Path.cwd() / 'nnUNet_results'}'")
    print(f"  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
    print(f"  nnUNetv2_train 1 2d all   # train 2D on all 50 patients (no CV)")


if __name__ == "__main__":
    main()
