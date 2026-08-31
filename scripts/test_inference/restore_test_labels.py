#!/usr/bin/env python3
"""Restore the 89 fused test masks from the official v2.01 dataset archive."""
import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


EXPECTED_MD5 = "fe250e5718e0a3b5df4c4ea9d58a62fe"
ORGANS = [
    "liver", "spleen", "kidney_left", "kidney_right", "stomach",
    "pancreas", "lung_upper_lobe_left", "lung_upper_lobe_right", "heart",
    "aorta",
]


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    image_dir = root / "data/slices/test"
    output_dir = root / "data/labels/test"
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = sorted(path.stem for path in image_dir.glob("*.npy"))
    if len(patient_ids) != 89:
        raise RuntimeError(f"Expected 89 test images, found {len(patient_ids)}")

    if not args.skip_md5:
        actual_md5 = file_md5(args.archive)
        if actual_md5 != EXPECTED_MD5:
            raise RuntimeError(f"Archive MD5 mismatch: {actual_md5} != {EXPECTED_MD5}")

    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        by_suffix = {name.lstrip("/"): name for name in names}

        def locate(patient_id, organ):
            suffix = f"/{patient_id}/segmentations/{organ}.nii.gz"
            matches = [name for name in names if name.endswith(suffix) or name == suffix[1:]]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one archive member ending {suffix}, found {len(matches)}"
                )
            return matches[0]

        with tempfile.TemporaryDirectory(prefix="totalseg_test_masks_") as temp_dir:
            temp_root = Path(temp_dir)
            for index, patient_id in enumerate(patient_ids, 1):
                output_path = output_dir / f"{patient_id}.npy"
                if output_path.exists():
                    print(f"[{index}/89] skip {patient_id}")
                    continue

                fused = None
                expected_shape = None
                for class_id, organ in enumerate(ORGANS, 1):
                    member = locate(patient_id, organ)
                    temporary_mask = temp_root / f"{patient_id}_{organ}.nii.gz"
                    with archive.open(member) as source, temporary_mask.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    mask = nib.load(temporary_mask).get_fdata().astype(np.uint8)
                    temporary_mask.unlink()
                    if fused is None:
                        expected_shape = mask.shape
                        fused = np.zeros(expected_shape, dtype=np.uint8)
                    if mask.shape != expected_shape:
                        mask = zoom(
                            mask,
                            tuple(a / b for a, b in zip(expected_shape, mask.shape)),
                            order=0,
                        )
                    fused[mask > 0] = class_id

                height, width, depth = fused.shape
                fused = zoom(fused, (224 / height, 224 / width, 1.0), order=0)
                fused = np.ascontiguousarray(fused.transpose(2, 0, 1))
                image_depth = np.load(image_dir / f"{patient_id}.npy", mmap_mode="r").shape[0]
                if fused.shape != (image_depth, 224, 224):
                    raise RuntimeError(
                        f"{patient_id}: restored mask {fused.shape} does not match "
                        f"image {(image_depth, 224, 224)}"
                    )
                np.save(output_path, fused)
                print(f"[{index}/89] wrote {output_path.name} shape={fused.shape}")


if __name__ == "__main__":
    main()
