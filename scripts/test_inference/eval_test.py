#!/usr/bin/env python3
"""Locked-checkpoint TotalSegmentator test evaluation.

Evaluates validation-selected downstream checkpoints only. It never trains,
selects checkpoints, or modifies model weights.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import timm

from src.data.labeled import LabeledSlices
from src.models.seg_head import LinearSegHead, ConvSegHead


ORGAN_NAMES = [
    "background", "liver", "spleen", "kidney_left", "kidney_right",
    "stomach", "pancreas", "lung_upper_lobe_left",
    "lung_upper_lobe_right", "heart", "aorta",
]


def build_vit():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)


def load_context_encoder(vit, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = {
            key.replace("context_enc.", ""): value
            for key, value in state["model"].items()
            if key.startswith("context_enc.")
        }
    missing, unexpected = vit.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected backbone keys: {unexpected[:10]}")
    print(f"[load] backbone missing={len(missing)} unexpected={len(unexpected)}")
    return vit


def extract_tokens(vit, images):
    features = vit.forward_features(images)
    n_tokens = features.shape[1]
    side = int(round(n_tokens ** 0.5))
    if side * side != n_tokens:
        features = features[:, 1:, :]
    return features


def per_class_dice(prediction, target, num_classes):
    sums = torch.zeros(num_classes)
    counts = torch.zeros(num_classes)
    for class_id in range(num_classes):
        pred_class = prediction == class_id
        target_class = target == class_id
        denominator = pred_class.sum().float() + target_class.sum().float()
        if denominator > 0:
            sums[class_id] += 2 * (pred_class & target_class).sum().float() / denominator
            counts[class_id] += 1
    return sums, counts


@torch.inference_mode()
def evaluate(vit, head, loader, num_classes, device):
    vit.eval()
    head.eval()
    sums = torch.zeros(num_classes)
    counts = torch.zeros(num_classes)
    use_amp = device.type == "cuda"
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = head(extract_tokens(vit, images))
        prediction = logits.argmax(dim=1).cpu()
        batch_sums, batch_counts = per_class_dice(prediction, masks, num_classes)
        sums += batch_sums
        counts += batch_counts
    return sums / counts.clamp(min=1)


def validate_test_data(images_dir, labels_dir, expected_volumes):
    image_ids = {path.stem for path in images_dir.glob("*.npy")}
    label_ids = {path.stem for path in labels_dir.glob("*.npy")}
    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)
    if len(image_ids) != expected_volumes or len(label_ids) != expected_volumes:
        raise RuntimeError(
            f"Expected {expected_volumes} test image/label pairs, found "
            f"{len(image_ids)} images and {len(label_ids)} labels. "
            f"Missing labels={missing_labels[:10]}, missing images={missing_images[:10]}"
        )
    if image_ids != label_ids:
        raise RuntimeError(
            f"Test IDs do not match. Missing labels={missing_labels[:10]}, "
            f"missing images={missing_images[:10]}"
        )


def make_models(entry, package_root, device):
    run_dir = package_root / entry["run_dir"]
    run_args = json.loads((run_dir / "args.json").read_text())
    seed = int(run_args.get("seed", entry.get("seed", 0)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_path = package_root / entry["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head_name = run_args.get("head", entry["head"])
    head_class = {"linear": LinearSegHead, "conv": ConvSegHead}[head_name]
    head = head_class(dim=768, num_classes=11, patch_size=16, img_size=224)
    vit = build_vit()

    if entry["mode"] == "finetune":
        vit.load_state_dict(checkpoint["vit"], strict=True)
        head.load_state_dict(checkpoint["head"], strict=True)
    elif entry["mode"] == "frozen":
        if entry.get("backbone"):
            vit = load_context_encoder(vit, package_root / entry["backbone"])
        head.load_state_dict(checkpoint["head"], strict=True)
    else:
        raise ValueError(f"Unknown mode: {entry['mode']}")

    return vit.to(device), head.to(device), run_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--images", default="data/slices/test")
    parser.add_argument("--labels", default="data/labels/test")
    parser.add_argument("--output", default="results/test_results.json")
    parser.add_argument("--only", nargs="*", help="Optional exact manifest names")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-volumes", type=int, default=89)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    images_dir = package_root / args.images
    labels_dir = package_root / args.labels
    validate_test_data(images_dir, labels_dir, args.expected_volumes)

    manifest = json.loads((package_root / args.manifest).read_text())
    entries = manifest["checkpoints"]
    if args.only:
        requested = set(args.only)
        entries = [entry for entry in entries if entry["name"] in requested]
        found = {entry["name"] for entry in entries}
        if found != requested:
            raise RuntimeError(f"Unknown --only names: {sorted(requested - found)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU not detected; test inference is intended for the RTX 5090")
    print(f"[device] {torch.cuda.get_device_name(0)}")

    output_path = package_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    if output_path.exists():
        completed = json.loads(output_path.read_text()).get("results", [])
    completed_names = {row["name"] for row in completed}

    for index, entry in enumerate(entries, 1):
        if entry["name"] in completed_names:
            print(f"[skip] {entry['name']} already present")
            continue
        print(f"[{index}/{len(entries)}] {entry['name']}", flush=True)
        started = time.time()
        vit, head, run_args = make_models(entry, package_root, device)
        # Preserve the original evaluation batch size. The historical Dice
        # implementation averages batch-level scores, so changing batch size
        # would change the reported metric.
        batch_size = int(run_args.get("bs", 8 if entry["mode"] == "finetune" else 16))
        dataset = LabeledSlices(images_dir, labels_dir, min_fg_frac=0.0)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        dice = evaluate(vit, head, loader, 11, device)
        row = {
            **entry,
            "test_fg": float(dice[1:].mean()),
            "test_per_class": {
                name: float(value) for name, value in zip(ORGAN_NAMES, dice)
            },
            "test_slices": len(dataset),
            "batch_size": batch_size,
            "elapsed_seconds": time.time() - started,
        }
        completed.append(row)
        output_path.write_text(json.dumps({"results": completed}, indent=2))
        print(f"[result] test_fg={row['test_fg']:.4f}", flush=True)
        del vit, head
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
