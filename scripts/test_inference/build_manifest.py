#!/usr/bin/env python3
"""Build the primary TotalSegmentator checkpoint manifest."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
entries = []


def add(name, run_dir, checkpoint, mode, head, initialization, n, seed, backbone=None):
    required = [ROOT / run_dir / "args.json", ROOT / checkpoint]
    if backbone:
        required.append(ROOT / backbone)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print(f"[unavailable] {name}: {', '.join(missing)}")
        return
    entry = {
        "name": name, "run_dir": run_dir, "checkpoint": checkpoint,
        "mode": mode, "head": head, "initialization": initialization,
        "n": n, "seed": seed,
    }
    if backbone:
        entry["backbone"] = backbone
    entries.append(entry)


for n in (5, 10, 20, 50, 100):
    for seed in range(4):
        suffix = "" if seed == 0 else f"_seed{seed}"
        ssl_run = f"runs/lin_loo_a3_n{n}{suffix}"
        add(
            f"linear_a2_ssl_n{n}_seed{seed}", ssl_run,
            f"{ssl_run}/head_best.pt", "frozen", "linear", "ssl", n, seed,
            "backbones/loo_a3_narrow_k_vit_ep039.pt",
        )
        random_run = f"runs/lin_v2_random_n{n}{suffix}"
        add(
            f"linear_random_n{n}_seed{seed}", random_run,
            f"{random_run}/head_best.pt", "frozen", "linear", "random", n, seed,
        )

for n in (20, 50, 100):
    for seed in range(4):
        ssl_run = f"runs/ft_a3_ssl_n{n}_bblr1e-4_seed{seed}"
        add(
            f"finetune_a2_ssl_n{n}_seed{seed}", ssl_run,
            f"{ssl_run}/ft_best.pt", "finetune", "conv", "ssl", n, seed,
        )
        random_run = (
            f"runs/ft_v2_random_n{n}_bblr1e-4" if seed == 0
            else f"runs/ft_v2_random_n{n}_bblr1e-4_seed{seed}"
        )
        add(
            f"finetune_random_n{n}_seed{seed}", random_run,
            f"{random_run}/ft_best.pt", "finetune", "conv", "random", n, seed,
        )

(ROOT / "manifest.json").write_text(json.dumps({"checkpoints": entries}, indent=2))
print(f"[done] wrote {len(entries)} available checkpoints to manifest.json")
