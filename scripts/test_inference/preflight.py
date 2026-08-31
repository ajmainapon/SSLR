#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
images = {p.stem for p in (root / "data/slices/test").glob("*.npy")}
labels = {p.stem for p in (root / "data/labels/test").glob("*.npy")}
manifest = json.loads((root / "manifest.json").read_text())
entries = manifest["checkpoints"]
missing_files = []
for entry in entries:
    for key in ("checkpoint", "backbone"):
        if entry.get(key) and not (root / entry[key]).exists():
            missing_files.append(entry[key])

print(f"test images:      {len(images)}")
print(f"test labels:      {len(labels)}")
print(f"matched IDs:      {len(images & labels)}")
print(f"manifest models:  {len(entries)}")
print(f"missing artifacts:{len(missing_files)}")
if len(images) != 89 or len(labels) != 89 or images != labels or missing_files:
    print("PREFLIGHT FAILED: restore all 89 matched test labels/artifacts before inference")
    sys.exit(1)
print("PREFLIGHT PASSED")
