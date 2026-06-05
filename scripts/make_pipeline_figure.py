"""make_pipeline_figure.py
================================================================
Generate the teaser-figure PNGs used by paper_draft_overleaf.tex.

The script renders five panels that together tell the end-to-end story:

  1. ct_volume.png        — axial montage hinting at the 3D volume
  2. ct_context.png       — single context slice  S_{z-k}
  3. ct_context_masked.png— same slice with 75% random patch mask
                            (14x14 grid, fixed seed, A3 recipe)
  4. ct_target.png        — target slice  S_z
  5. ct_seg_overlay.png   — segmentation overlay
                            * if labels are available (--labels_root), uses
                              the real ground-truth mask;
                            * otherwise renders an *illustrative* overlay
                              from HU intensity bands and flags the file as
                              schematic (caption in the paper says so).

Inputs assumed (matches preprocess.py / dataset.py byte-for-byte):

  data/slices/train/sXXXX/NNNN.npy   float32 in [0,1], shape (224,224)

Optional inputs (real-overlay mode):

  data/labels/train/sXXXX.npy        uint8     shape (Z,224,224)

Outputs:

  figures/pipeline/*.png

Reproducibility:
  - Volume selection deterministic (sorted glob) given --hero_pid or
    auto-pick scoring.
  - Mask RNG seeded by --mask_seed (default 0).

Usage:
  # auto-pick hero slice, no labels (today on Mac)
  python scripts/make_pipeline_figure.py

  # specify hero slice and use real labels (later on RTX box)
  python scripts/make_pipeline_figure.py \\
      --hero_pid s0050 --hero_z 200 \\
      --labels_root data/labels
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Ten-organ subset, same order as prepare_labels.py / train_decoder.py.
ORGAN_NAMES = [
    "background", "liver", "spleen", "kidney_L", "kidney_R", "stomach",
    "pancreas", "lung_upper_L", "lung_upper_R", "heart", "aorta",
]
# Distinct colors per class (11 incl. background-transparent).
ORGAN_COLORS = [
    (0, 0, 0, 0),                # 0 background: transparent
    (0.85, 0.20, 0.20, 0.55),    # 1 liver: red
    (0.95, 0.55, 0.10, 0.55),    # 2 spleen: orange
    (0.20, 0.60, 0.85, 0.55),    # 3 kidney L: blue
    (0.45, 0.70, 0.95, 0.55),    # 4 kidney R: light blue
    (0.85, 0.85, 0.20, 0.55),    # 5 stomach: yellow
    (0.60, 0.30, 0.70, 0.55),    # 6 pancreas: purple
    (0.30, 0.75, 0.40, 0.55),    # 7 lung up L: green
    (0.55, 0.85, 0.55, 0.55),    # 8 lung up R: light green
    (0.95, 0.40, 0.65, 0.55),    # 9 heart: pink
    (0.70, 0.45, 0.20, 0.55),    # 10 aorta: brown
]
ORGAN_CMAP = ListedColormap([c[:3] + (c[3],) for c in ORGAN_COLORS])


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def load_volume_from_slices(pid_dir: Path) -> np.ndarray:
    """Stack per-slice .npy files into (Z, 224, 224) float32 in [0,1]."""
    files = sorted(pid_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy slices under {pid_dir}")
    arrs = [np.load(f) for f in files]
    return np.stack(arrs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Hero-slice selection
# ---------------------------------------------------------------------------

def score_slice_for_organs(sl: np.ndarray) -> float:
    """Heuristic abdomen-rich score. We want slices where the central
    image area is mostly soft tissue (organs) rather than lung air. Lung
    cavities punch holes in the central tissue band, so we reward central
    soft-tissue coverage and penalise lung-air fraction in the centre."""
    # Tissue intensity band (excludes air ~0 and dense bone ~1).
    band = (sl > 0.25) & (sl < 0.85)
    if band.sum() < 1000:
        return 0.0
    H, W = sl.shape
    cy, cx = H // 2, W // 2
    central = sl[cy - 60: cy + 60, cx - 60: cx + 60]
    central_tissue = ((central > 0.25) & (central < 0.85)).mean()
    central_air = (central < 0.10).mean()
    # Abdomen: central_tissue ~0.9, central_air ~0.0.
    # Chest:   central_tissue ~0.5-0.8, central_air >0.1 (lung cavities).
    return float(central_tissue - 1.5 * central_air)


def pick_hero(slices_root: Path, n_candidates: int = 12) -> tuple[str, int]:
    """Return (pid, z) of the highest-scoring slice across a small sample
    of patients. n_candidates patients evenly spaced through the dataset."""
    pids = sorted(p.name for p in slices_root.iterdir() if p.is_dir())
    step = max(1, len(pids) // n_candidates)
    sampled = pids[::step][:n_candidates]
    best = (None, -1, -1.0)  # pid, z, score
    for pid in sampled:
        try:
            vol = load_volume_from_slices(slices_root / pid)
        except Exception:
            continue
        Z = vol.shape[0]
        # Search mid-volume only (avoid lung apex and pelvis); pick best slice.
        z_lo, z_hi = int(0.35 * Z), int(0.75 * Z)
        for z in range(z_lo, z_hi):
            s = score_slice_for_organs(vol[z])
            if s > best[2]:
                best = (pid, z, s)
        print(f"  candidate {pid}: best z={best[1]} score={best[2]:.4f}")
    if best[0] is None:
        raise RuntimeError("no usable patient found")
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def render_slice(slice_2d: np.ndarray, out_path: Path,
                 cmap: str = "gray", dpi: int = 180) -> None:
    """Save a 2D slice as a square PNG (no axes, tight)."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi)
    ax.imshow(slice_2d, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_volume_stack(vol: np.ndarray, hero_z: int, out_path: Path,
                        n_back: int = 3, offset_frac: float = 0.06,
                        dpi: int = 180) -> None:
    """Single CT slice with a 'deck of slices' depth indicator behind it.

    The hero slice is drawn at full size in the foreground; n_back outlined
    rectangles are stacked behind it, offset diagonally up-right, to suggest
    the 3D volume without taking the vertical space of a full montage.
    """
    from matplotlib.patches import Rectangle
    H, W = vol[hero_z].shape
    offset = int(round(offset_frac * H))  # px offset per back layer
    pad = n_back * offset

    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # The image takes its natural axes (0..W, 0..H) with y inverted by imshow.
    # We place the hero slice at the lower-left, leaving `pad` of room
    # up-right for the back-layer stack.
    # Axis limits: x in [-pad, W],  y in [H, -pad]  (y inverted for imshow).
    ax.set_xlim(-1, W + pad + 1)
    ax.set_ylim(H + 1, -pad - 1)

    # Back layers first (so they end up underneath the imshow).
    # Each layer is at higher x (right) and higher y (up = lower y value).
    for i in range(n_back, 0, -1):
        dx = i * offset
        dy = -i * offset
        rect = Rectangle((dx, dy), W, H,
                         facecolor="0.92", edgecolor="0.55",
                         linewidth=0.6, zorder=0)
        ax.add_patch(rect)

    # Foreground: hero slice with imshow. Extent matches the back layers'
    # foreground position (0, 0, W, H) but in imshow extent format:
    # (left, right, bottom, top) — bottom > top because imshow flips.
    ax.imshow(vol[hero_z], cmap="gray", vmin=0.0, vmax=1.0,
              interpolation="nearest",
              extent=(0, W, H, 0), zorder=2)
    fg_rect = Rectangle((0, 0), W, H, facecolor="none",
                        edgecolor="0.30", linewidth=0.9, zorder=3)
    ax.add_patch(fg_rect)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_masked_slice(slice_2d: np.ndarray, out_path: Path,
                        mask_ratio: float = 0.75,
                        grid: int = 14, seed: int = 0,
                        dpi: int = 180) -> None:
    """75% random per-patch mask on a grid x grid grid (A3 recipe).
    Masked patches drawn as flat grey to highlight the masking pattern."""
    rng = np.random.default_rng(seed)
    H, W = slice_2d.shape
    ph, pw = H // grid, W // grid
    n_total = grid * grid
    n_masked = int(round(mask_ratio * n_total))
    mask_idx = rng.choice(n_total, size=n_masked, replace=False)
    mask_grid = np.zeros((grid, grid), dtype=bool)
    mask_grid.flat[mask_idx] = True

    out = slice_2d.copy()
    for i in range(grid):
        for j in range(grid):
            if mask_grid[i, j]:
                out[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = 0.35  # grey

    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi)
    ax.imshow(out, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    # Faint grid lines emphasise patches.
    for i in range(1, grid):
        ax.axhline(i * ph - 0.5, color="white", lw=0.15, alpha=0.4)
        ax.axvline(i * pw - 0.5, color="white", lw=0.15, alpha=0.4)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_seg_overlay(slice_2d: np.ndarray, mask_2d: np.ndarray | None,
                       out_path: Path, dpi: int = 180,
                       schematic: bool = False) -> None:
    """CT slice with segmentation overlay.

    If mask_2d is given, treat as class-indexed GT (0..10) and overlay.
    Otherwise generate a clearly *illustrative* overlay from intensity
    thresholds — captioned as schematic in the paper."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi)
    ax.imshow(slice_2d, cmap="gray", vmin=0.0, vmax=1.0,
              interpolation="nearest")

    if mask_2d is not None and not schematic:
        # Render true GT mask via the discrete cmap.
        # Use translucent overlay; transparent for class 0.
        rgba = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
        for cls_idx, color in enumerate(ORGAN_COLORS):
            rgba[mask_2d == cls_idx] = color
        ax.imshow(rgba, interpolation="nearest")
    else:
        # Schematic: 3 illustrative regions from intensity bands, clipped to
        # the eroded body interior. The skin-air gradient at the body contour
        # passes through low intensities, which would otherwise leak into the
        # "lung" class as a halo — erosion removes that. These are NOT real
        # organ labels — caption in the paper says so.
        from scipy.ndimage import binary_erosion
        body_raw = slice_2d > 0.08
        # 5-pixel erosion strips the skin-contour gradient.
        body = binary_erosion(body_raw, iterations=5)
        soft = body & (slice_2d > 0.45) & (slice_2d < 0.70)
        bone = body & (slice_2d > 0.78)
        lung = body & (slice_2d < 0.10)  # genuine lung air inside the body
        overlay = np.zeros((*slice_2d.shape, 4), dtype=np.float32)
        overlay[soft] = (0.85, 0.20, 0.20, 0.45)   # red — "soft tissue"
        overlay[lung] = (0.30, 0.75, 0.40, 0.45)   # green — "low-density"
        overlay[bone] = (0.95, 0.85, 0.20, 0.55)   # yellow — "high-density"
        ax.imshow(overlay, interpolation="nearest")

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices_root", default="data/slices/train",
                    help="dir containing sXXXX/ patient subdirs of per-slice .npy")
    ap.add_argument("--labels_root", default=None,
                    help="optional dir containing {pid}.npy class-indexed masks")
    ap.add_argument("--out_dir", default="figures/pipeline",
                    help="output PNG dir")
    ap.add_argument("--hero_pid", default=None,
                    help="force specific patient id (e.g. s0050)")
    ap.add_argument("--hero_z", type=int, default=None,
                    help="force specific axial slice index")
    ap.add_argument("--k_gap", type=int, default=5,
                    help="context-target slice gap (A3 default mid=5)")
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--mask_seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=14,
                    help="patch-grid size (ViT-B/16 on 224 -> 14)")
    args = ap.parse_args()

    slices_root = Path(args.slices_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Hero selection
    if args.hero_pid is None:
        print("[hero] auto-picking abdominal slice ...")
        pid, z = pick_hero(slices_root)
    else:
        pid = args.hero_pid
        z = args.hero_z if args.hero_z is not None else None
    print(f"[hero] using pid={pid}")

    vol = load_volume_from_slices(slices_root / pid)
    Z = vol.shape[0]
    if z is None:
        # Default: mid-volume slice if user passed only --hero_pid.
        z_lo, z_hi = int(0.35 * Z), int(0.75 * Z)
        scores = [score_slice_for_organs(vol[i]) for i in range(z_lo, z_hi)]
        z = z_lo + int(np.argmax(scores))
    z = int(np.clip(z, args.k_gap, Z - args.k_gap - 1))
    z_ctx = max(0, z - args.k_gap)
    print(f"[hero] z={z} (Z={Z}), z_ctx={z_ctx} (k_gap={args.k_gap})")

    # 2) Labels (optional)
    seg_mask = None
    schematic = True
    if args.labels_root is not None:
        lbl_path = Path(args.labels_root) / f"{pid}.npy"
        if lbl_path.exists():
            seg_mask = np.load(lbl_path)[z]
            schematic = False
            print(f"[seg] using real labels from {lbl_path}")
        else:
            print(f"[seg] labels not found at {lbl_path} -- using schematic")
    else:
        print("[seg] no --labels_root -- rendering schematic overlay")

    # 3) Render panels
    print(f"[render] writing PNGs to {out_dir}/")
    render_volume_stack(vol, z, out_dir / "ct_volume.png")
    render_slice(vol[z_ctx], out_dir / "ct_context.png")
    render_masked_slice(vol[z_ctx], out_dir / "ct_context_masked.png",
                        mask_ratio=args.mask_ratio,
                        grid=args.grid, seed=args.mask_seed)
    render_slice(vol[z], out_dir / "ct_target.png")
    render_seg_overlay(vol[z], seg_mask, out_dir / "ct_seg_overlay.png",
                       schematic=schematic)

    # 4) Provenance sidecar
    (out_dir / "PROVENANCE.txt").write_text(
        f"hero_pid={pid}\nhero_z={z}\nz_ctx={z_ctx}\nk_gap={args.k_gap}\n"
        f"mask_ratio={args.mask_ratio}\nmask_seed={args.mask_seed}\n"
        f"grid={args.grid}\nschematic_overlay={schematic}\n"
        f"labels_root={args.labels_root}\n"
    )
    print("[done] regenerate any time by re-running this script with the same flags.")


if __name__ == "__main__":
    main()
