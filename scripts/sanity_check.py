"""Phase 2A sanity check — visualize what vit_ep034 has learned.

Produces two figures per slice:
  1. Last-layer CLS attention (where the global token "looks")
  2. Patch-to-patch cosine similarity from a centered query patch

Run on CPU so it doesn't disturb GPU training:
    cd ~/SSLP
    mkdir -p scripts sanity_out
    CUDA_VISIBLE_DEVICES="" python scripts/sanity_check.py \
        --ckpt checkpoints/vit_ep034.pt \
        --slices_dir data/slices/val \
        --out sanity_out \
        --n_slices 6
"""
import argparse, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def load_encoder(ckpt_path, device="cpu"):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    msd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    ctx = {k.replace("context_enc.", "", 1): v
           for k, v in msd.items() if k.startswith("context_enc.")}
    if not ctx:                       # old-format ckpt (just context_enc)
        ctx = msd
    vit = timm.create_model(
        "vit_base_patch16_224", pretrained=False,
        num_classes=0, global_pool="")
    missing, unexpected = vit.load_state_dict(ctx, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    vit.eval().to(device)
    return vit


def last_layer_attention(vit, x):
    """
    Capture post-softmax attention from the last block by hooking attn_drop.
    The input to attn_drop is the softmax(QK^T/sqrt(d)) tensor, shape [B, H, N, N].
    Robust across timm versions because we don't replace forward.
    """
    # Just before last_layer_attention(vit, x):
    for blk in vit.blocks:
        if hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = False

    last_block = vit.blocks[-1]
    last_attn = last_block.attn

    storage = {}

    def pre_hook(module, inputs):
        # inputs is a tuple; first element is the attention tensor pre-dropout
        a = inputs[0]
        storage["a"] = a.detach().cpu()

    handle = last_attn.attn_drop.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            _ = vit(x)
    finally:
        handle.remove()

    return storage["a"]   # [B, H, N, N]


def patch_features(vit, x):
    """Returns (B, N, D) patch tokens, no CLS."""
    with torch.no_grad():
        feats = vit.forward_features(x)   # (B, N+1, D) for ViT-B
    return feats[:, 1:, :]                # drop CLS


def load_random_slices(slices_dir, n=6, hw=224):
    """Load `n` random 224x224 slices from per-volume .npy files."""
    vols = sorted(Path(slices_dir).glob("*.npy"))
    if not vols:
        raise SystemExit(f"No .npy volumes in {slices_dir}")
    out = []
    for _ in range(n):
        f = random.choice(vols)
        vol = np.load(f, mmap_mode="r")        # (Z, H, W) uint8
        # pick a slice from the middle 60% (avoids empty top/bottom)
        z0, z1 = int(vol.shape[0] * 0.2), int(vol.shape[0] * 0.8)
        z = random.randint(z0, max(z0 + 1, z1 - 1))
        sl = np.array(vol[z]).astype(np.float32) / 255.0   # uint8 -> [0,1]
        if sl.shape != (hw, hw):
            sl = zoom(sl, (hw / sl.shape[0], hw / sl.shape[1]), order=1)
        out.append((f.stem, int(z), sl))
    return out



def render(vit, name, z, sl, out_dir):
    """Produce attention + similarity figures for one slice."""
    x = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).float()

    # --- Attention map (CLS -> patches, last layer)
    attn = last_layer_attention(vit, x)        # (1, N+1, N+1)
    # attn: [B, H, N, N] — average over heads, take CLS row (index 0), drop CLS col
    n_patches = attn.shape[-1] - 1               # 196
    side = int(n_patches ** 0.5)                 # 14
    cls_attn = attn.mean(dim=1)[0, 0, 1:].numpy().reshape(side, side)


    cls_up = zoom(cls_attn, (224/14, 224/14), order=1)

    # --- Patch similarity (centered query)
    feats = patch_features(vit, x)             # (1, 196, D)
    feats = F.normalize(feats, dim=-1)[0]      # (196, D)
    qi = 14 * 7 + 7                             # center patch (row 7, col 7)
    sim = (feats @ feats[qi]).numpy().reshape(14, 14)
    sim_up = zoom(sim, (224/14, 224/14), order=1)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(sl, cmap="gray")
    ax[0, 0].set_title(f"{name}  z={z}")
    ax[0, 1].imshow(sl, cmap="gray")
    ax[0, 1].imshow(cls_up, cmap="hot", alpha=0.5)
    ax[0, 1].set_title("CLS attention (last layer)")
    ax[1, 0].imshow(sl, cmap="gray")
    # mark query patch
    yq, xq = (qi // 14) * 16 + 8, (qi % 14) * 16 + 8
    ax[1, 0].scatter([xq], [yq], c="lime", s=80, marker="x")
    ax[1, 0].set_title(f"Query patch @ ({yq},{xq})")
    ax[1, 1].imshow(sl, cmap="gray")
    ax[1, 1].imshow(sim_up, cmap="coolwarm", alpha=0.55, vmin=-0.5, vmax=1.0)
    ax[1, 1].scatter([xq], [yq], c="lime", s=80, marker="x")
    ax[1, 1].set_title("Patch similarity to query (cosine)")
    for a in ax.flat: a.axis("off")
    plt.tight_layout()
    outp = Path(out_dir) / f"{name}_z{z:04d}.png"
    plt.savefig(outp, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slices_dir", default="data/slices/val")
    ap.add_argument("--out", default="sanity_out")
    ap.add_argument("--n_slices", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    vit = load_encoder(args.ckpt, device="cpu")
    slices = load_random_slices(args.slices_dir, n=args.n_slices)
    for name, z, sl in slices:
        render(vit, name, z, sl, args.out)
    print(f"\nDone. Inspect {args.out}/*.png")


if __name__ == "__main__":
    main()