"""
Phase 2C: End-to-end fine-tuning of pretrained ViT + segmentation head.

Mirrors train_decoder.py byte-for-byte on data selection, loss, and eval so the
result is apples-to-apples with the linear-probe sweep. Only difference: the ViT
is unfrozen and trained with a lower LR (and optional layer-wise LR decay).

Run-naming convention:
    runs/ft_v2_ssl_n50          # SSL-pretrained backbone, fine-tuned
    runs/ft_v2_random_n50       # random-init control (omit --ckpt)

Typical launch (RTX 3090 Ti):
    python scripts/train_finetune.py \\
        --ckpt ~/SSLP/checkpoints/vit_ep039.pt \\
        --slices_root data/slices --labels_root data/labels \\
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \\
        --head conv --epochs 50 --bs 8 \\
        --lr_head 1e-3 --lr_backbone 1e-5 --llrd 0.75 \\
        --warmup_epochs 3 --grad_checkpoint \\
        --out runs/ft_v2_ssl_n50

Random-init control:
    python scripts/train_finetune.py \\
        --slices_root data/slices --labels_root data/labels \\
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \\
        --head conv --epochs 50 --bs 8 \\
        --lr_head 1e-3 --lr_backbone 1e-5 --llrd 0.75 \\
        --warmup_epochs 3 --grad_checkpoint \\
        --out runs/ft_v2_random_n50
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.labeled import LabeledSlices
from src.models.seg_head import LinearSegHead, ConvSegHead


# ----------------------------------------------------------------------------
# Model loading (same as train_decoder.py — keep these byte-identical)
# ----------------------------------------------------------------------------

def build_vit():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)


def load_ctx_enc(vit, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        ctx_sd = {k.replace("context_enc.", ""): v
                  for k, v in sd["model"].items()
                  if k.startswith("context_enc.")}
    else:
        ctx_sd = sd
    missing, unexpected = vit.load_state_dict(ctx_sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return vit


def forward_tokens(vit, x):
    """Grad-enabled version of extract_tokens — drops CLS if present."""
    feats = vit.forward_features(x)
    n = feats.shape[1]
    side = int(round(n ** 0.5))
    if side * side != n:
        feats = feats[:, 1:, :]
    return feats


# ----------------------------------------------------------------------------
# Loss / eval (verbatim from train_decoder.py)
# ----------------------------------------------------------------------------

def dice_ce_loss(logits, target, ce_w=0.5, dice_w=0.5, bg_weight=0.1):
    C = logits.shape[1]
    weights = torch.ones(C, device=logits.device); weights[0] = bg_weight
    ce = F.cross_entropy(logits, target, weight=weights)
    probs = logits.softmax(dim=1)
    oh = F.one_hot(target, C).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * oh).sum(dim=dims)
    union = probs.sum(dim=dims) + oh.sum(dim=dims)
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    fg_dice = dice[1:].mean()
    return ce_w * ce + dice_w * (1.0 - fg_dice)


def per_class_dice(pred, target, C):
    sums = torch.zeros(C); counts = torch.zeros(C)
    for c in range(C):
        p = (pred == c); t = (target == c)
        denom = p.sum().float() + t.sum().float()
        if denom > 0:
            sums[c] += (2 * (p & t).sum().float() / denom)
            counts[c] += 1
    return sums, counts


@torch.no_grad()
def evaluate(vit, head, loader, C, device):
    vit.eval(); head.eval()
    s = torch.zeros(C); n = torch.zeros(C)
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = forward_tokens(vit, img)
            logits = head(tokens)
        pred = logits.argmax(dim=1).cpu()
        ds, dc = per_class_dice(pred, mask, C)
        s += ds; n += dc
    return s / n.clamp(min=1)


def get_audit(label_dir, eligible):
    cache = label_dir / "_audit.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"[audit] scanning {len(eligible)} labels (one-time, ~2 min)...", flush=True)
    audit = {}
    for i, pid in enumerate(eligible):
        m = np.load(label_dir / f"{pid}.npy", mmap_mode="r")
        cls = sorted(int(x) for x in np.unique(m) if x != 0)
        audit[pid] = {"classes": cls, "fg": int((m > 0).sum())}
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(eligible)}", flush=True)
    cache.write_text(json.dumps(audit))
    return audit


# ----------------------------------------------------------------------------
# Optimizer construction — two-tier LR with optional layer-wise decay
# ----------------------------------------------------------------------------

def build_param_groups(vit, head, lr_head, lr_backbone,
                       head_wd, backbone_wd, llrd):
    """Two-tier LR with optional layer-wise LR decay (LLRD) on the ViT.

    LLRD scales each block's LR by decay^(depth_from_top), so deeper layers
    (closer to the head) get larger updates and the patch embedder gets the
    smallest. Standard for ViT fine-tuning (DEIT, BEiT, I-JEPA, MAE).
    Set --llrd 1.0 to disable.
    """
    groups = []

    groups.append({
        "params": list(head.parameters()),
        "lr": lr_head, "weight_decay": head_wd, "name": "head",
    })

    if llrd is None or llrd >= 1.0:
        groups.append({
            "params": list(vit.parameters()),
            "lr": lr_backbone, "weight_decay": backbone_wd, "name": "vit",
        })
        return groups

    blocks = vit.blocks
    n_layers = len(blocks)

    # Embedding layer = depth 0 (smallest LR).
    embed_params = []
    for name, p in vit.named_parameters():
        if name.startswith("patch_embed") or name in ("pos_embed", "cls_token"):
            embed_params.append(p)
    if embed_params:
        scale = llrd ** (n_layers + 1)
        groups.append({
            "params": embed_params,
            "lr": lr_backbone * scale, "weight_decay": backbone_wd,
            "name": f"vit.embed (x{scale:.4f})",
        })

    for i, blk in enumerate(blocks):
        scale = llrd ** (n_layers - i)
        groups.append({
            "params": list(blk.parameters()),
            "lr": lr_backbone * scale, "weight_decay": backbone_wd,
            "name": f"vit.block{i:02d} (x{scale:.4f})",
        })

    # Final norm / fc_norm / anything else timm exposes at the top = full LR.
    matched = set(id(p) for g in groups for p in g["params"])
    rest = [p for p in vit.parameters() if id(p) not in matched]
    if rest:
        groups.append({
            "params": rest, "lr": lr_backbone, "weight_decay": backbone_wd,
            "name": "vit.top (x1.0000)",
        })
    return groups


def warmup_cosine(step, total_steps, warmup_steps):
    """Linear warmup [0, warmup_steps], cosine decay to 0 after."""
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Data (same as train_decoder.py)
    p.add_argument("--ckpt", default=None,
                   help="Pretrained ViT checkpoint. Omit for random-init control.")
    p.add_argument("--slices_root", required=True)
    p.add_argument("--labels_root", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--n_train_volumes", type=int, default=50)
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--require_full_coverage", action="store_true")
    p.add_argument("--min_fg_frac", type=float, default=0.0)
    # Head
    p.add_argument("--head", choices=["linear", "conv"], default="conv",
                   help="Conv default for fine-tuning. Use 'linear' for the "
                        "strictest apples-to-apples vs the linear probe.")
    # Optim
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=8,
                   help="Lower than linear-probe bs=16; backbone activations now in memory.")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=1e-5,
                   help="Standard is ~100x lower than head LR.")
    p.add_argument("--head_wd", type=float, default=1e-4)
    p.add_argument("--backbone_wd", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--llrd", type=float, default=0.75,
                   help="Layer-wise LR decay. 1.0 disables it.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Trade compute for VRAM. Use if OOM at bs >= 8.")
    # Misc
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data selection: identical logic to train_decoder.py -----------------
    train_slice_dir = Path(args.slices_root) / "train"
    train_label_dir = Path(args.labels_root) / "train"
    val_slice_dir   = Path(args.slices_root) / "val"
    val_label_dir   = Path(args.labels_root) / "val"

    img_pids   = {f.stem for f in train_slice_dir.glob("*.npy")}
    label_pids = {f.stem for f in train_label_dir.glob("*.npy") if f.stem != "_audit"}
    eligible   = sorted(img_pids & label_pids)

    if args.patients:
        train_pts = [pid for pid in args.patients if pid in eligible]
        missing   = [pid for pid in args.patients if pid not in eligible]
        if missing:
            print(f"[warn] forced patients not found: {missing}")
    else:
        audit = get_audit(train_label_dir, eligible)
        scored = [(audit[pid]["fg"], len(audit[pid]["classes"]), pid)
                  for pid in eligible if pid in audit]
        if args.require_full_coverage:
            n_organs = args.num_classes - 1
            scored = [s for s in scored if s[1] >= n_organs]
            print(f"[data] {len(scored)} patients with full {n_organs}-organ coverage")
        scored = sorted([s for s in scored if s[0] > 50000], key=lambda x: -x[0])
        pool = [pid for fg, n, pid in scored[: max(args.n_train_volumes * 4, 20)]]
        rng = random.Random(args.seed); rng.shuffle(pool)
        train_pts = pool[: args.n_train_volumes]

    (out / "train_volumes.json").write_text(json.dumps(train_pts, indent=2))
    print(f"[data] train volumes ({len(train_pts)}/{len(eligible)} eligible): {train_pts}")

    train_ds = LabeledSlices(train_slice_dir, train_label_dir,
                             patients=train_pts, min_fg_frac=args.min_fg_frac)
    val_ds   = LabeledSlices(val_slice_dir, val_label_dir,
                             min_fg_frac=args.min_fg_frac)
    print(f"[data] train slices={len(train_ds)}  val slices={len(val_ds)}")

    train_dl      = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    train_eval_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
    val_dl        = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

    # --- Model: ViT now requires_grad=True ----------------------------------
    vit = build_vit()
    if args.ckpt:
        vit = load_ctx_enc(vit, args.ckpt)
    else:
        print("[init] no --ckpt: random-init backbone (control run)")
    vit = vit.to(device)
    for p_ in vit.parameters():
        p_.requires_grad = True

    if args.grad_checkpoint and hasattr(vit, "set_grad_checkpointing"):
        vit.set_grad_checkpointing(True)
        print("[mem] gradient checkpointing ON for backbone")

    HeadCls = {"linear": LinearSegHead, "conv": ConvSegHead}[args.head]
    head = HeadCls(dim=768, num_classes=args.num_classes,
                   patch_size=16, img_size=224).to(device)

    # --- Optimizer with two-tier LR + LLRD ----------------------------------
    groups = build_param_groups(
        vit, head,
        lr_head=args.lr_head, lr_backbone=args.lr_backbone,
        head_wd=args.head_wd, backbone_wd=args.backbone_wd,
        llrd=args.llrd,
    )
    print(f"[opt] {len(groups)} param groups (head + {len(groups)-1} backbone tiers)")
    for g in groups:
        n_params = sum(p.numel() for p in g["params"]) / 1e6
        print(f"      {g['name']:<28}  lr={g['lr']:.2e}  wd={g['weight_decay']:.2e}  {n_params:.2f}M")
    opt = torch.optim.AdamW(groups)

    base_lrs     = [g["lr"] for g in opt.param_groups]
    steps_per_ep = max(1, len(train_dl))
    total_steps  = steps_per_ep * args.epochs
    warmup_steps = steps_per_ep * args.warmup_epochs

    # --- Train --------------------------------------------------------------
    best, log, gstep = 0.0, [], 0
    for ep in range(args.epochs):
        vit.train(); head.train()
        t0, running = time.time(), 0.0
        for img, mask in train_dl:
            mult = warmup_cosine(gstep, total_steps, warmup_steps)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * mult

            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = forward_tokens(vit, img)
                logits = head(tokens)
                loss   = dice_ce_loss(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                all_params = [p for g in opt.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            opt.step()
            running += loss.item()
            gstep += 1
        avg = running / steps_per_ep

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            md_t = evaluate(vit, head, train_eval_dl, args.num_classes, device)
            md_v = evaluate(vit, head, val_dl,        args.num_classes, device)
            fg_t = md_t[1:].mean().item()
            fg_v = md_v[1:].mean().item()
            per_v = [f"{d:.3f}" for d in md_v.tolist()]
            cur_head_lr = opt.param_groups[0]["lr"]
            print(f"ep{ep:03d} loss={avg:.4f} "
                  f"train_fg={fg_t:.4f} val_fg={fg_v:.4f} "
                  f"head_lr={cur_head_lr:.2e} "
                  f"val_per={per_v} ({time.time()-t0:.1f}s)")
            log.append({"ep": ep, "loss": avg,
                        "train_fg": fg_t, "val_fg": fg_v,
                        "train_per_class": md_t.tolist(),
                        "val_per_class":   md_v.tolist(),
                        "head_lr": cur_head_lr})
            if fg_v > best:
                best = fg_v
                torch.save({
                    "vit":  vit.state_dict(),
                    "head": head.state_dict(),
                    "ep": ep, "fg_mDice": fg_v, "args": vars(args),
                }, out / "ft_best.pt")
        else:
            print(f"ep{ep:03d} loss={avg:.4f} ({time.time()-t0:.1f}s)")

    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"[done] best val_fg_mDice={best:.4f}")


if __name__ == "__main__":
    main()
