"""
Phase 2B: Few-shot supervised decoder on frozen pretrained ViT.
Flat layout:
    data/slices/{train,val}/{pid}.npy   uint8 [Z,H,W]
    data/labels/{train,val}/{pid}.npy   uint8 [Z,H,W]   (from prepare_labels.py)
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.labeled import LabeledSlices
from src.models.seg_head import LinearSegHead, ConvSegHead


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


@torch.no_grad()
def extract_tokens(vit, x):
    feats = vit.forward_features(x)
    n = feats.shape[1]
    side = int(round(n ** 0.5))
    if side * side != n:
        feats = feats[:, 1:, :]
    return feats


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
    head.eval()
    s = torch.zeros(C); n = torch.zeros(C)
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = extract_tokens(vit, img)
            logits = head(tokens)
        pred = logits.argmax(dim=1).cpu()
        ds, dc = per_class_dice(pred, mask, C)
        s += ds; n += dc
    return s / n.clamp(min=1)


def get_audit(label_dir, eligible):
    """Cache per-patient class coverage. Scans once, reuses on subsequent runs."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--slices_root", required=True, help="data/slices")
    p.add_argument("--labels_root", required=True, help="data/labels")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--n_train_volumes", type=int, default=5)
    p.add_argument("--patients", nargs="+", default=None,
                   help="Force these patient IDs (overrides --n_train_volumes)")
    p.add_argument("--require_full_coverage", action="store_true",
                   help="Only pick patients that have all num_classes-1 organs present")
    p.add_argument("--head", choices=["linear", "conv"], default="linear")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_fg_frac", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    vit = build_vit()
    vit = (load_ctx_enc(vit, args.ckpt) if args.ckpt else vit).to(device).eval()
    for q in vit.parameters():
        q.requires_grad = False

    HeadCls = {"linear": LinearSegHead, "conv": ConvSegHead}[args.head]
    head = HeadCls(dim=768, num_classes=args.num_classes,
                   patch_size=16, img_size=224).to(device)

    opt   = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, log = 0.0, []
    for ep in range(args.epochs):
        head.train()
        t0, running = time.time(), 0.0
        for img, mask in train_dl:
            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = extract_tokens(vit, img)
                logits = head(tokens)
                loss   = dice_ce_loss(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()
        avg = running / max(1, len(train_dl))

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            md_t = evaluate(vit, head, train_eval_dl, args.num_classes, device)
            md_v = evaluate(vit, head, val_dl,        args.num_classes, device)
            fg_t = md_t[1:].mean().item()
            fg_v = md_v[1:].mean().item()
            per_v = [f"{d:.3f}" for d in md_v.tolist()]
            per_t = [f"{d:.3f}" for d in md_t.tolist()]
            print(f"ep{ep:03d} loss={avg:.4f} "
                  f"train_fg={fg_t:.4f} val_fg={fg_v:.4f} "
                  f"val_per={per_v} ({time.time()-t0:.1f}s)")
            log.append({"ep": ep, "loss": avg,
                        "train_fg": fg_t, "val_fg": fg_v,
                        "train_per_class": md_t.tolist(),
                        "val_per_class":   md_v.tolist()})
            if fg_v > best:
                best = fg_v
                torch.save({"head": head.state_dict(), "ep": ep, "fg_mDice": fg_v,
                            "args": vars(args)}, out / "head_best.pt")
        else:
            print(f"ep{ep:03d} loss={avg:.4f} ({time.time()-t0:.1f}s)")

    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"[done] best val_fg_mDice={best:.4f}")


if __name__ == "__main__":
    main()
