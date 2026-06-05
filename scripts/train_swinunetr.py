"""
Phase 2D: SwinUNETR baseline (Tang et al. 2022, MICCAI/CVPR).

Loads MONAI's pretrained SwinUNETR encoder (5050-CT-volume pretraining) and
trains a 3D segmentation head on the SAME 50/100-patient subsets as our v2 SSL
fine-tune. Same val set, same Dice/CE loss, same metric reporting — making the
v2-vs-SwinUNETR comparison apples-to-apples.

Run-naming convention:
    runs/swin_lin_n50         # linear probe (frozen encoder)
    runs/swin_lin_n100
    runs/swin_ft_n50          # full fine-tune
    runs/swin_ft_n100

---------------------------------------------------------------------------
PREREQUISITES (do these ONCE on the Ubuntu box before first launch):

1. Install MONAI in the existing venv:
       source ~/SSLP/venv/bin/activate
       pip install 'monai>=1.4' einops
       # MONAI 1.3.0 has a Python 3.12 incompatibility (uses deprecated
       # importer.find_module()). 1.4+ fixes this. SwinUNETR API is stable.

2. Download the pretrained SwinUNETR encoder checkpoint:
       cd ~/SSLP/checkpoints
       wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
       # File size ~145 MB. SHA-256 in MONAI docs.

3. Verify it loads cleanly with a 1-line check:
       python -c "import torch; sd = torch.load('~/SSLP/checkpoints/model_swinvit.pt', map_location='cpu'); print('keys:', len(sd))"

---------------------------------------------------------------------------
TYPICAL LAUNCH (linear probe at N=50):
    python scripts/train_swinunetr.py \
        --pretrained ~/SSLP/checkpoints/model_swinvit.pt \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \
        --mode linear --epochs 50 --bs 1 --crop_size 96 \
        --lr 1e-3 \
        --out runs/swin_lin_n50

TYPICAL LAUNCH (full fine-tune at N=50):
    python scripts/train_swinunetr.py \
        --pretrained ~/SSLP/checkpoints/model_swinvit.pt \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \
        --mode finetune --epochs 30 --bs 1 --crop_size 96 \
        --lr_head 1e-3 --lr_backbone 1e-5 \
        --out runs/swin_ft_n50

Random-init control (no --pretrained):
    python scripts/train_swinunetr.py \
        --slices_root data/slices --labels_root data/labels \
        --num_classes 11 --n_train_volumes 50 --require_full_coverage \
        --mode linear --epochs 50 --bs 1 --crop_size 96 \
        --lr 1e-3 \
        --out runs/swin_lin_random_n50
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ----------------------------------------------------------------------------
# 3D Dataset — reconstructs volumes from the existing 2.5D slice stacks
# ----------------------------------------------------------------------------

class LabeledVolumes3D(Dataset):
    """Loads full 3D volumes from data/slices/<pid>.npy (shape [Z,H,W]) and
    returns random crops of shape (crop_size, crop_size, crop_size) along with
    matching label crops. For val mode, returns the full volume (caller does
    sliding-window inference).

    Slices are uint8 (0-255). Labels are uint8 with class indices 0..num_classes-1.
    Same preprocessing as the 2D pipeline — guaranteeing apples-to-apples vs ViT.
    """
    def __init__(self, slice_dir, label_dir, patients=None,
                 crop_size=96, mode="train", min_fg_frac=0.0,
                 iters_per_patient=10):
        self.slice_dir = Path(slice_dir)
        self.label_dir = Path(label_dir)
        self.crop_size = crop_size
        self.mode = mode
        self.min_fg_frac = min_fg_frac
        self.iters_per_patient = iters_per_patient

        if patients is None:
            img_pids   = {f.stem for f in self.slice_dir.glob("*.npy")}
            label_pids = {f.stem for f in self.label_dir.glob("*.npy") if f.stem != "_audit"}
            patients   = sorted(img_pids & label_pids)
        self.patients = list(patients)

    def __len__(self):
        if self.mode == "train":
            # iters_per_patient random crops per patient per epoch.
            # Default 10; total iters = N * iters_per_patient * epochs.
            # For SwinUNETR baselines, ~5k-15k total iters is sufficient
            # (Tang et al. 2022 use ~30k on BTCV).
            return len(self.patients) * self.iters_per_patient
        return len(self.patients)

    def _load_volume(self, pid):
        img = np.load(self.slice_dir / f"{pid}.npy", mmap_mode="r")   # [Z,H,W] uint8
        msk = np.load(self.label_dir / f"{pid}.npy", mmap_mode="r")   # [Z,H,W] uint8
        return img, msk

    def _random_crop(self, img, msk, k):
        Z, H, W = img.shape
        # Pad to at least crop_size in every dim
        pz = max(0, k - Z); ph = max(0, k - H); pw = max(0, k - W)
        if pz or ph or pw:
            img = np.pad(img, ((0, pz), (0, ph), (0, pw)), mode="constant")
            msk = np.pad(msk, ((0, pz), (0, ph), (0, pw)), mode="constant")
            Z, H, W = img.shape
        z = random.randint(0, Z - k)
        h = random.randint(0, H - k)
        w = random.randint(0, W - k)
        return img[z:z+k, h:h+k, w:w+k].copy(), msk[z:z+k, h:h+k, w:w+k].copy()

    def __getitem__(self, idx):
        pid = self.patients[idx % len(self.patients)]
        img, msk = self._load_volume(pid)
        if self.mode == "train":
            # Up to 10 retries to find a crop with foreground (matches 2D pipeline's min_fg_frac)
            for _ in range(10):
                ic, mc = self._random_crop(img, msk, self.crop_size)
                if self.min_fg_frac == 0 or (mc > 0).sum() / mc.size >= self.min_fg_frac:
                    break
            img_t = torch.from_numpy(ic).float().unsqueeze(0) / 255.0   # [1,D,H,W]
            msk_t = torch.from_numpy(mc).long()                         # [D,H,W]
            return img_t, msk_t
        else:
            # Val: return whole volume + label (caller does sliding window)
            img_t = torch.from_numpy(np.asarray(img)).float().unsqueeze(0) / 255.0
            msk_t = torch.from_numpy(np.asarray(msk)).long()
            return img_t, msk_t


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def build_swinunetr(num_classes, pretrained_path=None):
    """Build MONAI SwinUNETR with optional pretrained encoder."""
    from monai.networks.nets import SwinUNETR

    # MONAI 1.4+ removed img_size from the constructor — SwinUNETR now handles
    # variable input sizes natively. spatial_dims=3 is the default but made
    # explicit for clarity (works on both 1.4 and 1.5+).
    model = SwinUNETR(
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,           # Standard for the Tang et al. pretraining
        use_checkpoint=True,       # Gradient checkpointing for memory
        spatial_dims=3,
    )

    if pretrained_path is None:
        print("[init] no --pretrained: random-init SwinUNETR")
        return model

    # Load only the encoder weights (swinViT submodule).
    sd = torch.load(pretrained_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]

    # Auxiliary pretraining heads that don't exist in MONAI's standard SwinUNETR encoder.
    # These come from the original SSL pretraining (rotation prediction + reconstruction).
    AUX_HEAD_PREFIXES = ("rotation_head", "rot_head", "convTrans3d", "conv_trans3d",
                         "norm.weight", "norm.bias")   # top-level norm is recon head

    cleaned = {}
    for k, v in sd.items():
        nk = k
        # Strip common DDP/wrapper prefixes
        for prefix in ("module.", "swinViT.", "swin_vit."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # Skip auxiliary pretraining heads
        if any(nk == aux or nk.startswith(aux + ".") for aux in AUX_HEAD_PREFIXES):
            continue
        # Older Swin convention → MONAI 1.4+ convention for MLP block
        nk = nk.replace(".mlp.fc1.", ".mlp.linear1.")
        nk = nk.replace(".mlp.fc2.", ".mlp.linear2.")
        cleaned[nk] = v

    missing, unexpected = model.swinViT.load_state_dict(cleaned, strict=False)
    print(f"[load] swinViT: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 5 or len(unexpected) > 5:
        print(f"[load] sample missing:    {missing[:5]}")
        print(f"[load] sample unexpected: {unexpected[:5]}")
        if len(missing) > 20:
            print("[load] WARNING: many missing keys — encoder may be partially random-init.")
    return model


# ----------------------------------------------------------------------------
# Loss / eval — match train_decoder.py protocol
# ----------------------------------------------------------------------------

def dice_ce_loss(logits, target, ce_w=0.5, dice_w=0.5, bg_weight=0.1):
    C = logits.shape[1]
    weights = torch.ones(C, device=logits.device); weights[0] = bg_weight
    ce = F.cross_entropy(logits, target, weight=weights)
    probs = logits.softmax(dim=1)
    oh = F.one_hot(target, C).permute(0, 4, 1, 2, 3).float()   # [B,C,D,H,W]
    dims = (0, 2, 3, 4)
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
def evaluate(model, loader, num_classes, device, crop_size):
    """Sliding-window inference on full volumes."""
    from monai.inferers import sliding_window_inference
    model.eval()
    s = torch.zeros(num_classes); n = torch.zeros(num_classes)
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = sliding_window_inference(
                inputs=img,
                roi_size=(crop_size, crop_size, crop_size),
                sw_batch_size=2,
                predictor=model,
                overlap=0.25,
                mode="gaussian",
            )
        pred = logits.argmax(dim=1).cpu()
        ds, dc = per_class_dice(pred, mask, num_classes)
        s += ds; n += dc
    return s / n.clamp(min=1)


def get_audit(label_dir, eligible):
    """Identical to train_decoder.py — cache per-patient class coverage."""
    cache = Path(label_dir) / "_audit.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"[audit] scanning {len(eligible)} labels...", flush=True)
    audit = {}
    for i, pid in enumerate(eligible):
        m = np.load(Path(label_dir) / f"{pid}.npy", mmap_mode="r")
        cls = sorted(int(x) for x in np.unique(m) if x != 0)
        audit[pid] = {"classes": cls, "fg": int((m > 0).sum())}
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(eligible)}", flush=True)
    cache.write_text(json.dumps(audit))
    return audit


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default=None,
                   help="Path to MONAI SwinUNETR encoder weights (model_swinvit.pt). "
                        "Omit for random-init control.")
    p.add_argument("--slices_root", required=True)
    p.add_argument("--labels_root", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--n_train_volumes", type=int, default=50)
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--require_full_coverage", action="store_true")
    p.add_argument("--mode", choices=["linear", "finetune"], default="linear",
                   help="linear: freeze encoder. finetune: train all params.")
    p.add_argument("--crop_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=1,
                   help="SwinUNETR at 96^3 needs bs=1 on 24 GB. With grad checkpoint maybe bs=2.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="(linear mode) LR for the decoder + head.")
    p.add_argument("--lr_head", type=float, default=1e-3,
                   help="(finetune mode) LR for decoder + head.")
    p.add_argument("--lr_backbone", type=float, default=1e-5,
                   help="(finetune mode) LR for the encoder (swinViT).")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_fg_frac", type=float, default=0.001,
                   help="Reject crops with less than this foreground fraction.")
    p.add_argument("--iters_per_patient", type=int, default=10,
                   help="Random 3D crops per patient per epoch. Total iters = N * "
                        "iters_per_patient * epochs. SwinUNETR baselines typically "
                        "use ~5k-15k total iters; 10 * 50 patients * 25 epochs = "
                        "12.5k → 4-5 hr on this GPU.")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data selection: same as train_decoder.py / train_finetune.py ---
    train_slice_dir = Path(args.slices_root) / "train"
    train_label_dir = Path(args.labels_root) / "train"
    val_slice_dir   = Path(args.slices_root) / "val"
    val_label_dir   = Path(args.labels_root) / "val"

    img_pids   = {f.stem for f in train_slice_dir.glob("*.npy")}
    label_pids = {f.stem for f in train_label_dir.glob("*.npy") if f.stem != "_audit"}
    eligible   = sorted(img_pids & label_pids)

    if args.patients:
        train_pts = [pid for pid in args.patients if pid in eligible]
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

    train_ds = LabeledVolumes3D(train_slice_dir, train_label_dir,
                                patients=train_pts, crop_size=args.crop_size,
                                mode="train", min_fg_frac=args.min_fg_frac,
                                iters_per_patient=args.iters_per_patient)
    val_ds   = LabeledVolumes3D(val_slice_dir, val_label_dir,
                                crop_size=args.crop_size, mode="val")
    print(f"[data] train iters/epoch={len(train_ds)//args.bs}  val volumes={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # --- Model ---
    model = build_swinunetr(args.num_classes, pretrained_path=args.pretrained).to(device)

    if args.mode == "linear":
        for p_ in model.swinViT.parameters():
            p_.requires_grad = False
        model.swinViT.eval()
        print("[mode] LINEAR PROBE: swinViT frozen, decoder + segmentation head trainable")
        trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    else:
        print("[mode] FINETUNE: all params trainable, two-tier LR")
        backbone_params = list(model.swinViT.parameters())
        head_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("swinViT.")]
        opt = torch.optim.AdamW([
            {"params": head_params,     "lr": args.lr_head,     "weight_decay": args.wd, "name": "head"},
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 0.05,    "name": "swinViT"},
        ])

    n_train_param = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad) / 1e6
    print(f"[model] trainable params: {n_train_param:.1f}M")

    base_lrs     = [g["lr"] for g in opt.param_groups]
    steps_per_ep = max(1, len(train_dl))
    total_steps  = steps_per_ep * args.epochs
    warmup_steps = steps_per_ep * args.warmup_epochs

    def warmup_cosine(step):
        if step < warmup_steps: return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # --- Train ---
    best, log, gstep = 0.0, [], 0
    for ep in range(args.epochs):
        if args.mode == "linear":
            model.swinViT.eval()
            for m_name, m_mod in model.named_children():
                if m_name != "swinViT":
                    m_mod.train()
        else:
            model.train()

        t0, running = time.time(), 0.0
        for img, mask in train_dl:
            mult = warmup_cosine(gstep)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = base * mult

            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(img)
                loss = dice_ce_loss(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p_ for g in opt.param_groups for p_ in g["params"]],
                    args.grad_clip)
            opt.step()
            running += loss.item()
            gstep += 1
        avg = running / steps_per_ep

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            md_v = evaluate(model, val_dl, args.num_classes, device, args.crop_size)
            fg_v = md_v[1:].mean().item()
            per_v = [f"{d:.3f}" for d in md_v.tolist()]
            print(f"ep{ep:03d} loss={avg:.4f} val_fg={fg_v:.4f} "
                  f"val_per={per_v} ({time.time()-t0:.1f}s)")
            log.append({"ep": ep, "loss": avg, "val_fg": fg_v,
                        "val_per_class": md_v.tolist()})
            if fg_v > best:
                best = fg_v
                torch.save({
                    "model": model.state_dict(),
                    "ep": ep, "fg_mDice": fg_v, "args": vars(args),
                }, out / "swin_best.pt")
        else:
            print(f"ep{ep:03d} loss={avg:.4f} ({time.time()-t0:.1f}s)")

    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"[done] best val_fg_mDice={best:.4f}")


if __name__ == "__main__":
    main()
