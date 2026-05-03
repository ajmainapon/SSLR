import os, glob, re, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from src.data.dataset import SliceTriplet
from src.models.jepa import ZMWM_JEPA

KEEP = 3
SAVE_EVERY = 500

def rotate(pattern, keep=KEEP):
    # Sort by mtime so a newly-saved checkpoint is always considered "newest",
    # even when it sorts before older files lexically (e.g. ep000 vs ep039
    # left over from a prior run -- the alphabetical sort would prune ep000).
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for f in files[:-keep]:
        os.remove(f)

def save_ckpt(path, model, opt, global_step, epoch):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
    }, path)

def load_ckpt(path, model, opt, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        # New format — full state
        model.load_state_dict(sd["model"])
        if opt is not None and "opt" in sd:
            opt.load_state_dict(sd["opt"])
        gstep = sd.get("global_step", 0)
        epoch = sd.get("epoch", 0)
        print(f"[resume] full-state ckpt loaded: epoch={epoch}, gstep={gstep}", flush=True)
        return gstep, epoch + 1
    else:
        # Old format — just context_enc weights
        model.context_enc.load_state_dict(sd)
        model.target_enc.load_state_dict(model.context_enc.state_dict())
        m_ep   = re.search(r'vit_ep(\d+)', os.path.basename(path))
        m_step = re.search(r'vit_step(\d+)', os.path.basename(path))
        if m_step:
            gstep = int(m_step.group(1))
            epoch = gstep // 9000
        elif m_ep:
            epoch = int(m_ep.group(1)) + 1
            gstep = epoch * 9000
        else:
            gstep, epoch = 0, 0
        print(f"[resume] OLD-format ckpt: {path}", flush=True)
        print(f"[resume]   context_enc restored, target_enc <- context_enc", flush=True)
        print(f"[resume]   WARN: predictor / mask_token / pred_pos_embed / pred_norm are RANDOM", flush=True)
        print(f"[resume]   WARN: optimizer state is fresh; expect ~1 epoch of loss spike", flush=True)
        print(f"[resume]   resuming at epoch={epoch}, gstep={gstep}", flush=True)
        return gstep, epoch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mask_mode", choices=["block", "random"], default="block",
                    help="block = I-JEPA-style rectangular targets (default); "
                         "random = legacy per-patch mask (v1 behavior)")
    ap.add_argument("--mask_ratio", type=float, default=None,
                    help="Default: 0.4 for block, 0.75 for random")
    ap.add_argument("--n_blocks", type=int, default=4,
                    help="Number of rectangular target blocks (block mode only)")
    args = ap.parse_args()

    if args.mask_ratio is None:
        args.mask_ratio = 0.4 if args.mask_mode == "block" else 0.75
    print(f"[cfg] mask_mode={args.mask_mode} mask_ratio={args.mask_ratio} "
          f"n_blocks={args.n_blocks}", flush=True)

    device = "cuda"
    ds = SliceTriplet()
    dl = DataLoader(ds, batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True,
                    prefetch_factor=2)
    model = ZMWM_JEPA(mask_ratio=args.mask_ratio,
                      mask_mode=args.mask_mode,
                      n_blocks=args.n_blocks).to(device)
    opt = torch.optim.AdamW(
        list(model.context_enc.parameters()) + list(model.predictor.parameters()),
        lr=1.5e-4, weight_decay=0.05)
    scaler = GradScaler("cuda", enabled=False)
    accum = 2
    global_step = 0
    start_epoch = 0

    if args.resume:
        global_step, start_epoch = load_ckpt(args.resume, model, opt, device)

    os.makedirs("checkpoints", exist_ok=True)
    model.train()
    for epoch in range(start_epoch, args.epochs):
        opt.zero_grad()
        for step, (ctx, tgt) in enumerate(dl):
            ctx = ctx.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                pred, z = model(ctx, tgt)
                loss = F.smooth_l1_loss(pred, z) / accum
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.context_enc.parameters()) + list(model.predictor.parameters()),
                    max_norm=1.0)
                opt.step(); opt.zero_grad()
                model.update_target()
            global_step += 1
            if step % 50 == 0:
                print(f"ep{epoch} step{step} gstep{global_step} loss={loss.item()*accum:.4f}", flush=True)
            if global_step % SAVE_EVERY == 0:
                p = f"checkpoints/vit_step{global_step:07d}.pt"
                save_ckpt(p, model, opt, global_step, epoch)
                rotate("checkpoints/vit_step*.pt")
        save_ckpt(f"checkpoints/vit_ep{epoch:03d}.pt", model, opt, global_step, epoch)
        rotate("checkpoints/vit_ep*.pt")

if __name__ == "__main__":
    main()