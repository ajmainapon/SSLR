"""Aggregate runs/*/log.json + runs/*/args.json into paper-ready figures.

Auto-discovers all run directories under --runs_root, groups by (head, init),
and produces three figures under --out:

  data_efficiency.png  best val_fg vs N labeled volumes (SSL vs Random per head)
  per_organ.png        per-organ Dice bars at the headline N (default N=50)
  trajectory.png       val_fg vs epoch for every run, overlaid

A run dir must contain both args.json and log.json (the train_decoder.py outputs).
The init label is "SSL" if args["ckpt"] is set, else "Random".

Usage:
    python scripts/make_figures.py --runs_root runs --out figures
    python scripts/make_figures.py --runs_root runs --out figures --headline_n 50
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORGANS = [
    "liver", "spleen", "kidney_L", "kidney_R", "stomach", "pancreas",
    "lung_uL", "lung_uR", "heart", "aorta",
]


def load_runs(runs_root):
    """Return list of {dir, head, init, n, ckpt, log} for every valid run dir."""
    out = []
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir():
            continue
        args_p, log_p = d / "args.json", d / "log.json"
        if not (args_p.exists() and log_p.exists()):
            continue
        args = json.loads(args_p.read_text())
        log = json.loads(log_p.read_text())
        if not log:
            continue
        out.append({
            "dir":  d.name,
            "head": args.get("head", "?"),
            "init": "SSL" if args.get("ckpt") else "Random",
            "n":    int(args.get("n_train_volumes", 0)),
            "ckpt": args.get("ckpt"),
            "log":  log,
        })
    return out


def best_val_fg(log):
    return max(e["val_fg"] for e in log if "val_fg" in e)


def best_per_class(log):
    """Per-class Dice at the epoch with best val_fg."""
    best = max((e for e in log if "val_per_class" in e), key=lambda e: e["val_fg"])
    return np.array(best["val_per_class"])


def plot_data_efficiency(runs, out_path):
    """One line per (head, init); x = N labeled volumes, y = best val_fg."""
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = {}
    for r in runs:
        if "val_fg" not in r["log"][0] and not any("val_fg" in e for e in r["log"]):
            continue
        groups.setdefault((r["head"], r["init"]), []).append((r["n"], best_val_fg(r["log"])))

    style = {
        ("linear", "SSL"):    {"color": "C0", "marker": "o", "ls": "-",  "label": "linear / SSL v2"},
        ("linear", "Random"): {"color": "C0", "marker": "o", "ls": "--", "label": "linear / Random"},
        ("conv",   "SSL"):    {"color": "C3", "marker": "s", "ls": "-",  "label": "conv / SSL v2"},
        ("conv",   "Random"): {"color": "C3", "marker": "s", "ls": "--", "label": "conv / Random"},
    }
    for key, pts in sorted(groups.items()):
        pts.sort()
        xs, ys = zip(*pts)
        s = style.get(key, {"label": f"{key[0]} / {key[1]}"})
        ax.plot(xs, ys, **s)

    ax.set_xlabel("Labeled training volumes (N)")
    ax.set_ylabel("Mean foreground Dice (best epoch)")
    ax.set_title("Data efficiency: SSL pretraining vs random init")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_per_organ(runs, headline_n, out_path):
    """Grouped bar chart at the headline N: per-organ Dice for SSL vs Random."""
    by_init = {}
    for r in runs:
        if r["n"] != headline_n:
            continue
        # Prefer linear if both heads available (it's the standard SSL probe)
        existing = by_init.get((r["head"], r["init"]))
        if existing is None or r["head"] == "linear":
            by_init[(r["head"], r["init"])] = r

    if not by_init:
        print(f"[skip] no runs with N={headline_n}")
        return

    # Pick one head to plot — prefer linear, else conv
    head_priority = ["linear", "conv"]
    chosen_head = next((h for h in head_priority
                        if any(k[0] == h for k in by_init)), None)
    ssl  = by_init.get((chosen_head, "SSL"))
    rnd  = by_init.get((chosen_head, "Random"))
    if not (ssl and rnd):
        print(f"[skip] need both SSL and Random at N={headline_n} for head={chosen_head}")
        return

    ssl_pc = best_per_class(ssl["log"])[1:]   # drop background
    rnd_pc = best_per_class(rnd["log"])[1:]

    x = np.arange(len(ORGANS))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, rnd_pc, w, label="Random init",   color="#888888")
    ax.bar(x + w/2, ssl_pc, w, label="SSL v2",        color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=30, ha="right")
    ax.set_ylabel("Dice")
    ax.set_title(f"Per-organ Dice at N={headline_n}, {chosen_head} head (best epoch)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_trajectory(runs, out_path):
    """val_fg vs epoch, every run overlaid."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in sorted(runs, key=lambda x: (x["head"], x["init"], x["n"])):
        eps = [e["ep"] for e in r["log"] if "val_fg" in e]
        ys  = [e["val_fg"] for e in r["log"] if "val_fg" in e]
        if not eps:
            continue
        label = f"{r['head']}/{r['init']} N={r['n']}"
        ls = "-" if r["init"] == "SSL" else "--"
        ax.plot(eps, ys, ls, label=label, alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Head training epoch")
    ax.set_ylabel("val_fg mean Dice")
    ax.set_title("Head training trajectories (val mean foreground Dice)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--headline_n", type=int, default=50)
    args = ap.parse_args()

    runs = load_runs(args.runs_root)
    if not runs:
        raise SystemExit(f"No runs with both args.json + log.json under {args.runs_root}")
    print(f"[loaded] {len(runs)} runs:")
    for r in runs:
        print(f"  {r['dir']:30s}  head={r['head']:6s}  init={r['init']:6s}  N={r['n']:4d}  best_val_fg={best_val_fg(r['log']):.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plot_data_efficiency(runs, out / "data_efficiency.png")
    plot_per_organ(runs, args.headline_n, out / "per_organ.png")
    plot_trajectory(runs, out / "trajectory.png")


if __name__ == "__main__":
    main()
