"""Aggregate runs/*/log.json + runs/*/args.json into paper-ready figures.

Auto-discovers all run directories under --runs_root, groups by (head, init,
era), and produces three figures under --out:

  data_efficiency.png  best val_fg vs N labeled volumes
  per_organ.png        per-organ Dice bars at the headline N (default N=50)
  trajectory.png       val_fg vs epoch for every run, overlaid

A run dir must contain both args.json and log.json (the train_decoder.py outputs).
* "init"  = "SSL" if args["ckpt"] is set, else "Random".
* "era"   = "v2" if "v2" appears in the dir name, else "v1". This matters
            because v1 and v2 backbones share the same vit_ep039.pt filename
            but are completely different models from different training runs.

Use --era v2 (or v1) to filter to a single era for clean paper figures.
Use --era all (default) to plot everything with era-tagged labels.

Usage:
    python scripts/make_figures.py --runs_root runs --out figures
    python scripts/make_figures.py --runs_root runs --out figures_v2 --era v2
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


def load_runs(runs_root, era_filter="all"):
    """Return list of {dir, head, init, era, n, ckpt, log} for every valid run dir.
    Skips runs that finished before their first eval epoch (no val_fg in log).
    If era_filter is "v1" or "v2", filter to that era only."""
    out, skipped = [], []
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir():
            continue
        args_p, log_p = d / "args.json", d / "log.json"
        if not (args_p.exists() and log_p.exists()):
            continue
        args = json.loads(args_p.read_text())
        log = json.loads(log_p.read_text())
        if not log or not any("val_fg" in e for e in log):
            skipped.append(d.name)
            continue
        era = "v2" if "v2" in d.name else "v1"
        if era_filter != "all" and era != era_filter:
            continue
        out.append({
            "dir":  d.name,
            "head": args.get("head", "?"),
            "init": "SSL" if args.get("ckpt") else "Random",
            "era":  era,
            "n":    int(args.get("n_train_volumes", 0)),
            "ckpt": args.get("ckpt"),
            "log":  log,
        })
    if skipped:
        print(f"[skipped] {len(skipped)} runs with no val_fg eval: {skipped}")
    return out


def best_val_fg(log):
    vals = [e["val_fg"] for e in log if "val_fg" in e]
    return max(vals) if vals else float("nan")


def best_per_class(log):
    """Per-class Dice at the epoch with best val_fg."""
    best = max((e for e in log if "val_per_class" in e), key=lambda e: e["val_fg"])
    return np.array(best["val_per_class"])


_STYLE = {
    ("linear", "SSL",    "v1"): {"color": "#1f77b4", "marker": "o", "ls": ":",  "label": "linear / SSL v1"},
    ("linear", "SSL",    "v2"): {"color": "#1f77b4", "marker": "o", "ls": "-",  "label": "linear / SSL v2"},
    ("linear", "Random", "v1"): {"color": "#1f77b4", "marker": "o", "ls": "--", "label": "linear / Random v1"},
    ("linear", "Random", "v2"): {"color": "#1f77b4", "marker": "o", "ls": "-.", "label": "linear / Random v2"},
    ("conv",   "SSL",    "v1"): {"color": "#d62728", "marker": "s", "ls": ":",  "label": "conv / SSL v1"},
    ("conv",   "SSL",    "v2"): {"color": "#d62728", "marker": "s", "ls": "-",  "label": "conv / SSL v2"},
    ("conv",   "Random", "v1"): {"color": "#d62728", "marker": "s", "ls": "--", "label": "conv / Random v1"},
    ("conv",   "Random", "v2"): {"color": "#d62728", "marker": "s", "ls": "-.", "label": "conv / Random v2"},
}


def plot_data_efficiency(runs, out_path, era_filter):
    """One line per (head, init, era); x = N labeled volumes, y = best val_fg.
    Multiple datapoints at the same N (e.g. seed replicates) are averaged with
    error bars showing min/max."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    groups = {}
    for r in runs:
        groups.setdefault((r["head"], r["init"], r["era"]), []) \
              .append((r["n"], best_val_fg(r["log"])))

    for key, pts in sorted(groups.items()):
        # Aggregate replicates at the same N
        from collections import defaultdict
        by_n = defaultdict(list)
        for n, v in pts:
            by_n[n].append(v)
        xs = sorted(by_n)
        ys = [np.mean(by_n[n]) for n in xs]
        lo = [min(by_n[n]) for n in xs]
        hi = [max(by_n[n]) for n in xs]
        s = _STYLE.get(key, {"label": " / ".join(key)})
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                    capsize=2, **s)

    title = "Data efficiency: SSL pretraining vs random init"
    if era_filter != "all":
        title += f" ({era_filter} only)"
    ax.set_xlabel("Labeled training volumes (N)")
    ax.set_ylabel("Mean foreground Dice (best epoch)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_per_organ(runs, headline_n, era_filter, out_path):
    """Grouped bar chart at the headline N. Prefers v2 era when era_filter='all'."""
    # Filter to runs at the headline N
    candidates = [r for r in runs if r["n"] == headline_n]

    # Pick which era to plot
    target_era = era_filter if era_filter != "all" else (
        "v2" if any(r["era"] == "v2" for r in candidates) else "v1")
    candidates = [r for r in candidates if r["era"] == target_era]

    if not candidates:
        print(f"[skip per_organ] no runs at N={headline_n} era={target_era}")
        return

    # Pick which head to plot — prefer linear (the SSL canonical probe)
    head_pref = "linear" if any(r["head"] == "linear" for r in candidates) else "conv"
    candidates = [r for r in candidates if r["head"] == head_pref]

    by_init = {}
    for r in candidates:
        # If multiple replicates of the same init, keep the one with highest best_val_fg
        existing = by_init.get(r["init"])
        if existing is None or best_val_fg(r["log"]) > best_val_fg(existing["log"]):
            by_init[r["init"]] = r

    ssl, rnd = by_init.get("SSL"), by_init.get("Random")
    if not (ssl and rnd):
        print(f"[skip per_organ] need SSL+Random at N={headline_n} head={head_pref} era={target_era}")
        return

    ssl_pc = best_per_class(ssl["log"])[1:]
    rnd_pc = best_per_class(rnd["log"])[1:]

    x = np.arange(len(ORGANS)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, rnd_pc, w, label=f"Random ({target_era})", color="#888888")
    ax.bar(x + w/2, ssl_pc, w, label=f"SSL {target_era}",     color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=30, ha="right")
    ax.set_ylabel("Dice")
    ax.set_title(f"Per-organ Dice at N={headline_n}, {head_pref} head, {target_era} backbone (best epoch)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}  (used {ssl['dir']} vs {rnd['dir']})")


def plot_trajectory(runs, out_path):
    """val_fg vs epoch, every run overlaid. Era is part of the label."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in sorted(runs, key=lambda x: (x["era"], x["head"], x["init"], x["n"])):
        eps = [e["ep"] for e in r["log"] if "val_fg" in e]
        ys  = [e["val_fg"] for e in r["log"] if "val_fg" in e]
        if not eps:
            continue
        label = f"{r['head']}/{r['init']} {r['era']} N={r['n']}"
        ls = "-" if r["init"] == "SSL" else "--"
        if r["era"] == "v1":
            ls = ":" if r["init"] == "SSL" else (0, (3, 1, 1, 1))  # dotted variants for v1
        ax.plot(eps, ys, label=label, linestyle=ls, alpha=0.85, linewidth=1.4)
    ax.set_xlabel("Head training epoch")
    ax.set_ylabel("val_fg mean Dice")
    ax.set_title("Head training trajectories (val mean foreground Dice)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--headline_n", type=int, default=50)
    ap.add_argument("--era", choices=["all", "v1", "v2"], default="all",
                    help="Filter to a single backbone era. Default: all eras with era-tagged labels.")
    args = ap.parse_args()

    runs = load_runs(args.runs_root, era_filter=args.era)
    if not runs:
        raise SystemExit(f"No runs matching era={args.era} under {args.runs_root}")
    print(f"[loaded] {len(runs)} runs (era={args.era}):")
    for r in runs:
        print(f"  {r['dir']:30s}  head={r['head']:6s}  init={r['init']:6s}  era={r['era']}  N={r['n']:4d}  best_val_fg={best_val_fg(r['log']):.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plot_data_efficiency(runs, out / "data_efficiency.png", era_filter=args.era)
    plot_per_organ(runs, args.headline_n, era_filter=args.era, out_path=out / "per_organ.png")
    plot_trajectory(runs, out / "trajectory.png")


if __name__ == "__main__":
    main()
