"""Generate paper-ready figures for the A3-backbone story.

Reads runs/*/log.json files and writes four figures used in paper_draft.tex:

    figures_a3/
      fig1_data_efficiency_linprobe.png   # A3 vs v2 vs Random under linear probing
      fig2_per_organ_a3_ft_n50.png        # Per-organ Dice at A3 SSL ft N=50 (seed=0; matched to nnU-Net N=50)
      fig3_loo_ablation.png               # v2 vs A1 vs A3 linear probe — the LOO finding
      fig4_finetune_multiseed.png         # A3 SSL vs Random ft with 4-seed error bars

Each figure also saves a .pdf alongside for LaTeX inclusion.

Usage:
    cd ~/SSLR             # or wherever the SSLR/runs/ tree is
    python scripts/make_figures_a3.py
    # Or with custom paths:
    python scripts/make_figures_a3.py --runs_root runs --out figures_a3
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORGANS = [
    "liver", "spleen", "kidney_L", "kidney_R", "stomach", "pancreas",
    "lung_uL", "lung_uR", "heart", "aorta",
]

# nnU-Net 2D supervised reference, trained on N=50, same 57-patient val split,
# resampled to 224x224. Completed 1000-epoch run (runs/nnunet_2d_n50_final,
# 2026-06-04); supersedes the earlier epoch-181 checkpoint (mean 0.7770).
# Per-organ + mean read directly from runs/nnunet_2d_n50_final/log.json
# (do not change without re-deriving from runs/).
NNUNET = {
    "liver": 0.7927, "spleen": 0.8749, "kidney_L": 0.7548, "kidney_R": 0.8066,
    "stomach": 0.7491, "pancreas": 0.6781, "lung_uL": 0.8901, "lung_uR": 0.7763,
    "heart": 0.8375, "aorta": 0.8977,
}
NNUNET_MEAN_FG = 0.8058

# Consistent color palette — same across all figures
COL_A3      = "#2E86AB"   # blue: A3 (the headline)
COL_V2      = "#A23B72"   # purple: v2 (I-JEPA baseline)
COL_A1      = "#F18F01"   # orange: A1 (random masking ablation)
COL_RANDOM  = "#7B7B7B"   # gray: random init
COL_NNUNET  = "#525252"   # dark gray: nnU-Net reference line


def _readme_clean(ax):
    """README-friendly polish: drop the (redundant) title, despine top/right,
    subtle y-only grid, frameless legend, slightly larger labels. Keeps a WHITE
    background (not transparent) so the axis text and grey reference lines stay
    readable in GitHub light + dark mode and in the paper."""
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.tick_params(labelsize=10)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def best_val_fg(run_dir):
    """Return best validation mean foreground Dice from log.json, or None."""
    p = Path(run_dir) / "log.json"
    if not p.exists():
        return None
    try:
        log = json.load(open(p))
    except Exception:
        return None
    vals = [e["val_fg"] for e in log if "val_fg" in e]
    return max(vals) if vals else None


def best_per_class(run_dir):
    """Per-class Dice at the epoch with best val_fg. Returns 1D np.array of length C."""
    log = json.load(open(Path(run_dir) / "log.json"))
    best = max((e for e in log if "val_per_class" in e), key=lambda e: e["val_fg"])
    return np.array(best["val_per_class"])


def collect_a3_ft(runs_root, n):
    """4 seeds (0,1,2,3) of A3 SSL fine-tune at N."""
    vals = []
    for s in (0, 1, 2, 3):
        p = runs_root / f"ft_a3_ssl_n{n}_bblr1e-4_seed{s}"
        v = best_val_fg(p)
        if v is not None:
            vals.append(v)
    return np.array(vals)


def collect_random_ft(runs_root, n):
    """4 seeds (0,1,2,3) of Random fine-tune at N. seed=0 has no _seed suffix per the legacy queue."""
    vals = []
    p0 = runs_root / f"ft_v2_random_n{n}_bblr1e-4"
    v = best_val_fg(p0)
    if v is not None:
        vals.append(v)
    for s in (1, 2, 3):
        p = runs_root / f"ft_v2_random_n{n}_bblr1e-4_seed{s}"
        v = best_val_fg(p)
        if v is not None:
            vals.append(v)
    return np.array(vals)


def collect_amos_a3(runs_root, n):
    """4 seeds of AMOS A3 SSL linear probe at N."""
    vals = []
    p0 = runs_root / f"lin_amos_a3_n{n}"
    v = best_val_fg(p0)
    if v is not None:
        vals.append(v)
    for s in (1, 2, 3):
        p = runs_root / f"lin_amos_a3_n{n}_seed{s}"
        v = best_val_fg(p)
        if v is not None:
            vals.append(v)
    return np.array(vals)


def collect_amos_random(runs_root, n):
    """4 seeds of AMOS Random linear probe at N."""
    vals = []
    p0 = runs_root / f"lin_amos_random_n{n}"
    v = best_val_fg(p0)
    if v is not None:
        vals.append(v)
    for s in (1, 2, 3):
        p = runs_root / f"lin_amos_random_n{n}_seed{s}"
        v = best_val_fg(p)
        if v is not None:
            vals.append(v)
    return np.array(vals)



# ---------------------------------------------------------------------------
# Figure 1 — Data efficiency under linear probing
# ---------------------------------------------------------------------------

def fig1_data_efficiency(runs_root, out):
    Ns = [5, 10, 20, 50, 100]
    a3  = [best_val_fg(runs_root / f"lin_loo_a3_n{n}")  for n in Ns]
    v2  = [best_val_fg(runs_root / f"lin_v2_ssl_n{n}")  for n in Ns]
    rnd = [best_val_fg(runs_root / f"lin_v2_random_n{n}") for n in Ns]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(Ns, a3,  "o-",  label="A3 SSL (multi-block mask + narrow k)",
            color=COL_A3,     linewidth=2.0, markersize=8)
    ax.plot(Ns, v2,  "s--", label="v2 SSL (I-JEPA-inspired baseline)",
            color=COL_V2,     linewidth=1.5, markersize=7, alpha=0.85)
    ax.plot(Ns, rnd, "v:",  label="Random init",
            color=COL_RANDOM, linewidth=1.5, markersize=7, alpha=0.85)
    ax.axhline(NNUNET_MEAN_FG, linestyle="-.", color=COL_NNUNET, alpha=0.5,
               linewidth=1.2, label=f"nnU-Net supervised (N=50): {NNUNET_MEAN_FG:.3f}")

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of labeled training volumes (log scale)")
    ax.set_ylabel("Linear-probe mean foreground Dice")
    ax.legend(loc="upper left", fontsize=9)
    _readme_clean(ax)
    fig.tight_layout()
    fig.savefig(out / "fig1_data_efficiency_linprobe.png", dpi=160)
    fig.savefig(out / "fig1_data_efficiency_linprobe.pdf")
    plt.close(fig)
    print(f"  saved {out}/fig1_data_efficiency_linprobe.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 2 — Per-organ Dice at A3 fine-tune N=50 (matched to nnU-Net N=50)
# ---------------------------------------------------------------------------

def fig2_per_organ_ft(runs_root, out):
    # Matched budget: both A3 fine-tune and the nnU-Net reference use N=50.
    # (nnU-Net was only ever trained at N=50; comparing our N=50 fine-tune to it
    #  is apples-to-apples, unlike the earlier N=100-vs-N=50 figure.)
    log = best_per_class(runs_root / "ft_a3_ssl_n50_bblr1e-4_seed0")
    # log[0] = background; skip
    a3  = log[1:]
    nnu = np.array([NNUNET[o] for o in ORGANS])

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = np.arange(len(ORGANS))
    w = 0.4
    ax.bar(x - w/2, a3,  w, label="A3 SSL FT (N=50, seed=0)", color=COL_A3,     edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, nnu, w, label="nnU-Net supervised (N=50)", color=COL_RANDOM, edgecolor="black", linewidth=0.5, alpha=0.75)

    # Annotate recovery % above each A3 bar
    for i, (v, n) in enumerate(zip(a3, nnu)):
        pct = 100.0 * v / n
        ax.annotate(f"{pct:.0f}%", xy=(x[i] - w/2, v + 0.015),
                    ha="center", fontsize=8, fontweight="bold", color=COL_A3)

    ax.set_xticks(x)
    ax.set_xticklabels(ORGANS, rotation=30, ha="right")
    ax.set_ylabel("Foreground Dice")
    ax.set_title(f"Per-organ Dice at A3 SSL fine-tune, N=50 (mean fg = {a3.mean():.4f})")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out / "fig2_per_organ_a3_ft_n50.png", dpi=150)
    fig.savefig(out / "fig2_per_organ_a3_ft_n50.pdf")
    plt.close(fig)
    print(f"  saved {out}/fig2_per_organ_a3_ft_n50.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 3 — LOO ablation: v2 vs A1 vs A3 under linear probing
# ---------------------------------------------------------------------------

def fig3_loo_ablation(runs_root, out):
    Ns = [5, 10, 20, 50, 100]
    v2 = [best_val_fg(runs_root / f"lin_v2_ssl_n{n}")    for n in Ns]
    a1 = [best_val_fg(runs_root / f"lin_loo_a1_n{n}")    for n in Ns]
    a3 = [best_val_fg(runs_root / f"lin_loo_a3_n{n}")    for n in Ns]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(Ns, v2, "s--", label="v2 (multi-block mask + wide k, I-JEPA-inspired)",
            color=COL_V2, linewidth=1.5, markersize=7)
    ax.plot(Ns, a1, "^-",  label="A1 (random mask + wide k)",
            color=COL_A1, linewidth=1.5, markersize=7)
    ax.plot(Ns, a3, "o-",  label="A3 (multi-block mask + narrow k)",
            color=COL_A3, linewidth=2.0, markersize=8)

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of labeled training volumes (log scale)")
    ax.set_ylabel("Linear-probe mean foreground Dice")
    ax.legend(loc="upper left", fontsize=9)
    _readme_clean(ax)
    fig.tight_layout()
    fig.savefig(out / "fig3_loo_ablation.png", dpi=160)
    fig.savefig(out / "fig3_loo_ablation.pdf")
    plt.close(fig)
    print(f"  saved {out}/fig3_loo_ablation.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 4 — Fine-tune cross-N with 4-seed error bars
# ---------------------------------------------------------------------------

def fig4_finetune_multiseed(runs_root, out):
    Ns = [20, 50, 100]
    a3_means, a3_stds = [], []
    rnd_means, rnd_stds = [], []
    for n in Ns:
        a3  = collect_a3_ft(runs_root, n)
        rnd = collect_random_ft(runs_root, n)
        if len(a3) < 2 or len(rnd) < 2:
            print(f"  WARN: incomplete seeds at N={n}: a3={len(a3)} rnd={len(rnd)}")
        a3_means.append(a3.mean());  a3_stds.append(a3.std(ddof=1) if len(a3) > 1 else 0.0)
        rnd_means.append(rnd.mean()); rnd_stds.append(rnd.std(ddof=1) if len(rnd) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(Ns, a3_means,  yerr=a3_stds,  fmt="o-",  color=COL_A3,
                linewidth=2.0, markersize=8, capsize=6, capthick=1.5,
                label="A3 SSL ft (4-seed mean ± std)")
    ax.errorbar(Ns, rnd_means, yerr=rnd_stds, fmt="v--", color=COL_RANDOM,
                linewidth=1.5, markersize=7, capsize=6, capthick=1.5,
                label="Random ft (4-seed mean ± std)", alpha=0.85)
    ax.axhline(NNUNET_MEAN_FG, linestyle="-.", color=COL_NNUNET, alpha=0.5,
               linewidth=1.2, label=f"nnU-Net supervised: {NNUNET_MEAN_FG:.3f}")

    # Annotate Δ rel above the A3 line at each N
    for n, am, rm in zip(Ns, a3_means, rnd_means):
        delta = 100.0 * (am - rm) / rm
        ax.annotate(f"+{delta:.1f}%", xy=(n, am + 0.03), ha="center",
                    fontsize=9, fontweight="bold", color=COL_A3)

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of labeled training volumes (log scale)")
    ax.set_ylabel("Fine-tune mean foreground Dice")
    ax.set_title("End-to-end fine-tune (lr$_{bb}$=1e-4): A3 SSL vs Random init")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_finetune_multiseed.png", dpi=150)
    fig.savefig(out / "fig4_finetune_multiseed.pdf")
    plt.close(fig)
    print(f"  saved {out}/fig4_finetune_multiseed.{{png,pdf}}")

    # Also print the table for sanity / docs
    print("\n  4-seed fine-tune summary (for sanity):")
    print(f"  {'N':<5} {'A3 SSL mean±std':<24} {'Random mean±std':<24} {'Δ rel':<10}")
    for n, am, asd, rm, rsd in zip(Ns, a3_means, a3_stds, rnd_means, rnd_stds):
        d = 100.0 * (am - rm) / rm
        print(f"  {n:<5} {am:.4f} ± {asd:.4f}      {rm:.4f} ± {rsd:.4f}      {d:+.2f}%")


# ---------------------------------------------------------------------------
# Figure 5 — AMOS cross-dataset linear probe
# ---------------------------------------------------------------------------

def fig5_amos_cross_dataset(runs_root, out):
    """A3 backbone transferred to AMOS22 (frozen weights, linear probe).
    Side-by-side with TotalSeg in-dataset to show:
        - SSL absolute Dice is comparable across datasets (or even higher on AMOS)
        - Random init *falls* on AMOS, *widening* the SSL/Random gap to +148-249%
    """
    Ns = [20, 50, 100]
    amos_a3_means, amos_a3_stds = [], []
    amos_rnd_means, amos_rnd_stds = [], []
    for n in Ns:
        a3  = collect_amos_a3(runs_root, n)
        rnd = collect_amos_random(runs_root, n)
        amos_a3_means.append(a3.mean())
        amos_a3_stds.append(a3.std(ddof=1) if len(a3) > 1 else 0.0)
        amos_rnd_means.append(rnd.mean())
        amos_rnd_stds.append(rnd.std(ddof=1) if len(rnd) > 1 else 0.0)

    amos_a3_means = np.array(amos_a3_means)
    amos_rnd_means = np.array(amos_rnd_means)

    ts_a3     = [best_val_fg(runs_root / f"lin_loo_a3_n{n}")      for n in Ns]
    ts_rnd    = [best_val_fg(runs_root / f"lin_v2_random_n{n}")   for n in Ns]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    # Plot AMOS with error bars
    ax.errorbar(Ns, amos_a3_means, yerr=amos_a3_stds, fmt="o-", color=COL_A3, linewidth=2.2, markersize=9, capsize=5, capthick=1.5,
            label="AMOS A3 SSL (cross-dataset, 4-seed mean ± std)")
    ax.errorbar(Ns, amos_rnd_means, yerr=amos_rnd_stds, fmt="v-", color=COL_RANDOM, linewidth=1.8, markersize=8, capsize=5, capthick=1.5,
            label="AMOS Random init (cross-dataset, 4-seed mean ± std)")
    ax.plot(Ns, ts_a3,    "o--", color=COL_A3,     linewidth=1.5, markersize=7,
            alpha=0.55, label="TotalSeg A3 SSL (in-dataset, for comparison)")
    ax.plot(Ns, ts_rnd,   "v--", color=COL_RANDOM, linewidth=1.5, markersize=7,
            alpha=0.55, label="TotalSeg Random (in-dataset, for comparison)")

    # Annotate Delta_rel above each AMOS A3 point
    for n, a3, rnd in zip(Ns, amos_a3_means, amos_rnd_means):
        rel = 100.0 * (a3 - rnd) / rnd
        ax.annotate(f"+{rel:.1f}%", xy=(n, a3 + 0.012), ha="center",
                    fontsize=10, fontweight="bold", color=COL_A3)

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of labeled training volumes (log scale)")
    ax.set_ylabel("Linear-probe mean foreground Dice")
    ax.set_title("Cross-dataset transfer: A3 backbone (TotalSeg-pretrained) on AMOS22\n"
                 "SSL is dataset-invariant; Random collapses on the new distribution")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(list(amos_a3_means) + ts_a3) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "fig5_amos_cross_dataset.png", dpi=150)
    fig.savefig(out / "fig5_amos_cross_dataset.pdf")
    plt.close(fig)
    print(f"  saved {out}/fig5_amos_cross_dataset.{{png,pdf}}")

    # Also print the cross-dataset sanity table
    print("\n  AMOS cross-dataset vs TotalSeg in-dataset (linear probe):")
    print(f"  {'N':<5} {'TS-A3':<8} {'TS-Rnd':<8} {'AMOS-A3 mean±std':<20} {'AMOS-Rnd mean±std':<20} {'TS gap':<10} {'AMOS gap':<10}")
    for i, n in enumerate(Ns):
        ta = ts_a3[i]
        tr = ts_rnd[i]
        aa = amos_a3_means[i]
        asd = amos_a3_stds[i]
        ar = amos_rnd_means[i]
        rsd = amos_rnd_stds[i]
        ts_rel = 100*(ta-tr)/tr
        amos_rel = 100*(aa-ar)/ar
        print(f"  {n:<5} {ta:.4f}   {tr:.4f}   {aa:.4f}±{asd:.4f}        {ar:.4f}±{rsd:.4f}        {ts_rel:+.1f}%      {amos_rel:+.1f}%")


def fig6_mri_cross_modality(runs_root, out):
    """CT-pretrained A3 backbone applied to AMOS-MR (frozen, linear probe).
    The backbone never saw an MR slice. Shows the SSL/Random gap survives the
    CT->MR modality shift, though smaller than the within-CT cross-dataset gap.
    4 seeds: lin_mri_{a3,random}_n{N} (seed 0) + _seed{1,2,3}.

    N in {20, 50} only. Only 39 MR train volumes pass full 7-class coverage, so
    N=50 is capped to all 39 and N=100 uses the identical 39-volume set (a pure
    duplicate). N=100 is dropped to avoid a spurious data-efficiency point.
    """
    Ns = [20, 50]
    SEEDS = [0, 1, 2, 3]

    def collect_mri(init, n):
        vals = []
        for s in SEEDS:
            name = f"lin_mri_{init}_n{n}" if s == 0 else f"lin_mri_{init}_n{n}_seed{s}"
            v = best_val_fg(runs_root / name)
            if v is not None:
                vals.append(v)
        return np.array(vals)

    a3  = [collect_mri("a3", n)     for n in Ns]
    rnd = [collect_mri("random", n) for n in Ns]

    if any(len(v) == 0 for v in a3 + rnd):
        print("  [fig6] MRI runs missing — skipping")
        return

    a3_mean  = np.array([v.mean() for v in a3])
    a3_std   = np.array([v.std()  for v in a3])   # population std (ddof=0), matches queue summary
    rnd_mean = np.array([v.mean() for v in rnd])
    rnd_std  = np.array([v.std()  for v in rnd])
    n_seeds  = min(len(v) for v in a3 + rnd)

    # AMOS-CT cross-dataset A3 for a dashed CT reference (shows modality erosion).
    amos_a3 = np.array([collect_amos_a3(runs_root, n).mean() for n in Ns])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(Ns, a3_mean, yerr=a3_std, fmt="o-", color=COL_A3, linewidth=2.2,
                markersize=9, capsize=4,
                label=f"AMOS-MR A3 SSL (CT-pretrained, frozen, {n_seeds} seeds)")
    ax.errorbar(Ns, rnd_mean, yerr=rnd_std, fmt="v-", color=COL_RANDOM,
                linewidth=1.8, markersize=8, capsize=4,
                label=f"AMOS-MR Random init ({n_seeds} seeds)")
    ax.plot(Ns, amos_a3, "o--", color=COL_A3, linewidth=1.5, markersize=7,
            alpha=0.5, label="AMOS-CT A3 SSL (cross-dataset, for comparison)")

    for n, a3m, rndm in zip(Ns, a3_mean, rnd_mean):
        rel = 100.0 * (a3m - rndm) / rndm
        ax.annotate(f"+{rel:.1f}%", xy=(n, a3m + 0.012), ha="center",
                    fontsize=10, fontweight="bold", color=COL_A3)

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Number of labeled MR training volumes (log scale)")
    ax.set_ylabel("Linear-probe mean foreground Dice")
    ax.set_title("Cross-modality transfer: CT-pretrained A3 backbone on AMOS-MR\n"
                 f"SSL features survive the CT->MR modality shift ({n_seeds} seeds)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(a3_mean.max(), amos_a3.max()) * 1.18)
    fig.tight_layout()
    fig.savefig(out / "fig6_mri_cross_modality.png", dpi=150)
    fig.savefig(out / "fig6_mri_cross_modality.pdf")
    plt.close(fig)
    print(f"  saved {out}/fig6_mri_cross_modality.{{png,pdf}}")

    print(f"\n  AMOS-MR cross-modality linear probe ({n_seeds} seeds):")
    print(f"  {'N':<5} {'MR-A3 mean±std':<18} {'MR-Rnd mean±std':<18} {'MR gap':<10}")
    for i, n in enumerate(Ns):
        rel = 100 * (a3_mean[i] - rnd_mean[i]) / rnd_mean[i]
        print(f"  {n:<5} {a3_mean[i]:.4f}±{a3_std[i]:.4f}    "
              f"{rnd_mean[i]:.4f}±{rnd_std[i]:.4f}    {rel:+.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="runs",
                   help="Directory containing run dirs (e.g. ft_a3_ssl_n50_bblr1e-4_seed0/).")
    p.add_argument("--out", default="figures_a3",
                   help="Output directory for figures.")
    p.add_argument("--figs", default="all",
                   help="Comma-separated figure numbers to regenerate "
                        "(e.g. '1,2'), or 'all'. Lets you regen a subset "
                        "without overwriting others.")
    args = p.parse_args()

    runs_root = Path(args.runs_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(exist_ok=True, parents=True)

    if args.figs.strip().lower() == "all":
        want = {1, 2, 3, 4, 5, 6}
    else:
        want = {int(x) for x in args.figs.split(",") if x.strip()}

    print(f"Reading runs from: {runs_root}")
    print(f"Writing figures to: {out}")
    print(f"Regenerating figures: {sorted(want)}\n")

    if 1 in want:
        print("Figure 1: Data efficiency under linear probing")
        fig1_data_efficiency(runs_root, out)

    if 2 in want:
        print("Figure 2: Per-organ Dice at A3 SSL ft N=50 (matched to nnU-Net N=50)")
        fig2_per_organ_ft(runs_root, out)

    if 3 in want:
        print("Figure 3: LOO ablation")
        fig3_loo_ablation(runs_root, out)

    if 4 in want:
        print("Figure 4: Fine-tune cross-N with 4-seed error bars")
        fig4_finetune_multiseed(runs_root, out)

    if 5 in want:
        print("Figure 5: AMOS cross-dataset linear probe")
        fig5_amos_cross_dataset(runs_root, out)

    if 6 in want:
        print("Figure 6: AMOS-MR cross-modality linear probe")
        fig6_mri_cross_modality(runs_root, out)

    print(f"\nFigures saved to: {out}/")
    print("LaTeX inclusion: replace figures_v2/data_efficiency.png with figures_a3/fig1_data_efficiency_linprobe.png")


if __name__ == "__main__":
    main()
