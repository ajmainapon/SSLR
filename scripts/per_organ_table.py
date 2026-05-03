"""Generate per-organ Dice comparison table from a pair of train_decoder.py runs.

Reads the best-epoch per-class Dice from runs/<ssl_run>/log.json and
runs/<random_run>/log.json, prints a markdown + LaTeX table, and saves both
to the figures dir.

Default: compares lin_v2_ssl_n50 vs lin_v2_random_n50 (the headline N=50
linear-probe pair). Override with --ssl_run / --random_run for other pairs.
"""
import argparse, json
from pathlib import Path
import numpy as np


ORGANS = [
    "liver", "spleen", "kidney_L", "kidney_R", "stomach", "pancreas",
    "lung_uL", "lung_uR", "heart", "aorta",
]


def best_per_class(log_path):
    log = json.loads(Path(log_path).read_text())
    best = max((e for e in log if "val_per_class" in e),
               key=lambda e: e["val_fg"])
    return np.array(best["val_per_class"]), best["val_fg"], best["ep"]


def fmt_ratio(s, r):
    if r == 0:
        return "∞" if s > 0 else "—"
    return f"{s / r:.2f}×"


def render_markdown(ssl_pc, ssl_fg, ssl_ep, rnd_pc, rnd_fg, rnd_ep, n, head):
    lines = [
        f"### Per-organ Dice at N={n}, {head} head (best epoch)",
        "",
        f"SSL v2 mean foreground Dice: **{ssl_fg:.4f}** (ep{ssl_ep:03d})",
        f"Random init mean foreground Dice: **{rnd_fg:.4f}** (ep{rnd_ep:03d})",
        f"Headline gap: **+{(ssl_fg - rnd_fg):.4f} ({100*(ssl_fg/rnd_fg - 1):+.0f}%)**",
        "",
        "| Organ | Random | SSL v2 | Δ | SSL/Random |",
        "|---|---|---|---|---|",
    ]
    for i, organ in enumerate(ORGANS):
        s, r = float(ssl_pc[i+1]), float(rnd_pc[i+1])  # +1 to skip background
        delta = s - r
        winner = "**" if s > r else ""
        lines.append(
            f"| {organ} | {r:.3f} | {winner}{s:.3f}{winner} | "
            f"{delta:+.3f} | {fmt_ratio(s, r)} |"
        )
    ssl_wins = sum(1 for i in range(len(ORGANS)) if ssl_pc[i+1] > rnd_pc[i+1])
    lines.extend(["", f"**SSL wins {ssl_wins} / {len(ORGANS)} organs.**"])
    return "\n".join(lines)


def render_latex(ssl_pc, ssl_fg, ssl_ep, rnd_pc, rnd_fg, rnd_ep, n, head):
    rows = []
    for i, organ in enumerate(ORGANS):
        s, r = float(ssl_pc[i+1]), float(rnd_pc[i+1])
        if s > r:
            rows.append(f"  {organ.replace('_', '\\_')} & {r:.3f} & \\textbf{{{s:.3f}}} & {s-r:+.3f} \\\\")
        else:
            rows.append(f"  {organ.replace('_', '\\_')} & \\textbf{{{r:.3f}}} & {s:.3f} & {s-r:+.3f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Per-organ Dice at N={n} labeled volumes, {head} head, best epoch.
SSL v2 vs random init under the standard linear-probe protocol.
SSL mean fg Dice {ssl_fg:.4f} vs random {rnd_fg:.4f} (+{100*(ssl_fg/rnd_fg-1):.0f}\\%).}}
\\label{{tab:per_organ_n{n}}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Organ & Random & SSL v2 & $\\Delta$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl_run",    default="runs/lin_v2_ssl_n50")
    ap.add_argument("--random_run", default="runs/lin_v2_random_n50")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--head", default="linear")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    ssl_pc, ssl_fg, ssl_ep = best_per_class(Path(args.ssl_run)    / "log.json")
    rnd_pc, rnd_fg, rnd_ep = best_per_class(Path(args.random_run) / "log.json")

    md = render_markdown(ssl_pc, ssl_fg, ssl_ep, rnd_pc, rnd_fg, rnd_ep, args.n, args.head)
    tex = render_latex   (ssl_pc, ssl_fg, ssl_ep, rnd_pc, rnd_fg, rnd_ep, args.n, args.head)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"per_organ_n{args.n}.md").write_text(md + "\n")
    (out / f"per_organ_n{args.n}.tex").write_text(tex)

    print(md)
    print(f"\n[wrote] {out}/per_organ_n{args.n}.md")
    print(f"[wrote] {out}/per_organ_n{args.n}.tex")


if __name__ == "__main__":
    main()
