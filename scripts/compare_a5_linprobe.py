#!/usr/bin/env python3
"""
Head-to-head linear-probe comparison for the a5 (double-inversion) backbone.

a5  = loo_a5_dense_narrow = random-0.75 masking + narrow k=3-7 (BOTH inversions)
A3  = loo_a3_narrow_k     = block-0.4 masking + narrow k=3-7   (gap-only, headline)
v2  = block-0.4 masking + wide k=8-20                          (I-JEPA-inspired)

Reports best val_fg per N for each backbone and the a5-vs-A3 delta. If a5 beats
A3 at most N, the combined recipe wins (additive inversions); if not, A3 stays
the headline and a5 is a clean "combined != additive" result.

Run on the box:  venv/bin/python scripts/compare_a5_linprobe.py
"""
import json
import os

RUNS = os.path.expanduser("~/SSLP/runs")
NS = (5, 10, 20, 50, 100)


def best(name):
    p = os.path.join(RUNS, name, "log.json")
    if not os.path.exists(p):
        return None
    log = json.load(open(p))
    vals = [r["val_fg"] for r in log if "val_fg" in r]
    return max(vals) if vals else None


def fmt(x):
    return f"{x:.4f}" if x is not None else "  --  "


def main():
    hdr = ("N", "v2 SSL", "A3 (gap)", "a5 (both)", "a5-A3", "a5 vs A3")
    print("{:<5}{:<10}{:<10}{:<11}{:<10}{:<10}".format(*hdr))
    print("-" * 56)
    wins = 0
    counted = 0
    for n in NS:
        v2 = best(f"lin_v2_ssl_n{n}")
        a3 = best(f"lin_loo_a3_n{n}")
        a5 = best(f"lin_a5_n{n}")
        d = f"{a5 - a3:+.4f}" if (a5 is not None and a3 is not None) else "  --  "
        pct = f"{100 * (a5 - a3) / a3:+.1f}%" if (a5 is not None and a3 is not None) else "  --  "
        if a5 is not None and a3 is not None:
            counted += 1
            if a5 > a3:
                wins += 1
        print("{:<5}{:<10}{:<10}{:<11}{:<10}{:<10}".format(
            n, fmt(v2), fmt(a3), fmt(a5), d, pct))
    print("-" * 56)
    if counted:
        verdict = ("a5 WINS -> combined inversion is additive; stronger thesis"
                   if wins > counted / 2 else
                   "A3 holds -> combined is NOT additive; A3 stays headline")
        print(f"a5 beats A3 at {wins}/{counted} N values  =>  {verdict}")
    print("\n(a5 = random-0.75 mask + narrow gap; A3 = block mask + narrow gap.)")


if __name__ == "__main__":
    main()
