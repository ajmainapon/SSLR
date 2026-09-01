"""Supplement: matched 7-organ comparison, TotalSegmentator vs AMOS-CT.

TotalSeg probes are 11-class (bg + 10 organs). AMOS probes are 8-class
(bg + 7 organs). To compare like with like we recompute the TotalSeg mean
over ONLY the 7 organs AMOS contains:
    TotalSeg class idx 1,2,3,4,5,6,10
  = liver, spleen, kidney_L, kidney_R, stomach, pancreas, aorta
Epoch selection mirrors the AMOS protocol: best epoch by the 7-organ mean.
"""
import ast, os, re, statistics as st
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("SSLR_ROOT", Path.home() / "SSLP"))
RUNS = PROJECT_ROOT / "runs"
TS_IDX = [1, 2, 3, 4, 5, 6, 10]      # 7 overlap organs in TotalSeg ordering
AMOS_IDX = [1, 2, 3, 4, 5, 6, 7]     # all 7 organs in AMOS ordering
LINE = re.compile(r"val_per=(\[[^\]]*\])")

def best7(log, idx):
    if not log.exists():
        return None
    best = None
    for line in log.read_text(errors='ignore').splitlines():
        m = LINE.search(line)
        if not m:
            continue
        try:
            vals = [float(x) for x in ast.literal_eval(m.group(1))]
        except Exception:
            continue
        if max(idx) >= len(vals):
            continue
        mean7 = sum(vals[i] for i in idx) / len(idx)
        if best is None or mean7 > best:
            best = mean7
    return best

def collect(prefix, N, idx):
    out = []
    for s in range(4):
        name = f'{prefix}_n{N}' if s == 0 else f'{prefix}_n{N}_seed{s}'
        v = best7(RUNS / f'{name}.log', idx)
        if v is not None:
            out.append(v)
    return out

print(f"{'dataset':<12}{'init':<8}{'N':>5}{'seeds':>7}{'mean7':>9}{'std':>9}")
res = {}
for label, (ssl_p, rnd_p, idx) in {
    'TotalSeg': ('lin_loo_a3', 'lin_v2_random', TS_IDX),
    'AMOS-CT':  ('lin_amos_a3', 'lin_amos_random', AMOS_IDX),
}.items():
    for init, pre in (('A2 SSL', ssl_p), ('Random', rnd_p)):
        for N in (20, 50, 100):
            v = collect(pre, N, idx)
            if not v:
                print(f'{label:<12}{init:<8}{N:>5}{0:>7}{"--":>9}')
                continue
            m = st.mean(v); s = st.stdev(v) if len(v) > 1 else 0.0
            res[(label, init, N)] = (m, s, len(v))
            print(f'{label:<12}{init:<8}{N:>5}{len(v):>7}{m:>9.4f}{s:>9.4f}')

print()
print(f"{'dataset':<12}{'N':>5}{'SSL':>9}{'Random':>9}{'delta rel':>11}")
for label in ('TotalSeg', 'AMOS-CT'):
    for N in (20, 50, 100):
        a = res.get((label, 'A2 SSL', N)); b = res.get((label, 'Random', N))
        if a and b:
            print(f'{label:<12}{N:>5}{a[0]:>9.4f}{b[0]:>9.4f}{100*(a[0]/b[0]-1):>10.1f}%')
