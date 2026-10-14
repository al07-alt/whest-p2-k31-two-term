"""Audit point 9: V29's (3,1) coefficient relative to the least-squares coefficient of its own form (c * Sigma_ij),
on every network with an EF_K31OR_PRINT log (runlogs/s15all_<net>.log, oracle scale 1.5 => slot = 0.75 K^T).
slope(V29 slot on K^T) = 0.75 * printed slope; ratio = slope / R2(c*C form on the exact slice) (see k31v29ratio.py)."""
import glob, re, numpy as np
from truek31 import k31
rows = {}
for f in sorted(glob.glob('./logs/runlogs/s15all_*.log')):
    i = int(re.search(r'_(\d+)\.log', f).group(1))
    sl = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'\[EF\] li (\d+) \|wk431\| \S+ \|0\.5K\^T\| \S+ slope\(wk on 0\.5K\^T\) ([-0-9.]+)', open(f).read())}
    if not sl: continue
    M = np.load(f'./data/kp/marg_{i:03d}.npz'); out = []
    for l in (4, 8, 11, 14):
        if l not in sl: continue
        Kt = k31(i, l); M11 = np.load(f'./data/kp/pre_M11_{i:03d}.npy', mmap_mode='r')[l].astype(float)
        m = M['pre_mean'][l].astype(float); S = M11 - np.outer(m, m); off = ~np.eye(1024, dtype=bool)
        y = Kt.T[off]; x = S[off]; b = (x @ y) / (x @ x); R2 = 1 - np.sum((y - b * x) ** 2) / np.sum((y - y.mean()) ** 2)
        s = 0.75 * sl[l]; out.append((l, s, R2, s / R2))
    rows[i] = out
    print(i, ' '.join('l=%d slope %.3f ratio %.2f' % (l, s, r) for l, s, _, r in out), flush=True)
if rows:
    for k, l in enumerate((4, 8, 11, 14)):
        v = np.array([[o[1], o[3]] for r in rows.values() for o in r if o[0] == l])
        if len(v): print('l=%2d over %d nets: V29 slope on K %.3f-%.3f, V29 coef / LS coef %.2f-%.2f (median %.2f)' % (
            l, len(v), v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), np.median(v[:, 1])))
