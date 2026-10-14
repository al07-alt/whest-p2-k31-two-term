"""Two-term law and row-scale U-shape of the true (3,1) slice on many public nets (dense N=1e8 dumps).
Per net and layer l=1..14: LS fit of K_ab (off-diagonal, K[a,b]=kappa(a,a,a,b)) on [v_a S_ab, kappa3_a T_aab / v_a];
radial-only R2; two-term R2; b1 refit with b2 fixed at 1. At l=14: radial row scale by |alpha| bin / global.
Output: data/law23.npz and a printed summary."""
import numpy as np, sys
from truek31 import k31, t21
nets = [int(x) for x in sys.argv[1].split(',')]
bins = np.array([0, 0.75, 1.5, 2.5, 4, 99])
res = {}
for i in nets:
    K = np.load(f'./data/kp/marg_{i:03d}.npz')
    rows = []
    for l in range(1, 15):
        Kt = k31(i, l); T, k3, v = t21(i, l)
        M11 = np.load(f'./data/kp/pre_M11_{i:03d}.npy', mmap_mode='r')[l].astype(float)
        m = K['pre_mean'][l].astype(float); S = M11 - np.outer(m, m)
        off = ~np.eye(len(m), dtype=bool)
        R = (v[:, None] * S)[off]; Q = ((k3 / v)[:, None] * T)[off]; y = Kt[off]
        X = np.stack([R, Q], 1); b = np.linalg.lstsq(X, y, rcond=None)[0]
        r2two = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
        br = (R @ y) / (R @ R); r2rad = 1 - np.sum((y - br * R) ** 2) / np.sum((y - y.mean()) ** 2)
        b1fix = (R @ (y - Q)) / (R @ R)
        u = []
        if l == 14:
            a = np.abs(m / np.sqrt(v)); Sz = S.copy(); np.fill_diagonal(Sz, 0); Kd = Kt.copy(); np.fill_diagonal(Kd, 0)
            RR = v[:, None] * Sz; g = (Kd * RR).sum() / (RR * RR).sum()
            for lo, hi in zip(bins[:-1], bins[1:]):
                s_ = (a >= lo) & (a < hi); u.append((Kd[s_] * RR[s_]).sum() / (RR[s_] ** 2).sum() / g)
        rows.append((b[0], b[1], r2rad, r2two, b1fix, br))
        if l == 14: res[f'u_{i}'] = np.array(u)
    res[f'c_{i}'] = np.array(rows)
    c = np.array(rows)
    print(i, 'b2(l>=4) %.3f-%.3f  R2 rad %.2f->two %.2f (l=14)  U(|a|<.75)/global %.2f' % (c[3:, 1].min(), c[3:, 1].max(), c[-1, 2], c[-1, 3], res[f'u_{i}'][0]), flush=True)
np.savez(f'./data/law23_{nets[0]}_{nets[-1]}.npz', **res)
