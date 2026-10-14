"""Out-of-sample R2 of the two-term law with FIXED coefficients (audit point 14).
For networks given in argv[1], layers 1..14: R2 of K_ab (off-diagonal, true N=1e8 slice) predicted by
b1_l v_a S_ab + b2 kappa3_a T_aab / v_a with (i) the submitted b1 table (joint fit, nets 0-2) and b2 = 1,
(ii) the refit table and b2 = 1, (iii) per-net LS two-term, (iv) per-net LS radial only.
R2 = 1 - SSE / sum((y - mean y)^2), the same convention as law23.py. Output: data/law_oos.npz and a summary."""
import numpy as np, sys
from truek31 import k31, t21
JOINT = [0.00974, 0.01424, 0.01752, 0.02066, 0.02259, 0.0244, 0.02539, 0.02668, 0.02694, 0.02645, 0.02603, 0.02589, 0.02442, 0.02452]
REFIT = [0.0098, 0.0144, 0.0178, 0.02108, 0.02331, 0.02553, 0.02694, 0.02865, 0.02959, 0.02992, 0.03093, 0.03199, 0.03212, 0.03281]
nets = [int(x) for x in sys.argv[1].split(',')]
out = np.zeros((len(nets), 14, 4))
for n, i in enumerate(nets):
    for l in range(1, 15):
        Kt = k31(i, l); T, k3, v = t21(i, l)
        M11 = np.load(f'./data/kp/pre_M11_{i:03d}.npy', mmap_mode='r')[l].astype(float)
        m = np.load(f'./data/kp/marg_{i:03d}.npz')['pre_mean'][l].astype(float); S = M11 - np.outer(m, m)
        off = ~np.eye(len(m), dtype=bool)
        R = (v[:, None] * S)[off]; Q = ((k3 / v)[:, None] * T)[off]; y = Kt[off]; sst = np.sum((y - y.mean()) ** 2)
        r2 = lambda p: 1 - np.sum((y - p) ** 2) / sst
        X = np.stack([R, Q], 1); b = np.linalg.lstsq(X, y, rcond=None)[0]
        out[n, l - 1] = [r2(JOINT[l - 1] * R + Q), r2(REFIT[l - 1] * R + Q), r2(X @ b), r2((R @ y) / (R @ R) * R)]
    print(i, 'l=14 R2 fixed-joint %.3f fixed-refit %.3f LS-two %.3f LS-rad %.3f' % tuple(out[n, -1]), flush=True)
np.savez('./data/law_oos.npz', nets=np.array(nets), r2=out)
q = np.percentile(out, [0, 50, 100], axis=0)
print('layer | fixed-joint min/med/max | fixed-refit | LS-two | LS-rad (median)')
for l in range(14):
    print(l + 1, '| %.2f/%.2f/%.2f | %.2f/%.2f/%.2f | %.2f | %.2f' % (q[0, l, 0], q[1, l, 0], q[2, l, 0], q[0, l, 1], q[1, l, 1], q[2, l, 1], q[1, l, 2], q[1, l, 3]))
