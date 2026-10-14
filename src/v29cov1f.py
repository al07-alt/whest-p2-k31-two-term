"""Local feature study for the (3,1) slot in V29's covariance program (exact pre-activation state of one layer).
The predicted next-layer variances are linear in the slot, so for a set of candidate features f_k (each in
K-orientation, K[a,b] = kappa(a,a,a,b)) we can find the coefficients that minimise the squared relative next-variance
error against the same-sample truth, and report the share of the exact slice's reduction that the feature set
reaches. Shares are in-sample (coefficients fitted per network and layer), so they are upper bounds for a
fixed-coefficient closure; 'fixed' rows use the closure's table coefficients instead.
usage: v29cov1f.py <net> <layers>"""
import numpy as np, sys, ast
from truek31 import k31
from lib import W
from scipy.special import ndtr
src = open('./v29_experiments/estimator_v29_ef.py').read()
def grab(name):
    i = src.index(name + ' = '); j = i + len(name) + 3; depth = 0; k = j
    while True:
        c = src[k]
        if c in '[{(': depth += 1
        elif c in ']})': depth -= 1
        k += 1
        if depth == 0 and c in ']})': break
    return ast.literal_eval(src[j:k])
WP = grab('WICK_PAIRS'); WU = grab('WICK_UNIFIED'); TS = grab('TERM_SPECS'); I = WP.index
pdf = lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
B1 = [0.00974, 0.01424, 0.01752, 0.02066, 0.02259, 0.0244, 0.02539, 0.02668, 0.02694, 0.02645, 0.02603, 0.02589, 0.02442, 0.02452]
i = int(sys.argv[1]); Ls = [int(x) for x in sys.argv[2].split(',')]
Kz = np.load(f'./data/kp/marg_{i:03d}.npz'); Wt = W(i)
M11p = np.load(f'./data/kp/M11_{i:03d}.npy', mmap_mode='r'); M11 = np.load(f'./data/kp/pre_M11_{i:03d}.npy', mmap_mode='r')
M21 = np.load(f'./data/kp/pre_M21_{i:03d}.npy', mmap_mode='r'); M22 = np.load(f'./data/kp/pre_M22_{i:03d}.npy', mmap_mode='r')
for l in Ls:
    m = Kz['pre_mean'][l].astype(float); X2 = Kz['pre_m2'][l]; X3 = Kz['pre_m3'][l]; X4 = Kz['pre_m4'][l]
    A = m[:, None]; B = m[None, :]; S = M11[l].astype(float) - A * B; v = np.diag(S).copy(); s = np.sqrt(v); a = m / s
    m21 = M21[l].astype(float); m22 = M22[l].astype(float); m11 = M11[l].astype(float)
    T = m21 - B * X2[:, None] - 2 * A * m11 + 2 * A * A * B
    c22 = m22 - 2 * B * m21 - 2 * A * m21.T + B * B * X2[:, None] + A * A * X2[None, :] + 4 * A * B * m11 - 3 * A * A * B * B
    Q = c22 - np.outer(v, v) - 2 * S * S
    k3 = X3 - 3 * m * X2 + 2 * m ** 3; k4 = X4 - 4 * m * X3 + 6 * m * m * X2 - 3 * m ** 4 - 3 * v * v
    K = k31(i, l)
    ph = pdf(a); Ph = ndtr(a)
    Wv = np.array([const * s ** e * (sum(c * a ** k for k, c in enumerate(p1)) * ph + sum(c * a ** k for k, c in enumerate(p2)) * Ph) for (const, e, p1, p2) in WU])
    off = ~np.eye(len(m), dtype=bool)
    Coff = S.copy(); np.fill_diagonal(Coff, 0); Qo = Q.copy(); np.fill_diagonal(Qo, 0)
    base = {'ones2': np.ones_like(S), 'c_off': Coff, 'd21': T, 'd21T': T.T, 'd3col': np.repeat(k3[:, None], len(m), 1), 'd3row': np.repeat(k3[None, :], len(m), 0),
            'g4col': np.repeat(k4[:, None], len(m), 1), 'g4row': np.repeat(k4[None, :], len(m), 0), 'wk4m': Qo}
    mp = Kz['mean'][l].astype(float); Ct = M11p[l].astype(float) - np.outer(mp, mp)
    w = Wt[l + 1]; vt = np.einsum('ji,jk,ki->i', w, Ct, w)
    # slot-free prediction
    PK = np.zeros_like(S)
    for (x, y, wl, wr, coef) in TS[(1, 1)]:
        Ax = base[x] * base[y] if (x != 'ones2' and y != 'ones2') else (base[y] if x == 'ones2' else base[x])
        PK += coef * Ax * Wv[wl][:, None] * Wv[wr][None, :]
    pk = 0.5 * (PK + PK.T); np.fill_diagonal(pk, np.diag(Ct)); v0 = np.einsum('ji,jk,ki->i', w, pk, w)
    G = Wv[I((1, 1))][:, None] * Wv[I((3, 1))][None, :] / 3.0
    def contrib(F):                     # next-variance contribution of slot = F^T (F in K-orientation)
        P = F.T * G; P = 0.5 * (P + P.T); np.fill_diagonal(P, 0.0)
        return np.einsum('ji,jk,ki->i', w, P, w) / vt
    r = (vt - v0) / vt
    cK = contrib(K); e0 = r @ r; ex = (r - cK) @ (r - cK)
    rad = v[:, None] * S; lat = (k3 / v)[:, None] * T
    absa = np.abs(a); bins = [(0, .75), (.75, 1.5), (1.5, 2.5), (2.5, 4), (4, 1e9)]
    radbins = [rad * ((absa >= lo) & (absa < hi))[:, None] for lo, hi in bins]
    latbins = [lat * ((absa >= lo) & (absa < hi))[:, None] for lo, hi in bins]
    feats = {'rad': rad, 'lat': lat, 'cS': S.copy(), 'm2S': (m * m)[:, None] * S, 'vmm': (v * m)[:, None] * m[None, :], 'm3m': (m ** 3)[:, None] * m[None, :],
             'k4S': (k4 / v)[:, None] * S, 'radT': rad.T, 'latT': lat.T, 'Tsym': (k3 / v)[None, :] * T.T}
    C = {k: contrib(f) for k, f in feats.items()}
    CB = [contrib(f) for f in radbins]; CL = [contrib(f) for f in latbins]
    def share(cols):
        X = np.stack(cols, 1); b = np.linalg.lstsq(X, r, rcond=None)[0]; e = r - X @ b
        return (e0 - e @ e) / (e0 - ex), b
    out = []
    fixed = B1[l - 1] * C['rad'] + C['lat']; ef = (r - fixed) @ (r - fixed)
    out.append('fixed two-term %.0f%%' % (100 * (e0 - ef) / (e0 - ex)))
    for name, cols in [('V29 form cS', [C['cS']]), ('rad', [C['rad']]), ('rad+lat', [C['rad'], C['lat']]),
                       ('rad+lat+means', [C['rad'], C['lat'], C['m2S'], C['vmm'], C['m3m']]), ('rad+lat+k4S', [C['rad'], C['lat'], C['k4S']]),
                       ('rad bins+lat', CB + [C['lat']]), ('rad bins+lat bins', CB + CL),
                       ('rad+lat+transposes', [C['rad'], C['lat'], C['radT'], C['latT']]),
                       ('all', CB + CL + [C['m2S'], C['vmm'], C['m3m'], C['k4S'], C['radT'], C['latT'], C['cS']])]:
        sh, b = share(cols)
        out.append('%s %.0f%%' % (name, 100 * sh) + (' b=%s' % np.array2string(b, precision=4, separator=',') if len(cols) <= 3 else ''))
    print('net %d l=%d | ' % (i, l) + ' | '.join(out), flush=True)
