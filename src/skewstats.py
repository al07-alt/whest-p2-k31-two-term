"""Audit point 13: error bars for the fixed-input skewness identity.
(i) seed-to-seed spread (seeds 5, 6, 7; N = 4e4 forward passes each) of slope and R2 on nets 0-2;
(ii) network spread over nets 0-9 (seed 5);
(iii) out-of-sample comparison with a fitted gamma = c_l * alpha table (c_l fitted on the other nine networks)."""
import numpy as np
D = './data/skewid_%03d%s.npz'
def m(i, s=''): return np.load(D % (i, s))
def sl_r2(g, gp): return np.sum(g * gp) / np.sum(gp * gp), 1 - np.mean((g - gp) ** 2) / np.var(g)
L = [4, 8, 12, 14]
print('(i) seed spread, nets 0-2: slope and R2 per seed (5/6/7)')
for l in L:
    out = []
    for i in (0, 1, 2):
        v = [sl_r2(m(i, s)[f'g_{l}'], m(i, s)[f'gp_{l}']) for s in ('', '_s6', '_s7')]
        sl = np.array([a for a, _ in v]); r2 = np.array([b for _, b in v])
        out.append('net %d slope %.3f±%.3f R2 %.3f±%.3f' % (i, sl.mean(), sl.std(ddof=1), r2.mean(), r2.std(ddof=1)))
    print('l=%d  ' % l + ' | '.join(out))
print('(ii) network spread, nets 0-9 (seed 5) and (iii) leave-one-network-out gamma = c_l alpha')
nets = range(10)
for l in L:
    S = np.array([sl_r2(m(i)[f'g_{l}'], m(i)[f'gp_{l}']) for i in nets])
    g = {i: m(i)[f'g_{l}'] for i in nets}; a = {i: m(i)[f'a_{l}'] for i in nets}
    loo = []; corr_id = []; corr_a = []
    for i in nets:
        oth = [j for j in nets if j != i]
        c = sum((g[j] * a[j]).sum() for j in oth) / sum((a[j] ** 2).sum() for j in oth)
        loo.append(1 - np.mean((g[i] - c * a[i]) ** 2) / np.var(g[i]))
        corr_id.append(np.corrcoef(g[i], m(i)[f'gp_{l}'])[0, 1]); corr_a.append(np.corrcoef(g[i], a[i])[0, 1])
    loo = np.array(loo)
    print('l=%2d identity: slope %.3f [%.3f-%.3f] R2 %.3f [%.3f-%.3f] | LOO c*alpha R2 %.3f [%.3f-%.3f] | identity minus LOO R2: mean %+.3f, identity better on %d/10 | corr(identity) - corr(alpha): %+.4f' % (
        l, S[:, 0].mean(), S[:, 0].min(), S[:, 0].max(), S[:, 1].mean(), S[:, 1].min(), S[:, 1].max(), loo.mean(), loo.min(), loo.max(),
        (S[:, 1] - loo).mean(), int((S[:, 1] > loo).sum()), np.mean(np.array(corr_id) - np.array(corr_a))))
