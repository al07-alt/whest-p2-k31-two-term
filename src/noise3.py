"""Three-way noise decomposition of the final-layer mean (audit point 12).
Three independent Monte-Carlo estimates of E h_15 for networks 0-5: the organisers' truth (truth.npz), Keenan Pepper's
N=1e8 dumps (kp/marg 'mean'), and the independently seeded N=1e8 bake (bk/marg 'post_mean').
Pairwise MSE = s_a^2 + s_b^2 for independent errors, so the three variances follow from the three pairwise MSEs."""
import numpy as np
T = np.load('./data/truth.npz'); ids = list(T['ids'])
rows = []
for i in range(6):
    t = T['means'][ids.index(i)][-1].astype(float)
    d = np.load(f'./data/kp/marg_{i:03d}.npz')['mean'][-1].astype(float)
    b = np.load(f'./data/bk/marg_{i:03d}.npz')['post_mean'][-1].astype(float)
    td, tb, db = np.mean((t - d) ** 2), np.mean((t - b) ** 2), np.mean((d - b) ** 2)
    sT, sD, sB = (td + tb - db) / 2, (td + db - tb) / 2, (tb + db - td) / 2
    rows.append((td, tb, db, sT, sD, sB))
    print('net %d  MSE(T,D) %.2e  MSE(T,B) %.2e  MSE(D,B) %.2e  ->  truth %.2e  dumps %.2e  bake %.2e' % (i, td, tb, db, sT, sD, sB))
r = np.array(rows).mean(0)
print('mean   MSE(T,D) %.2e  MSE(T,B) %.2e  MSE(D,B) %.2e  ->  truth %.2e  dumps %.2e  bake %.2e' % tuple(r))
print('organisers\' truth noise as a fraction of V29 raw MSE (2.1e-8): %.2f %%' % (100 * r[3] / 2.1e-8))
