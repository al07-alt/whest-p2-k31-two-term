"""Figures of report v7, from the raw run logs (network-level everywhere)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/research')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from runstats import load, stats
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
R = load()
for n in (0, 1, 2):
    for old, new in (('s2w1', 'wrad'), ('g1.5_all', 's15all')):
        if n in R.get(old, {}): R.setdefault(new, {})[n] = R[old][n]
def d(tag, ref, nets):
    nn = [i for i in nets if i in R.get(tag, {}) and i in R.get(ref, {})]
    return nn, 100 * np.array([R[tag][i] / R[ref][i] - 1 for i in nn])

# Figure 1: anatomy forest plot over networks 0-22
rows = [('exact slice', 's2all'), ('exact slice ×0.75', 's15all'),
        ('radial alone, global LS scale', 's2radall'), ('radial alone, $F_3F_1$-weighted', 'wrad'), ('radial alone, one scale per row', 'rowrad'),
        ('row-wise radial + top-16', 'rowrad16'), ('two-term closure (§4)', 'two1'),
        ('placebo: permuted remainder', 'perm'), ('placebo: permuted within |α| bins (2 seeds)', 'perms'),
        ('placebo: another network\'s remainder', 'donor')]
for n in range(23):
    v = [R[t][n] for t in ('perms1', 'perms2') if n in R.get(t, {})]
    if v: R.setdefault('perms', {})[n] = float(np.mean(v))
f, ax = plt.subplots(figsize=(5.6, 3.3)); y = 0; yt = []; yl = []
for name, tag in rows:
    nn, x = d(tag, 'base', range(23))
    if len(nn) < 3: continue
    s = stats(x / 100)
    ax.scatter(x, np.full(len(x), -y) + np.random.default_rng(y).uniform(-.18, .18, len(x)), s=5, c='#9aa5b1', zorder=1)
    ax.errorbar(100 * s['mean'], -y, xerr=[[100 * (s['mean'] - s['lo'])], [100 * (s['hi'] - s['mean'])]], fmt='o', c='#c44536' if s['mean'] > 0 else '#2a6f97', ms=4, capsize=2, zorder=2)
    yt.append(-y); yl.append('%s  (n=%d)' % (name, len(nn))); y += 1
ax.axvline(0, c='k', lw=.6); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5); ax.set_xlabel('final-MSE change against V29 (%)')
ax.set_xlim(-40, 60)
f.tight_layout(); f.savefig('fig/anatomy_forest.pdf'); f.savefig('fig/anatomy_forest.png', dpi=220)

# Figure 3: closure per network on every set
f, ax = plt.subplots(figsize=(6.4, 2.5)); x0 = 0; ticks = []
for spec, nets, col in (('held out 3–22', range(3, 23), '#6a994e'), ('stage 1: 23–49', range(23, 50), '#2a6f97'), ('stage 2: 50–99', range(50, 100), '#1d3557')):
    nn, x = d('two1', 'base', nets)
    if not nn: continue
    ax.bar(np.arange(x0, x0 + len(nn)), x, color=[col if v < 0 else '#c44536' for v in x], width=.85)
    s = stats(x / 100); ax.hlines(100 * s['mean'], x0 - .5, x0 + len(nn) - .5, colors='k', linestyles='--', lw=.8)
    ax.text(x0 + len(nn) / 2, 2.2, '%s\nmean %.1f %% (%d/%d)' % (spec, 100 * s['mean'], s['improved'], s['n']), ha='center', fontsize=7)
    x0 += len(nn) + 3
ax.axhline(0, c='k', lw=.6); ax.set_xticks([]); ax.set_ylabel('final-MSE change (%)'); ax.set_ylim(-9, 4)
f.tight_layout(); f.savefig('fig/two_term_sets.pdf'); f.savefig('fig/two_term_sets.png', dpi=220)
print('ok')

# Figure 2: row-wise radial scale by |alpha| bin at l = 14, 23 networks (law23.py output)
z = np.load('/home/claude/data/law23_0_22.npz'); u = np.array([z[f'u_{i}'] for i in range(23)])
f, ax = plt.subplots(figsize=(3.6, 2.4)); xs = np.arange(5)
for row in u: ax.plot(xs, row, c='#9aa5b1', lw=.6, alpha=.6)
ax.plot(xs, np.median(u, 0), 'o-', c='#2a6f97', lw=1.6, ms=4, label='median of 23 networks')
ax.axhline(1, c='k', lw=.6, ls=':'); ax.set_xticks(xs); ax.set_xticklabels(['<0.75', '0.75–1.5', '1.5–2.5', '2.5–4', '>4'], fontsize=7.5)
ax.set_xlabel('|α| of the tripled neuron'); ax.set_ylabel('row radial scale / global'); ax.legend(frameon=False, fontsize=7)
f.tight_layout(); f.savefig('fig/k31_ushape.pdf'); f.savefig('fig/k31_ushape.png', dpi=220)
print('ushape ok')
