"""Analysis exactly as pre-registered in prereg_sealed.md (sha256 in prereg_sealed.sha256).
Stage 1 = networks 23-49, stage 2 = networks 50-99, held-out = 3-22 (descriptive).
Primary H1: two1 vs base, one-sided t-test mean d < 0 at alpha 0.05, per stage; plus t-CI, sign test, bootstrap, ratio of sums.
Secondary (stage 1, Holm, two-sided): S1 abl_v29k3t vs base, S2 abl_b1only vs base, S3 two1 vs k3075c09, S4 two1r vs two1.
Descriptive: code-drift check (base_rerun vs base on 3,4); held-out V29-table split 3-7 vs 8-22; residual-time TOST."""
import os, re, glob, numpy as np
from runstats import stats, holm, nets_from, _precise, tq

def load(files):
    res, rt = {}, {}
    for f in files:
        if not os.path.exists(f): continue
        for line in open(f, errors='ignore'):
            m = re.match(r'^(\S+) (\d+) final ([0-9.e+-]+) all \S+ flops \S+ \(\S+\) resid (\d+)ms', line)
            if m:
                res.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
                rt.setdefault(m.group(1), {})[int(m.group(2))] = int(m.group(4))
    for t in res:
        for n in res[t]:
            res[t][n] = _precise(t, n, res[t][n])[0]   # full precision from the saved prediction when it matches
    return res, rt

def line(name, s, extra=''):
    return ('%-28s n=%-3d mean %+6.2f %%  t-CI [%+6.2f, %+6.2f]  boot [%+6.2f, %+6.2f]  improved %d/%d  sign p=%.2g  '
            't one-sided p=%.2g  two-sided p=%.2g %s') % (name, s['n'], 100 * s['mean'], 100 * s['lo'], 100 * s['hi'], 100 * s['blo'],
            100 * s['bhi'], s['improved'], s['n'], s['p_sign'], s['p_one_lt0'], s['p_two'], extra)

def contrast(res, a, b, nets):
    nn = [i for i in nets if i in res.get(a, {}) and i in res.get(b, {})]
    x = np.array([res[a][i] for i in nn]); y = np.array([res[b][i] for i in nn])
    return nn, x, y

if __name__ == '__main__':
    res, rt = load(['./logs/or29_sealed.log', './logs/or29_sealed2.log'])
    print('== H1 (primary): two1 vs base')
    for name, spec in (('stage 1 (23-49)', '23-49'), ('stage 2 (50-99)', '50-99'), ('stages 1+2 (23-99)', '23-99')):
        nn, x, y = contrast(res, 'two1', 'base', nets_from(spec))
        if len(nn) >= 3:
            s = stats(x / y - 1)
            print(line(name, s, ' sum-ratio %+.2f %%  H1 %s' % (100 * (x.sum() / y.sum() - 1), 'CONFIRMED' if s['p_one_lt0'] < 0.05 else 'not confirmed')))
    print('== secondary, stage 1, Holm over four')
    S = [('S1 abl_v29k3t vs base', 'abl_v29k3t', 'base'), ('S2 abl_b1only vs base', 'abl_b1only', 'base'),
         ('S3 two1 vs k3075c09', 'two1', 'k3075c09'), ('S4 two1r vs two1', 'two1r', 'two1')]
    out = []
    for name, a, b in S:
        nn, x, y = contrast(res, a, b, nets_from('23-49'))
        if len(nn) >= 3:
            out.append((name, stats(x / y - 1)))
    if out:
        adj = holm(np.array([s['p_two'] for _, s in out]))
        for (name, s), p in zip(out, adj):
            print(line(name, s, ' Holm p=%.2g' % p))
    print('== amendment 1 (exploratory, outside the Holm family): orientation fix')
    for name, spec in (('H1b stage 1', '23-49'), ('H1b stage 2', '50-99'), ('H1b stages 1+2', '23-99')):
        nn, x, y = contrast(res, 'two1t21', 'base', nets_from(spec))
        if len(nn) >= 3:
            print(line(name + ' two1t21 vs base', stats(x / y - 1)))
    for name, spec in (('S5 stage 1', '23-49'), ('S5 stages 1+2', '23-99')):
        nn, x, y = contrast(res, 'two1t21', 'two1', nets_from(spec))
        if len(nn) >= 3:
            print(line(name + ' two1t21 vs two1', stats(x / y - 1)))
    print('== descriptive: every configuration vs base, stage 1')
    for t in ('two1', 'k3075c09', 'abl_v29k3t', 'abl_b1only', 'two1r'):
        nn, x, y = contrast(res, t, 'base', nets_from('23-49'))
        if len(nn) >= 3:
            print(line(t, stats(x / y - 1)))
    print('== residual time, two1 - base (ms), stages 1+2; TOST margin +-20 ms (5 % of the 400 ms cap; margin chosen post hoc)')
    nn = [i for i in nets_from('23-99') if i in rt.get('two1', {}) and i in rt.get('base', {})]
    if len(nn) >= 3:
        d = np.array([rt['two1'][i] - rt['base'][i] for i in nn], float); se = d.std(ddof=1) / np.sqrt(len(d))
        print('n=%d mean %+.1f ms, 90 %% CI [%+.1f, %+.1f], base mean %.0f ms (this VM is slower than the grader)' % (
            len(d), d.mean(), d.mean() - tq(0.95, len(d) - 1) * se, d.mean() + tq(0.95, len(d) - 1) * se, np.mean([rt['base'][i] for i in nn])))
    dr, _ = load(['./logs/or29_drift.log']); old, _ = load(['./logs/or29_u.log'])
    for i in (3, 4):
        if i in dr.get('base_rerun', {}):
            print('code drift net %d: base now %.4e, before %.4e (%+.3f %%)' % (i, dr['base_rerun'][i], old['base'][i], 100 * (dr['base_rerun'][i] / old['base'][i] - 1)))
