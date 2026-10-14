"""Squared-unit summary of v29cov1s / v29cov1o (one step of V29's covariance program from the exact state, nets 0-2):
share of the exact slice's reduction of the squared next-variance error, f(X) = (e_none^2 - e_X^2) / (e_none^2 - e_exact^2)."""
import re, glob, numpy as np
F = {}
for f in sorted(glob.glob('./logs/v29cov1s_n*.log')):
    for line in open(f):
        m = re.match(r'net (\d+) l=(\d+) next-var rel err \| (.*) \| antisym share of \|K\|\^2 ([0-9.]+)', line)
        if not m: continue
        d = {k.strip(): float(v) for k, v in (p.split(':') for p in m.group(3).split(' | '))}
        d['antisym_share'] = float(m.group(4)); F[(int(m.group(1)), int(m.group(2)))] = d
for f in sorted(glob.glob('./logs/v29cov1o_n*.log')):
    for line in open(f):
        m = re.match(r'net (\d+) l=(\d+) \|.*slot=K \(transposed\): offdiag rel err \S+ \| next-var rel err ([0-9.e+-]+)', line)
        if m and (int(m.group(1)), int(m.group(2))) in F: F[(int(m.group(1)), int(m.group(2)))]['slot=K (wrong)'] = float(m.group(3))
keys = ['sym part', 'antisym part', 'two-term^T', 'two-term sym', 'c*Sigma (LS)', 'slot=K (wrong)']
print('net l | ' + ' | '.join(keys) + ' | antisym share of |K|^2')
fr = {k: [] for k in keys}
for (n, l), d in sorted(F.items()):
    e0, ex = d['none'] ** 2, d['exact K^T'] ** 2
    vals = []
    for k in keys:
        if k in d:
            v = (e0 - d[k] ** 2) / (e0 - ex); fr[k].append(v); vals.append('%+.0f %%' % (100 * v))
        else: vals.append('—')
    print('%d %2d | ' % (n, l) + ' | '.join(vals) + ' | %.2f' % d['antisym_share'])
print('ranges over 3 nets x 3 layers: ' + '; '.join('%s %.0f..%.0f %%' % (k, 100 * min(v), 100 * max(v)) for k, v in fr.items() if v))
