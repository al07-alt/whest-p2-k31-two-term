"""Audit point 15: the one-step comparison of T-PCP's next-layer variances in squared units.
For each net and layer: relative rms errors of the predicted next-layer variances for base / exact slice (edge+k31x) /
radial slice (edge+k31rad), against the same-sample truth (no sampling noise) and against the independent bake.
Fraction of the exact slice's improvement achieved by the radial slice, in squared units:
  f = (e_base^2 - e_rad^2) / (e_base^2 - e_exact^2).
Against the same-sample truth no floor needs subtracting; against the bake, an independent noise floor adds the same
constant to every e^2 and cancels in f as well, so the two columns differ only through sampling cross-terms."""
import re, numpy as np
print('net layer | e_exact/floor (bake) | f same-sample (sq) | f bake (sq) | f same-sample (rms, old convention)')
rows = []
for n in (0, 1, 2):
    txt = open(f'./logs/onestep10_n{n}.log').read().split('\n')
    cur = None; E = {}
    for line in txt:
        m = re.match(r'^(\d+) same-sample vs independent next-var truth: rel diff ([0-9.e+-]+)', line)
        if m: cur = int(m.group(1)); E[cur] = {'floor': float(m.group(2))}; continue
        m = re.match(r'^\s+(\S+)\s+err vs same-sample ([0-9.e+-]+) \| vs independent ([0-9.e+-]+)', line)
        if m and cur is not None: E[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)))
    for l in sorted(E):
        b, x, r, fl = E[l]['base'], E[l]['edge+k31x'], E[l]['edge+k31rad'], E[l]['floor']
        fs = (b[0] ** 2 - r[0] ** 2) / (b[0] ** 2 - x[0] ** 2); fb = (b[1] ** 2 - r[1] ** 2) / (b[1] ** 2 - x[1] ** 2)
        frms = (b[0] - r[0]) / (b[0] - x[0])
        rows.append((n, l, x[1] / fl, fs, fb, frms))
        print('%d %2d | %.2f | %.0f %% | %.0f %% | %.0f %%' % (n, l, x[1] / fl, 100 * fs, 100 * fb, 100 * frms))
a = np.array(rows)
for l in sorted(set(a[:, 1])):
    s = a[a[:, 1] == l]
    print('layer %2d: exact/floor %.2f-%.2f | radial share of the exact improvement, squared units: same-sample %.0f-%.0f %%, bake %.0f-%.0f %%' % (
        l, s[:, 2].min(), s[:, 2].max(), 100 * s[:, 3].min(), 100 * s[:, 3].max(), 100 * s[:, 4].min(), 100 * s[:, 4].max()))
