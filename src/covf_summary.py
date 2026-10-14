"""Ranges of the local shares from v29cov1f (fixed two-term law; V29's form at its best local scale; two-term law and
all 17 features at in-sample optimal coefficients), separately for nets 0-2 (where the b1 table was fitted) and 3-5."""
import re, glob, numpy as np
keys = ['fixed two-term', 'V29 form cS', 'rad+lat', 'all']
for grp, nets in (('nets 0-2 (b1 fitted here)', (0, 1, 2)), ('nets 3-5 (out of sample)', (3, 4, 5))):
    v = {k: [] for k in keys}; gap = []; b2 = []
    for n in nets:
        for line in open(f'./logs/v29cov1f_n{n}.log'):
            d = dict((m.group(1).strip(), int(m.group(2))) for m in re.finditer(r'(?:\| )([^|%]+?) (\d+)%', line))
            for k in keys: v[k].append(d[k])
            gap.append(d['rad+lat'] - d['fixed two-term'])
            m = re.search(r'rad\+lat \d+% b=\[\s*[0-9.]+\s*,\s*([0-9.]+)', line); b2.append(float(m.group(1)))
    print(grp + ': ' + '; '.join('%s %d-%d %%' % (k, min(x), max(x)) for k, x in v.items()) +
          '; fixed law below its in-sample optimum by %d-%d points; locally preferred b2 %.2f-%.2f' % (min(gap), max(gap), min(b2), max(b2)))
