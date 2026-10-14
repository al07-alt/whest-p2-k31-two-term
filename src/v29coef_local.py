"""V29's adapted (3,1) coefficient (printed 'lam' in the EF_K31OR_PRINT runs, runlogs/s15all_<net>.log; the slot is
lam * C_off) against the coefficient of V29's form that is optimal for one step of the covariance update from the
exact state (v29cov1f_n<net>.log, 'V29 form cS ... b=[c]'), nets 3-5, l = 5, 10, 14."""
import re
for n in (3, 4, 5):
    loc = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'net \d+ l=(\d+) \|.*?V29 form cS \d+% b=\[\s*([0-9.]+)', open(f'./logs/v29cov1f_n{n}.log').read())}
    lam = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'\[EF\] li (\d+) .*?lam ([0-9.e+-]+)', open(f'./logs/runlogs/s15all_{n}.log').read())}
    print('net %d: ' % n + ' | '.join('l=%d V29 %.4f local-opt %.4f ratio %.2f' % (l, lam[l], loc[l], lam[l] / loc[l]) for l in (5, 10, 14)))
