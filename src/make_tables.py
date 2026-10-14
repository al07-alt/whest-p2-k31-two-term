"""Every number table of report v7, computed from the raw run logs (writes ../writeup/tables_v7.md).
Unit of inference: the network. d_i = MSE_final(config)/MSE_final(ref) - 1; mean, 95 % t-CI over networks,
number improved; fractions of a gain are ratios of mean gains with a 10 000-resample network bootstrap CI."""
import json, os, numpy as np
from runstats import load, stats, holm, ratio_of_means, nets_from

R = load()
ALIAS = {('s2w1', n): 'wrad' for n in (0, 1, 2)}
ALIAS.update({('g1.5_all', n): 's15all' for n in (0, 1, 2)})
for (old, n), new in ALIAS.items():
    if n in R.get(old, {}):
        R.setdefault(new, {})[n] = R[old][n]
OUT = []
def P(s=''): OUT.append(s)
def fmt(s): return '%+.1f [%+.1f, %+.1f]' % (100 * s['mean'], 100 * s['lo'], 100 * s['hi'])
def d(tag, ref, nets):
    nn = [i for i in nets if i in R.get(tag, {}) and i in R.get(ref, {})]
    return nn, np.array([R[tag][i] / R[ref][i] - 1 for i in nn])
def cell(tag, ref, nets, need=2):
    nn, x = d(tag, ref, nets)
    if len(nn) < need: return '—', None
    if len(nn) < 3: return ' / '.join('%+.1f' % (100 * v) for v in x), None
    s = stats(x); return '%s, %d/%d' % (fmt(s), s['improved'], s['n']), s

DEV, HO, S1, S2 = [0, 1, 2], list(range(3, 23)), list(range(23, 50)), list(range(50, 100))
ALL23 = DEV + HO
KEY = {}

# ---------------- A. anatomy -----------------
P('### Table A (anatomy). Change in final MSE against V29, % — mean [95 % t-CI over networks], networks improved')
P()
P('| replacement of V29\'s (3,1) term, both uses | networks 0–2 | networks 3–22 | all |')
P('|---|---|---|---|')
rows = [('exact slice', 's2all'), ('exact slice at scale 0.75', 's15all'),
        ('exact non-radial part + radial part rebuilt on V29\'s Σ', 's2nonradall'),
        ('radial part alone, global LS scale', 's2radall'), ('radial part alone, $F_3F_1$-weighted scale', 'wrad'),
        ('radial part alone, one LS scale per row', 'rowrad'),
        ('row-wise radial + top-4 of its remainder', 'rowrad4'), ('row-wise radial + top-16 of its remainder', 'rowrad16'),
        ('row-wise radial + top-64 of its remainder', 'rowrad64'),
        ('placebo: remainder with labels permuted', 'perm'), ('placebo: remainder permuted within $|\\alpha|$ bins, seed 1', 'perms1'),
        ('placebo: remainder permuted within $|\\alpha|$ bins, seed 2', 'perms2'),
        ('placebo: another network\'s remainder (same layer)', 'donor'), ('placebo: another network\'s whole slice', 'xnet'),
        ('no (3,1) term at all', 'wk0')]
for name, tag in rows:
    c1, _ = cell(tag, 'base', DEV); c2, s2 = cell(tag, 'base', HO); c3, s3 = cell(tag, 'base', ALL23)
    if c1 == '—' and c2 == '—': continue
    P('| %s | %s | %s | %s |' % (name, c1, c2, c3))
    if s3 is not None: KEY['anat_' + tag] = s3
P()
# fractions of the exact-slice gain (ratio of mean gains)
P('**Fractions of the exact-slice gain** (ratio of mean gains over networks 3–22, network bootstrap 95 % CI):')
P()
for name, tag in [('two-term closure', 'two1'), ('radial part alone (global)', 's2radall'), ('row-wise radial alone', 'rowrad'),
                  ('row-wise radial + top-4', 'rowrad4'), ('row-wise radial + top-16', 'rowrad16'), ('row-wise radial + top-64', 'rowrad64'), ('exact non-radial + radial on V29 Σ', 's2nonradall'), ('exact slice at 0.75', 's15all')]:
    nn = [i for i in HO if i in R.get(tag, {}) and i in R.get('s2all', {}) and i in R.get('base', {})]
    if len(nn) < 3: continue
    num = np.array([1 - R[tag][i] / R['base'][i] for i in nn]); den = np.array([1 - R['s2all'][i] / R['base'][i] for i in nn])
    r, lo, hi = ratio_of_means(num, den); KEY['frac_' + tag] = (r, lo, hi, len(nn))
    P('* %s: %.0f %% [%.0f, %.0f] (n = %d)' % (name, 100 * r, 100 * lo, 100 * hi, len(nn)))
P()

# ---------------- B. selection history on the development networks -----------------
P('### Table B (every non-oracle closure variant run on the development networks 0–2 before the choice, by mean change)')
P()
P('| variant | tag | nets 0 / 1 / 2 (%) | mean | improved |')
P('|' + '-' * 46 + '|' + '-' * 13 + '|' + '-' * 22 + '|' + '-' * 7 + '|' + '-' * 9 + '|')
VAR = [('two1', 'two-term, $b_2=1$, joint $b_1$ table (**chosen, submitted**)'), ('two', 'two-term, $b_2$ per-layer table'),
       ('k3075c09', 'V29 term ×0.75 higher-cumulant, ×0.9 covariance'),
       ('k3075', 'V29 term ×0.75 higher-cumulant'), ('k3085', 'V29 term ×0.85 higher-cumulant'), ('k305', 'V29 term ×0.5 higher-cumulant'),
       ('cov125', 'V29 term ×1.25 covariance'), ('cov15', 'V29 term ×1.5 covariance'), ('cov125k3075', '×1.25 covariance, ×0.75 higher-cumulant'),
       ('wk05', 'V29 term ×0.5, both uses'), ('wk15', 'V29 term ×1.5, both uses'), ('wk2', 'V29 term ×2, both uses'),
       ('k3t43', 'V29 term + (4/3) κ3T/v, both uses'), ('k3t43cov', 'V29 term + (4/3) κ3T/v, covariance only'),
       ('tree1all', 'V29 term + Gaussian tree births, both uses'), ('treecov', 'V29 term + tree births, covariance only'),
       ('radv', 'V29\'s coefficient on the radial form $v_i\\Sigma_{ij}$, both uses'), ('zrad2all', 'measured-λ radial, z-space, both uses'), ('rad1all', 'measured-λ radial ×½, both uses'), ('rad1cov', 'measured-λ radial ×½, covariance only'),
       ('eta2all', 'single latent (κ4/v)Σ, both uses'), ('eta1all', 'single latent ×½, both uses'), ('eta1cov', 'single latent ×½, covariance only')]
dev_rows = []
for tag, name in VAR:
    nn, x = d(tag, 'base', DEV)
    if len(nn) >= 2: dev_rows.append((x.mean(), tag, name, nn, x))
for m, tag, name, nn, x in sorted(dev_rows):
    cells = ' / '.join(('%+.1f' % (100 * x[nn.index(i)])) if i in nn else '—' for i in DEV)
    P('| %s | `%s` | %s | %+.1f | %d/%d |' % (name, tag, cells, 100 * m, int((x < 0).sum()), len(nn)))
KEY['n_dev_variants'] = len(dev_rows)
P()
P('Run after the choice (not part of the selection): `two1t21` (orientation fix, §5): ' +
  ' / '.join('%+.1f' % (100 * (R['two1t21'][i] / R['base'][i] - 1)) for i in DEV if i in R.get('two1t21', {})) + ' %.')
P()

# ---------------- C. the closure on every set -----------------
P('### Table C (two-term closure against V29 on every set)')
P()
P('| networks | role | n | mean (%) | 95 % t-CI | bootstrap CI | improved | Σ MSE (%) | p, t (1-sided) | p, sign (2-sided) |')
P('|' + '-' * 7 + '|' + '-' * 22 + '|' + '-' * 4 + '|' + '-' * 9 + '|' + '-' * 16 + '|' + '-' * 16 + '|' + '-' * 8 + '|' + '-' * 9 + '|' + '-' * 8 + '|' + '-' * 7 + '|')
for spec, role in [('0-2', 'development'), ('3-22', 'held out'), ('3-7', 'held out, in V29\'s fit range'),
                   ('8-22', 'held out, outside it'), ('23-49', 'pre-registered, stage 1'), ('50-99', 'pre-registered, stage 2'),
                   ('23-99', 'stages 1 and 2')]:
    nets = nets_from(spec); nn, x = d('two1', 'base', nets)
    if len(nn) < 3: continue
    if spec == '23-99' and not any(i >= 50 for i in nn): continue
    s = stats(x); a = np.array([R['two1'][i] for i in nn]); b = np.array([R['base'][i] for i in nn])
    KEY['two1_' + spec] = s
    P('| %s | %s | %d | %+.2f | [%+.2f, %+.2f] | [%+.2f, %+.2f] | %d/%d | %+.2f | %.1g | %.1g |' % (spec.replace('-', '–'), role, s['n'], 100 * s['mean'],
      100 * s['lo'], 100 * s['hi'], 100 * s['blo'], 100 * s['bhi'], s['improved'], s['n'], 100 * (a.sum() / b.sum() - 1), s['p_one_lt0'], s['p_sign']))
P()

# ---------------- D. ablations / comparators -----------------
def contrast_row(name, num, den, nets):
    nn = [i for i in nets if i in R.get(num, {}) and i in R.get(den, {})]
    if len(nn) < 3: return None
    return name, stats(np.array([R[num][i] / R[den][i] - 1 for i in nn]))
P('### Table D1 (held-out networks 3–22; not pre-registered)')
P()
P('Each row: mean change against V29 [95 % t-CI], networks improved; paired change of the variant against the full closure; '
  'two-sided p of that paired change, Holm-adjusted over the rows shown.')
P()
P('| variant | vs V29 | improved | variant vs closure | Holm p |')
P('|' + '-' * 40 + '|' + '-' * 22 + '|' + '-' * 9 + '|' + '-' * 22 + '|' + '-' * 8 + '|')
rows = [('V29 term + κ3T/v (no table)', 'abl_v29k3t'), ('$b_1$ table alone (no κ3T/v)', 'abl_b1only'),
        ('V29 term ×0.75 in the higher-cumulant use', 'abl_k3075'), ('closure with refitted $b_1$', 'two1r'),
        ('closure with the measured $b_2$ table (dropped on nets 0–2)', 'two')]
got = []
for name, t in rows:
    a_ = contrast_row(name, t, 'base', HO); c_ = contrast_row(name, t, 'two1', HO)
    if a_ and c_ and a_[1]['n'] >= 15: got.append((name, t, a_[1], c_[1]))
s2 = stats(np.array([R['two1'][i] / R['base'][i] - 1 for i in HO]))
P('| full closure | %s | %d/%d | — | — |' % (fmt(s2), s2['improved'], s2['n']))
if got:
    adj = holm(np.array([c['p_two'] for _, _, _, c in got]))
    for (name, t, a_, c_), p_ in zip(got, adj):
        P('| %s | %s | %d/%d | %s | %.1g |' % (name, fmt(a_), a_['improved'], a_['n'], fmt(c_), p_))
        KEY['abl_%s_3-22' % t] = a_; KEY['vsclosure_%s_3-22' % t] = c_
P()
P('### Table D2 (pre-registered stage 1, networks 23–49: every configuration against V29)')
P()
P('| configuration | vs V29 | improved |')
P('|' + '-' * 60 + '|' + '-' * 22 + '|' + '-' * 9 + '|')
for name, t in [('full closure (submitted)', 'two1'), ('V29 term + κ3T/v', 'abl_v29k3t'), ('$b_1$ table alone', 'abl_b1only'),
                ('equal-budget comparator: V29 term ×0.75 higher-cumulant, ×0.9 covariance', 'k3075c09'), ('closure with refitted $b_1$', 'two1r')]:
    r_ = contrast_row(name, t, 'base', S1)
    if r_: P('| %s | %s | %d/%d |' % (name, fmt(r_[1]), r_[1]['improved'], r_[1]['n'])); KEY['abl_%s_23-49' % t] = r_[1]
P()
P('**Pre-registered secondary family** (stage 1; statistic exactly as registered; two-sided; Holm over the four):')
P()
P('| contrast | MSE ratio − 1 | mean [95 % t-CI] | < 0 | p | Holm p |')
P('|' + '-' * 34 + '|' + '-' * 30 + '|' + '-' * 20 + '|' + '-' * 7 + '|' + '-' * 7 + '|' + '-' * 7 + '|')
fam = [('S1', 'V29 term + κ3T/v vs V29', 'abl_v29k3t', 'base'), ('S2', '$b_1$ table alone vs V29', 'abl_b1only', 'base'),
       ('S3', 'closure vs equal-budget comparator', 'two1', 'k3075c09'), ('S4', 'refitted $b_1$ vs closure', 'two1r', 'two1')]
fr = [(k, n_, num, den, contrast_row(n_, num, den, S1)) for k, n_, num, den in fam]
fr = [f for f in fr if f[4]]
if len(fr) == 4:
    adj = holm(np.array([f[4][1]['p_two'] for f in fr]))
    for (k, n_, num, den, (_, st)), p_ in zip(fr, adj):
        P('| %s %s | `%s` / `%s` | %s | %d/%d | %.1g | %.1g |' % (k, n_, num, den, fmt(st), st['improved'], st['n'], st['p_two'], p_))
        KEY['prereg_' + k] = dict(st, holm=p_)
P()

# ---------------- E. orientation fix -----------------
P('### Table E (orientation of the (3,1) slot in the (2,1) update; networks 0–2, %)')
P()
P('| configuration | as in V29 | with $K$ in the transposed term |')
P('|' + '-' * 40 + '|' + '-' * 26 + '|' + '-' * 26 + '|')
for name, a, b in [('V29 (symmetric slot)', 'base', 'base_t21'), ('two-term closure', 'two1', 'two1t21'), ('exact slice, both uses', 's2all', 's2allt21'),
                   ('exact slice, higher-cumulant use only', 's2k3', 's2k3t21'), ('exact slice ×0.75, both uses', 's15all', 's15allt21')]:
    ca = ' / '.join('%+.1f' % (100 * (R[a][i] / R['base'][i] - 1)) if i in R.get(a, {}) else '—' for i in DEV)
    cb = ' / '.join('%+.1f' % (100 * (R[b][i] / R['base'][i] - 1)) if i in R.get(b, {}) else '—' for i in DEV)
    P('| %s | %s | %s |' % (name, ca, cb))
P()

# ---------------- F. multi-network process -----------------
P('### Table F (one process for all networks 3–22: later predict() calls use deeper Strassen levels)')
P()
mb, mt = './logs/multi_base.json', './logs/multi_two1.json'
if os.path.exists(mb) and os.path.exists(mt):
    A = {int(k): v for k, v in json.load(open(mb)).items()}; B = {int(k): v for k, v in json.load(open(mt)).items()}
    nn = sorted(set(A) & set(B)); x = np.array([B[i]['final'] / A[i]['final'] - 1 for i in nn]); s = stats(x)
    sp = np.array([A[i]['final'] / R['base'][i] - 1 for i in nn if i in R['base']])
    P('* closure vs V29, both in one process: %s, %d/%d improved (n = %d).' % (fmt(s), s['improved'], s['n'], len(nn)))
    P('* V29 in one process vs V29 one process per network: mean %+.3f %%, max |·| %.3f %%.' % (100 * sp.mean(), 100 * np.abs(sp).max()))
    fd = sorted(set(B[i]['flops'] - A[i]['flops'] for i in nn)); fr = sorted(set(round(A[i]['frac'], 4) for i in nn))
    P('* FLOPs added by the closure per network: %s; V29 FLOP fractions in the process: %s (first call first).' % (fd, [round(A[i]['frac'], 4) for i in nn[:3]] + ['…']))
    KEY['multi'] = s
else:
    P('(pending)')
P()
open('./writeup/tables_v7.md', 'w').write('\n'.join(OUT) + '\n')
json.dump({k: (v if not isinstance(v, dict) else {a: float(b) for a, b in v.items()}) for k, v in KEY.items()},
          open('./writeup/key_v7.json', 'w'), indent=1, default=float)
print('\n'.join(OUT))
