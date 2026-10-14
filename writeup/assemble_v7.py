"""Fill the template report_v7.md with the generated tables (tables_v7.md) and key numbers, writing report_v7_full.md.
Placeholders: ⟨TB⟩ ⟨TC⟩ ⟨TD⟩ ⟨TE⟩ (tables), ⟨S2⟩ ⟨S2b⟩ (stage 2), ⟨MP⟩ (multi-process), ⟨RL⟩ (rank ladder), ⟨T21S⟩."""
import json, re, sys
sys.path.insert(0, '/home/claude/research')
import numpy as np
from runstats import load, stats, nets_from
T = open('/home/claude/writeup/tables_v7.md').read()
blocks = {}
for m in re.finditer(r'^### (Table \S+)[^\n]*\n(.*?)(?=^### |\Z)', T, re.S | re.M):
    blocks[m.group(1)] = m.group(2).strip()
R = load()
def d(tag, ref, nets):
    nn = [i for i in nets if i in R.get(tag, {}) and i in R.get(ref, {})]
    return nn, np.array([R[tag][i] / R[ref][i] - 1 for i in nn])
def fmt(s, k=2): return '%+.*f %% [%+.*f, %+.*f]' % (k, 100 * s['mean'], k, 100 * s['lo'], k, 100 * s['hi'])
fill = {}
fill['⟨TB⟩'] = blocks.get('Table B', '(pending)')
fill['⟨TC⟩'] = blocks.get('Table C', '(pending)')
fill['⟨TD⟩'] = (blocks.get('Table D1', '') + '\n\n**Pre-registered stage 1 (networks 23–49).**\n\n' + blocks.get('Table D2', '')).strip()
fill['⟨TE⟩'] = blocks.get('Table E', '(pending)')
nn, x = d('two1', 'base', nets_from('50-99'))
if len(nn) >= 3:
    s = stats(x); nn2, x2 = d('two1', 'base', nets_from('23-99')); s2 = stats(x2)
    fill['⟨S2⟩'] = '%s, %d of %d improved, one-sided $p=%s$; over both stages (%d networks) %s, %d improved' % (
        fmt(s), s['improved'], s['n'], ('%.0e' % s['p_one_lt0']).replace('e-0', r'\times10^{-').replace('e-', r'\times10^{-') + '}',
        s2['n'], fmt(s2), s2['improved'])
    fill['⟨S2b⟩'] = 'it holds as well (%s, %d/%d improved)' % (fmt(s), s['improved'], s['n'])
mp = '/home/claude/research/multi_two1.json'; mb = '/home/claude/research/multi_base.json'
try:
    A = {int(k): v for k, v in json.load(open(mb)).items()}; B = {int(k): v for k, v in json.load(open(mp)).items()}
    k = sorted(set(A) & set(B)); s = stats(np.array([B[i]['final'] / A[i]['final'] - 1 for i in k]))
    sp = np.array([A[i]['final'] / R['base'][i] - 1 for i in k])
    fill['⟨MP⟩'] = '%s (%d/%d improved); V29 itself changes by at most %.2f %% between the two set-ups' % (fmt(s, 1), s['improved'], s['n'], 100 * np.abs(sp).max())
except Exception:
    pass
rl = []
for t in ('rowrad4', 'rowrad64'):
    nn, x = d(t, 'base', nets_from('3-12'))
    if len(nn) >= 3: s = stats(x); rl.append('%+.1f %% / ' % (100 * s['mean']) if t == 'rowrad4' else '%+.1f %%' % (100 * s['mean']))
if len(rl) == 2:
    n4, x4 = d('rowrad4', 'base', nets_from('3-12')); n64, x64 = d('rowrad64', 'base', nets_from('3-12'))
    fill['⟨RL⟩'] = '%s%s | %d/%d, %d/%d' % (rl[0], rl[1], int((x4 < 0).sum()), len(x4), int((x64 < 0).sum()), len(x64))
nn, x = d('two1t21', 'base', nets_from('23-99')); nn2, x2 = d('two1t21', 'two1', nets_from('23-99'))
if len(nn2) >= 3:
    s = stats(x); s2 = stats(x2)
    fill['⟨T21S⟩'] = '%s against V29 (%d/%d) and %s against the submitted closure (%d networks)' % (fmt(s), s['improved'], s['n'], fmt(s2), s2['n'])
import runstats as _rs, sealed_analysis as _sa
from runstats import tq
_res, _rt = _sa.load(['/home/claude/research/or29_sealed.log', '/home/claude/research/or29_sealed2.log'])
_nn = [i for i in nets_from('23-99') if i in _rt.get('two1', {}) and i in _rt.get('base', {})]
if len(_nn) >= 3:
    _d = np.array([_rt['two1'][i] - _rt['base'][i] for i in _nn], float); _se = _d.std(ddof=1) / np.sqrt(len(_d)); _q = tq(0.95, len(_d) - 1)
    fill['⟨RT⟩'] = '$%+.1f$ ms (90 %% CI $[%+.1f,%+.1f]$)' % (_d.mean(), _d.mean() - _q * _se, _d.mean() + _q * _se)
_R0 = load(precise=False); _tot = sum(len(v) for v in _R0.values())
_np_ = sum(_rs._precise(t, n, v)[1] for t in _R0 for n, v in _R0[t].items())
fill['⟨PREC⟩'] = '%d of %d' % (_np_, _tot)
src = open('/home/claude/writeup/report_v7.md').read()
for k, v in fill.items():
    src = src.replace(k, re.sub(r'(?<![\w${^])-(?=\d)', '\u2212', v))   # typographic minus in numbers
left = sorted(set(re.findall(r'⟨[^⟩]*⟩', src)))
open('/home/claude/writeup/report_v7_full.md', 'w').write(src)
print('filled', sorted(fill), '| still open:', left)
