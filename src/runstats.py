"""Network-level statistics of V29 runs, from every or29*.log.
Usage: runstats.py [nets-spec] [tag ...]   nets-spec like 3-22 or 23-49,0-2 ; tags compared with 'base' (or REF=tag).
For each tag: mean of d_i = MSE_tag/MSE_ref - 1 over the networks where both exist, 95 % t-CI, 10 000-resample
network bootstrap CI, exact two-sided sign-test p, number improved, ratio of summed MSEs - 1.
Also importable: load() -> {tag: {net: final}}, stats(d) -> dict, ratio_of_means(num, den) with bootstrap CI."""
import glob, math, os, re, sys
import numpy as np

LOGDIR = './logs/'


# known mislabels: (log file, tag) -> replacement tag. or29_b.log's 'base' rows for nets 0-2 are the scale-1 oracle
# (identical to its 'or1-15' rows), not V29.
RELABEL = {('or29_b.log', 'base'): 'or1-15_dup'}
TRUTH = None


def _precise(tag, net, logged):
    """Final-layer MSE recomputed from the saved prediction, if that file belongs to this run (agrees with the
    4-digit logged value); otherwise the logged value."""
    global TRUTH
    f = f'./data/pred/{tag}_{net}.npy'
    if os.environ.get('PRECISE', '1') == '0' or not os.path.exists(f):   # PRECISE=0: 4-digit logged values only
        return logged, False
    if TRUTH is None:
        T = np.load('./data/truth.npz'); TRUTH = dict(zip(T['ids'].tolist(), T['means']))
    p = np.load(f).astype(np.float64)
    mse = float(np.mean((p[-1] - TRUTH[net][-1].astype(np.float64)) ** 2))
    return (mse, True) if abs(mse / logged - 1) < 6e-4 else (logged, False)


def load(pattern='or29*.log', verbose=False, precise=True):
    res, src = {}, {}
    for f in sorted(glob.glob(os.path.join(LOGDIR, pattern))):
        b = os.path.basename(f)
        for line in open(f, errors='ignore'):
            m = re.match(r'^(\S+) (\d+) final ([0-9.e+-]+)', line)
            if not m:
                continue
            tag, net, val = RELABEL.get((b, m.group(1)), m.group(1)), int(m.group(2)), float(m.group(3))
            old = res.setdefault(tag, {}).get(net)
            if old is not None and abs(old / val - 1) > 1e-3 and verbose:
                print(f'# conflict: {tag} net {net}: {old:.4e} ({src[(tag, net)]}) vs {val:.4e} ({b}); keeping the later file')
            res[tag][net] = val
            src[(tag, net)] = b
    if precise:
        nprec = 0
        for tag in res:
            for net in res[tag]:
                res[tag][net], ok = _precise(tag, net, res[tag][net]); nprec += ok
        if verbose:
            print(f'# {nprec} of {sum(len(v) for v in res.values())} MSEs recomputed at full precision from saved predictions')
    load.src = src
    return res


def _tcdf(t, df):
    # Student-t CDF via the regularized incomplete beta (continued fraction)
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1; c, d = 1.0, 1 - qab * x / qap
        d = 1 / d if abs(d) > 1e-300 else 1e300; h = d
        for m in range(1, 300):
            m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; c = 1 + aa / c; d = 1 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; c = 1 + aa / c; d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-12: break
        return h
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    ib = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x)) * betacf(a, b, x) / a if 0 < x < 1 else (0.0 if x <= 0 else 1.0)
    return 1 - 0.5 * ib if t > 0 else 0.5 * ib


def tq(p, df):
    """t quantile (CDF = p) by bisection on _tcdf."""
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _tcdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def stats(d, seed=0, B=10000):
    d = np.asarray(d, float); n = len(d)
    m = d.mean(); se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
    tq_ = tq(0.975, n - 1) if n > 1 else float('nan')
    rng = np.random.default_rng(seed); bs = d[rng.integers(0, n, (B, n))].mean(1)
    k = int((d < 0).sum()); nz = int((d != 0).sum())
    p_sign = min(1.0, 2 * sum(math.comb(nz, j) for j in range(0, min(k, nz - k) + 1)) / 2 ** nz) if nz else 1.0
    t = m / se if se > 0 else float('nan')
    p_one = _tcdf(t, n - 1) if n > 1 else float('nan')          # one-sided p for mean < 0
    return dict(n=n, mean=m, lo=m - tq_ * se, hi=m + tq_ * se, blo=np.percentile(bs, 2.5), bhi=np.percentile(bs, 97.5),
                improved=k, p_sign=p_sign, p_one_lt0=p_one, p_two=2 * min(p_one, 1 - p_one), t=t)


def paired(res, tag, ref, nets):
    nn = [i for i in nets if i in res.get(tag, {}) and i in res.get(ref, {})]
    a = np.array([res[tag][i] for i in nn]); b = np.array([res[ref][i] for i in nn])
    return nn, a, b


def ratio_of_means(num, den, seed=0, B=10000):
    """mean(num)/mean(den) over networks, with a network bootstrap 95 % CI (paired resampling)."""
    num, den = np.asarray(num, float), np.asarray(den, float); n = len(num)
    rng = np.random.default_rng(seed); ix = rng.integers(0, n, (B, n))
    r = num[ix].mean(1) / den[ix].mean(1)
    return num.mean() / den.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def nets_from(spec):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == '__main__':
    res = load(verbose=True)
    nets = nets_from(sys.argv[1]) if len(sys.argv) > 1 else list(range(3, 23))
    ref = os.environ.get('REF', 'base')
    tags = sys.argv[2:] or sorted(t for t in res if t != ref)
    print(f'ref={ref}  nets={sys.argv[1] if len(sys.argv) > 1 else "3-22"}')
    print('%-14s %3s %8s %18s %18s %6s %9s %9s %8s' % ('tag', 'n', 'mean%', 't-CI95%', 'boot-CI95%', 'impr', 'p_sign', 'p_1s<0', 'sum%'))
    for t in tags:
        nn, a, b = paired(res, t, ref, nets)
        if len(nn) < 2:
            continue
        s = stats(a / b - 1)
        print('%-14s %3d %+8.2f [%+7.2f,%+7.2f] [%+7.2f,%+7.2f] %3d/%-2d %9.2g %9.2g %+8.2f' % (
            t, s['n'], 100 * s['mean'], 100 * s['lo'], 100 * s['hi'], 100 * s['blo'], 100 * s['bhi'], s['improved'], s['n'],
            s['p_sign'], s['p_one_lt0'], 100 * (a.sum() / b.sum() - 1)))
