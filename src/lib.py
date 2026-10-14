import numpy as np, json
from scipy.special import ndtr
SQ2PI=np.sqrt(2*np.pi)
D='data'
T=np.load(f'{D}/truth.npz'); IDS=T['ids']; TRUTH=T['means']
def W(i, dtype=np.float64):
    return np.load(f'{D}/npy/w{i:03d}.npy', mmap_mode='r').astype(dtype)
def pdf(x): return np.exp(-0.5*x*x)/SQ2PI
def relu_moments(m, v):
    s=np.sqrt(np.maximum(v,1e-30)); a=m/s; P=ndtr(a); p=pdf(a)
    mu=m*P+s*p; ez2=(m*m+v)*P+m*s*p
    return mu, np.maximum(ez2-mu*mu,0), a, s, P, p
def score(pred, i):
    t=TRUTH[list(IDS).index(i)]
    return np.mean((pred[-1]-t[-1])**2), np.mean((pred-t)**2)
def evaluate(fn, ids=None, verbose=False):
    ids = IDS if ids is None else ids
    fl=[]; al=[]
    for i in ids:
        p=fn(W(i)); f,a=score(p,i); fl.append(f); al.append(a)
        if verbose: print(i, f, a, flush=True)
    return np.mean(fl), np.mean(al), np.array(fl)
