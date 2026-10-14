import numpy as np, glob
from scipy.special import ndtr
from lib import *
lam=np.load('data/lam_table.npy')
ids=sorted(int(f[-7:-4]) for f in glob.glob('data/feat/f*.npz'))
D={}
for i in ids:
    F=np.load(f'data/feat/f{i:03d}.npz'); Wt=W(i); tr=TRUTH[list(IDS).index(i)]
    e=tr-F['pred']
    for l in range(1,16):
        m=F['m'][l]; v=F['v'][l]; s=np.sqrt(v); a=m/s; P=ndtr(a); p=pdf(a)
        dm=e[l-1]@Wt[l]
        eps=e[l]-P*dm-0.5*p/s*dm**2
        D.setdefault(l,[]).append(dict(i=i,a=a,g3=F['k3'][l]/s**3,s=s,eps=eps/s,e=e[l]/s,prop=(P*dm)/s))
for l in [2,4,8,12,15]:
    R=D[l]; tr_=[r for r in R if r['i']%2==0]; te=[r for r in R if r['i']%2==1]
    c=lambda S,k: np.concatenate([r[k] for r in S])
    print(l,'rms e/s %.2e prop %.2e eps %.2e'%(np.sqrt(np.mean(c(R,'e')**2)),np.sqrt(np.mean(c(R,'prop')**2)),np.sqrt(np.mean(c(R,'eps')**2))))
    def X(S):
        a=c(S,'a'); g=c(S,'g3'); p=pdf(a)
        return np.stack([p,p*a,p*a*a,p*a**3,p*g,p*g*a,p*g*a*a,p*g*g,p*g**3],1)
    b=np.linalg.lstsq(X(tr_),c(tr_,'eps'),rcond=None)[0]; y=c(te,'eps')
    print('   R2 eps local poly %.3f'%(1-np.mean((y-X(te)@b)**2)/np.mean(y**2)))
