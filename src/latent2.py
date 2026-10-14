import numpy as np, time, sys
from lib import *
i=int(sys.argv[1]); Wt=W(i,np.float32); n=1024; N=200000; B=10000; Ls=[4,8,12,15]
names=['S1','S2','M1']   # S1=||h_{l-1}||^2/n, S2=||h_{l-2}||^2/n, M1=mean(h_{l-1})
def newacc(): return {l:{nm:dict(zX=np.zeros(n),z2X=np.zeros(n),X=0.,X2=0.) for nm in names}|dict(z=np.zeros(n),z2=np.zeros(n),cnt=0) for l in Ls}
halves=[newacc(),newacc()]; rng=np.random.default_rng(23); t=time.time()
for b in range(N//B):
    A=halves[b%2]; h=rng.standard_normal((B,n)).astype(np.float32); Sprev=np.mean(h.astype(np.float64)**2,1)
    for l in range(16):
        S1=np.mean(h.astype(np.float64)**2,1); M1=np.mean(h.astype(np.float64),1)
        z=(h@Wt[l]).astype(np.float64)
        if l in A:
            a=A[l]; a['z']+=z.sum(0); a['z2']+=(z*z).sum(0); a['cnt']+=B
            for nm,X in (('S1',S1),('S2',Sprev),('M1',M1)):
                d=a[nm]; d['zX']+=(z*X[:,None]).sum(0); d['z2X']+=(z*z*X[:,None]).sum(0); d['X']+=X.sum(); d['X2']+=(X*X).sum()
        Sprev=S1; h=np.maximum(z,0).astype(np.float32)
print('%.0fs'%(time.time()-t))
zt=np.load(f'data/kp/marg_{i:03d}.npz'); r1,r2,r3,r4=zt['pre_mean'],zt['pre_m2'],zt['pre_m3'],zt['pre_m4']
c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
def preds(a,nm):
    N_=a['cnt']; d=a[nm]; Ez=a['z']/N_; Ez2=a['z2']/N_; EX=d['X']/N_; VX=d['X2']/N_-EX**2
    czX=d['zX']/N_-Ez*EX; cz2X=d['z2X']/N_-Ez2*EX; cv=cz2X-2*Ez*czX
    return 3*czX*cv/VX, 3*cv**2/VX
for l in Ls:
    v=c2[l]; sk=c3[l]/v**1.5; ek=c4[l]/v**2; row=[]
    for nm in names:
        p3a,p4a=preds(halves[0][l],nm); p3b,p4b=preds(halves[1][l],nm)
        rel3=np.corrcoef(p3a/v**1.5,p3b/v**1.5)[0,1]; rel4=np.corrcoef(p4a/v**2,p4b/v**2)[0,1]
        p3=(p3a+p3b)/2; p4=(p4a+p4b)/2
        c3s=np.corrcoef(p3/v**1.5,sk)[0,1]; c4s=np.corrcoef(p4/v**2,ek)[0,1]
        rel3f=2*rel3/(1+rel3); rel4f=2*rel4/(1+rel4)
        row.append('%s: skew corr %.3f (rel %.2f, disattn %.3f) exkurt corr %.3f (rel %.2f)'%(nm,c3s,rel3f,c3s/np.sqrt(max(rel3f,1e-9)),c4s,rel4f))
    print('net',i,'L%d'%l,' || '.join(row),flush=True)
