import numpy as np, time, sys
from lib import *
i=int(sys.argv[1]); Wt=W(i,np.float32); n=1024; N=int(sys.argv[2]) if len(sys.argv)>2 else 60000; B=10000
def passes(stage, mu=None):
    rng=np.random.default_rng(31); S1=np.zeros((17,n)); S2=np.zeros((17,n,n)) if stage==1 else None
    D2=[[] for _ in range(17)]; MD=[[] for _ in range(17)]
    for b in range(N//B):
        h=rng.standard_normal((B,n)).astype(np.float32)
        for k in range(17):
            if k>0: h=np.maximum(h@Wt[k-1],0)
            hd=h.astype(np.float64)
            if stage==1: S1[k]+=hd.sum(0); S2[k]+=hd.T@hd
            else:
                d=hd-mu[k]; D2[k].append((d*d).sum(1)); MD[k].append(d@mu[k])
    return S1,S2,D2,MD
t=time.time(); S1,S2,_,_=passes(1); mu=S1/N; C=np.stack([S2[k]/N-np.outer(mu[k],mu[k]) for k in range(17)])
_,_,D2,MD=passes(2,mu); print('%.0fs'%(time.time()-t))
zt=np.load(f'data/kp/marg_{i:03d}.npz'); r1,r2,r3,r4=zt['pre_mean'],zt['pre_m2'],zt['pre_m3'],zt['pre_m4']
c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
print('k(post h_k) | lam_true(z_{k+1}) ratio-of-means | identity | n_eff | skew slope true | identity')
for k in range(16):   # h index k in 0..16 where 0=x, k=post h_{k-1}; z_k = W_k^T h_{k} (h stored index k)
    Ck=C[k]; trC=np.trace(Ck); F2=np.sum(Ck*Ck); d2=np.concatenate(D2[k]); md=np.concatenate(MD[k])
    lam_id=3*(d2.var()-2*F2)/trC**2
    l=k  # pre-activation layer index z_l uses h stored at index k=l
    lam_tr=len(c4[l])*np.sum(c4[l])/np.sum(c2[l])**2
    a=r1[l]/np.sqrt(c2[l]); g=c3[l]/c2[l]**1.5; sl_tr=np.sum(a*g)/np.sum(a*a)
    sl_id=3*np.mean(md*(d2-d2.mean()))/(np.dot(mu[k],mu[k])*trC)
    print('z%-2d lam_true %.4f  identity %.4f | n_eff %5.0f | skew slope true %.4f identity %.4f | corr(gamma,alpha) %.3f'%(l,lam_tr,lam_id,trC**2/F2,sl_tr,sl_id,np.corrcoef(a,g)[0,1]),flush=True)
