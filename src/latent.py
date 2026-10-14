import numpy as np, time
from lib import *
import sys; i=int(sys.argv[1]) if len(sys.argv)>1 else 0; Wt=W(i,np.float32); n=1024; N=200000; B=10000
Ls=[2,4,8,12,15]; acc={l:dict(z=np.zeros(n),z2=np.zeros(n),zS=np.zeros(n),z2S=np.zeros(n),S=0.,S2=0.,S3=0.) for l in Ls}
rng=np.random.default_rng(11); t=time.time()
for b in range(N//B):
    h=rng.standard_normal((B,n)).astype(np.float32)
    for l in range(16):
        S=np.mean(h.astype(np.float64)**2,1)
        z=(h@Wt[l]).astype(np.float64)
        if l in acc:
            a=acc[l]; Sc=S
            a['z']+=z.sum(0); a['z2']+=(z*z).sum(0); a['zS']+=(z*Sc[:,None]).sum(0); a['z2S']+=(z*z*Sc[:,None]).sum(0); a['S']+=Sc.sum(); a['S2']+=(Sc*Sc).sum()
        h=np.maximum(z,0).astype(np.float32)
print('%.0fs'%(time.time()-t))
for i_ in [i]:
    zt=np.load(f'data/kp/marg_{i_:03d}.npz')
    r1,r2,r3,r4=zt['pre_mean'],zt['pre_m2'],zt['pre_m3'],zt['pre_m4']
    c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
for l in Ls:
    a=acc[l]; Ez=a['z']/N; Ez2=a['z2']/N; ES=a['S']/N; VS=a['S2']/N-ES**2
    cz2S=a['z2S']/N-Ez2*ES; czS=a['zS']/N-Ez*ES
    cvar=cz2S-2*Ez*czS          # Cov(var-like (z-Ez)^2, S) approx: Cov(z^2,S) - 2 Ez Cov(z,S)
    pred=3*cvar**2/VS            # single-latent kappa4
    pred3=3*czS*cvar/VS          # single-latent kappa3 (leading)
    y=c4[l]; s=np.sum(pred*y)/np.sum(pred*pred); r2_=1-np.sum((y-pred)**2)/np.sum(y*y)
    y3=c3[l]; s3=np.sum(pred3*y3)/np.sum(pred3*pred3); r23=1-np.sum((y3-pred3)**2)/np.sum(y3*y3)
    np.savez(f'data/latent_{i}_{l}.npz',pred3=pred3,k3=y3,pred4=pred,k4=y,V=VS/ES**2)
    print('layer',l,'V=%.4f | kappa4: slope(true/pred) %.3f  R2(pred as-is) %.3f corr %.3f | kappa3: slope %.3f corr %.3f'%(VS/ES**2,s,r2_,np.corrcoef(pred,y)[0,1],s3,np.corrcoef(pred3,y3)[0,1]))
