import numpy as np
from truek31 import k31
bins=np.array([-4,-2.5,-1.5,-0.75,0,0.75,1.5,2.5,4]); out={}
for i in (0,1,2):
    K=np.load(f'data/kp/marg_{i:03d}.npz')
    for l in (8,14):
        Kt=k31(i,l); M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
        m=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]; X4=K['pre_m4'][l]
        S=M11-np.outer(m,m); v=np.diag(S).copy(); a=m/np.sqrt(v); eta=(X4-4*m*X3+6*m*m*X2-3*m**4-3*v*v)/v**2
        np.fill_diagonal(S,0); Kd=Kt.copy(); np.fill_diagonal(Kd,0); R=v[:,None]*S
        sl=[]; et=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            s_=(a>=lo)&(a<hi); sl.append((Kd[s_]*R[s_]).sum()/(R[s_]**2).sum()); et.append(eta[s_].mean())
        out[f's_{i}_{l}']=np.array(sl); out[f'eta_{i}_{l}']=np.array(et); out[f'glob_{i}_{l}']=(Kd*R).sum()/(R*R).sum()
        print(i,l,np.round(sl,4),'glob %.4f'%out[f'glob_{i}_{l}'],flush=True)
np.savez('data/k31alpha3.npz',bins=bins,**out)
