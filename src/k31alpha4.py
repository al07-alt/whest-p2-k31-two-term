import numpy as np
from truek31 import k31
bins=np.array([-99,-4,-2.5,-1.5,-0.75,0,0.75,1.5,2.5,4,99])
for i in (0,1,2):
    K=np.load(f'data/kp/marg_{i:03d}.npz')
    for l in (14,):
        Kt=k31(i,l); M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
        m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); v=np.diag(S).copy(); a=m/np.sqrt(v)
        np.fill_diagonal(S,0); Kd=Kt.copy(); np.fill_diagonal(Kd,0); R=v[:,None]*S
        g=(Kd*R).sum()/(R*R).sum(); out=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            s_=(a>=lo)&(a<hi); sl=(Kd[s_]*R[s_]).sum()/(R[s_]**2).sum(); w=(R[s_]**2).sum()/(R*R).sum()
            out.append('[%g,%g) n=%d %.2f (w %.2f)'%(lo,hi,s_.sum(),sl/g,w))
        print('net',i,'l',l,' | '.join(out),flush=True)
