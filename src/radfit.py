import numpy as np
from lib import *
o_=~np.eye(1024,dtype=bool)
def Ttrue(i,l):
    K=np.load(f'data/kp/marg_{i:03d}.npz'); M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r'); M21=np.load(f'data/kp/pre_M21_{i:03d}.npy',mmap_mode='r')
    a=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; m11=M11[l].astype(float); m21=M21[l].astype(float); A=a[:,None]; B=a[None,:]
    return m21-B*X2[:,None]-2*A*m11+2*A*A*B
res={}
for i in [0,1,2]:
    R=np.load(f'data/rec{i}_full.npz')
    for l in range(1,16):
        T=Ttrue(i,l); Tc=R['T'][l].astype(float); Sc=R['S'][l].astype(float); mc=R['m'][l]; vc=np.diag(Sc)
        R1=2*mc[:,None]*Sc+mc[None,:]*vc[:,None]; R2=np.outer(mc*mc,mc)
        D=T-Tc; X=np.stack([R1.ravel(),R2.ravel()],1); b=np.linalg.lstsq(X,D.ravel(),rcond=None)[0]
        r2=1-np.sum((D.ravel()-X@b)**2)/np.sum(D**2)
        res[(i,l)]=(b,r2,np.linalg.norm(D)/np.linalg.norm(T))
for l in range(1,16):
    print(l,' '.join('MLP%d b %.2e g %.2e R2 %.2f err %.3f |'%(i,*res[(i,l)][0],res[(i,l)][1],res[(i,l)][2]) for i in [0,1,2]))
np.save('data/radfit.npy',np.array([[res[(i,l)][0] for l in range(1,16)] for i in [0,1,2]]))
