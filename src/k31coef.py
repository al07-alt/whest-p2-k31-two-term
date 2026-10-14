import numpy as np
from truek31 import k31, t21
res={}
for i in (0,1,2):
    K=np.load(f'data/kp/marg_{i:03d}.npz')
    for l in range(1,15):
        Kt=k31(i,l); T,k3,v=t21(i,l)
        M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
        m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m)
        off=~np.eye(len(m),dtype=bool)
        X=np.stack([(v[:,None]*S)[off],((k3/v)[:,None]*T)[off]],1); y=Kt[off]
        b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
        b1=np.sum(X[:,0]*y)/np.sum(X[:,0]**2)
        res[(i,l)]=(b[0],b[1],1-e.var()/y.var(),b1)
for l in range(1,15):
    print('l=%2d '%l+' | '.join('net%d b_rad %.4f b_k3T %.3f R2 %.3f (rad-only b %.4f)'%(i,*res[(i,l)]) for i in (0,1,2)))
np.save('data/k31coef.npy',np.array([[res[(i,l)] for l in range(1,15)] for i in (0,1,2)]))
