import numpy as np, sys
from truek31 import k31
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz')
for l in Ls:
    Kt=k31(i,l)  # [a,b] = kappa(a,a,a,b)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); v=np.diag(S).copy()
    off=~np.eye(len(m),dtype=bool)
    F={'vS':v[:,None]*S,'m2S':(m*m)[:,None]*S,'vmm':(v*m)[:,None]*m[None,:],'m3m':(m**3)[:,None]*m[None,:]}
    y=Kt[off]; X=np.stack([F[k][off] for k in F],1)
    def r2(cols):
        Xc=X[:,cols]; b,*_=np.linalg.lstsq(Xc,y,rcond=None); e=y-Xc@b; return 1-e.var()/y.var(), b
    out=[]
    for nm,cols in [('vS',[0]),('vS+m2S',[0,1]),('vS+vmm',[0,2]),('all4',[0,1,2,3])]:
        R,b=r2(cols); out.append('%s R2 %.3f b=%s'%(nm,R,np.array2string(b,precision=4)))
    print('l=%d | '%l+' | '.join(out),flush=True)
