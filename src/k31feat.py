import numpy as np, sys
from truek31 import k31
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz')
for l in Ls:
    Kt=k31(i,l)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    M21=np.load(f'data/kp/pre_M21_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]; X4=K['pre_m4'][l]
    S=M11-np.outer(m,m); v=np.diag(S).copy()
    A=m[:,None]; B=m[None,:]
    T=M21-B*X2[:,None]-2*A*M11+2*A*A*B   # kappa(a,a,b)
    k3=X3-3*m*X2+2*m**3; k4=X4-4*m*X3+6*m*m*X2-3*m**4-3*v*v
    eta=k4/v**2; s=np.sqrt(v); al=m/s
    off=~np.eye(len(m),dtype=bool)
    F={'vS':v[:,None]*S,'m2S':(m*m)[:,None]*S,'vmm':(v*m)[:,None]*B,'m3m':(m**3)[:,None]*B,
       'T3T':(k3/v)[:,None]*T,'sqeta':(np.sqrt(np.clip(eta,0,None))*v)[:,None]*S,'eta':(eta*v)[:,None]*S,
       'Tj':  (k3/v)[None,:]*T.T, 'vSj':v[None,:]*S}
    y=Kt[off]
    def r2(names):
        X=np.stack([F[k][off] for k in names],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; return 1-e.var()/y.var()
    res=[]
    for names in [['vS'],['eta'],['sqeta'],['vS','eta'],['vS','T3T'],['vS','m2S','vmm','m3m'],['vS','m2S','vmm','m3m','T3T','eta'],['vS','m2S','vmm','m3m','T3T','eta','Tj','vSj']]:
        res.append('%s:%.3f'%('+'.join(names),r2(names)))
    print('l=%d '%l+'  '.join(res),flush=True)
