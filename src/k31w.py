import numpy as np, sys
from scipy.special import ndtr
from truek31 import k31, t21
from lib import W
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz'); Wt=W(i)
pdf=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
for l in Ls:
    Kt=k31(i,l); T,k3,v=t21(i,l)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); s=np.sqrt(v); a=m/s
    F1=ndtr(a); F3=-a*pdf(a)/v
    off=~np.eye(len(m),dtype=bool)
    def dC(Km):
        X=(F3[:,None]*F1[None,:])*Km/6; X=X+X.T; X[~off]=0; return X
    Wn=Wt[l] if l<16 else None
    def dvar(X): return np.einsum('ac,ab,bc->c',Wn,X,Wn) if Wn is not None else None
    feats={'vS':v[:,None]*S,'eta':(((K['pre_m4'][l]-4*m*K['pre_m3'][l]+6*m*m*K['pre_m2'][l]-3*m**4-3*v*v)/v**2)*v)[:,None]*S,
           'T3T':(k3/v)[:,None]*T,'m2S':(m*m)[:,None]*S,'vmm':(v*m)[:,None]*m[None,:]}
    yC=dC(Kt); yV=dvar(yC)
    DC={k:dC(f) for k,f in feats.items()}; DV={k:dvar(x) for k,x in DC.items()}
    out=[]
    for names in [['vS'],['vS','T3T'],['vS','eta'],['vS','m2S','vmm'],['vS','T3T','eta','m2S','vmm']]:
        X=np.stack([DC[k][off] for k in names],1); b=np.linalg.lstsq(X,yC[off],rcond=None)[0]; e=yC[off]-X@b
        r2c=1-e.var()/yC[off].var()
        Xv=np.stack([DV[k] for k in names],1); bv=np.linalg.lstsq(Xv,yV,rcond=None)[0]; ev=yV-Xv@bv
        # var-space R2 using the dC-fitted coefficients
        pv=Xv@b; r2v_c=1-np.mean((yV-pv)**2)/np.mean((yV-yV.mean())**2)
        out.append('%s: dC %.3f | dvar(own fit) %.3f dvar(dC-fit) %.3f b=%s'%('+'.join(names),r2c,1-ev.var()/yV.var(),r2v_c,np.array2string(b,precision=3)))
    print('l=%d |dvar_true| rms %.2e  rel to var %.2e'%(l,np.sqrt(np.mean(yV**2)),np.sqrt(np.mean((yV/np.einsum('ac,ab,bc->c',Wn,S*0+np.diag(np.diag(S)),Wn))**2))))
    for o in out: print('   ',o,flush=True)
