import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
Z=np.load('data/old0_pa.npz')
for l in [6,10,15]:
    m=Z[f'm_{l}']; S=Z[f'S_{l}'].astype(float); v=np.diag(S); n=len(m); Tn=Z[f'Tn_{l}'].astype(float)
    R1=2*m[:,None]*S+m[None,:]*v[:,None]; R2=np.outer(m*m,m)
    Lf=lambda u: 2*u[:,None]*S+u[None,:]*v[:,None]
    Lt=lambda R: 2*np.sum(R*S,1)+R.T@v
    Aop=LinearOperator((n,n),matvec=lambda u: Lt(Lf(u)))
    ages=sorted(int(k.split('_')[2]) for k in Z.files if k.startswith(f'Ts_{l}_'))
    for a in ages:
        Ts=Z[f'Ts_{l}_{a}'].astype(float)
        X=np.stack([R1.ravel(),R2.ravel()],1); b=np.linalg.lstsq(X,Ts.ravel(),rcond=None)[0]
        r2=1-np.sum((Ts.ravel()-X@b)**2)/np.sum(Ts**2)
        u,_=cg(Aop,Lt(Ts),maxiter=300,rtol=1e-9); r2u=1-np.sum((Ts-Lf(u))**2)/np.sum(Ts**2)
        print(l,'age',a,'|Ts|/|Tn| %.3f'%(np.linalg.norm(Ts)/np.linalg.norm(Tn)),'R2 radial(b,g) %.3f'%r2,'b %.2e g %.2e'%tuple(b),'R2 free-u %.3f'%r2u,flush=True)
