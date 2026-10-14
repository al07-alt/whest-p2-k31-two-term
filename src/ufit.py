import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
Z=np.load('data/old0_A3.npz')
for l in [5,8,11,14,15]:
    To=Z[f'Told_{l}'].astype(float); Tn=Z[f'Tn_{l}'].astype(float); m=Z[f'm_{l}']; S=Z[f'S_{l}'].astype(float); v=np.diag(S); n=len(m)
    Lf=lambda u: 2*u[:,None]*S+u[None,:]*v[:,None]
    Lt=lambda R: 2*np.sum(R*S,1)+R.T@v
    A=LinearOperator((n,n),matvec=lambda u: Lt(Lf(u)))
    u,_=cg(A,Lt(To),maxiter=500,rtol=1e-10)
    R1=Lf(m); b=np.sum(R1*To)/np.sum(R1*R1)
    r2=lambda P: 1-np.sum((To-P)**2)/np.sum(To**2)
    cu=np.corrcoef(u,m)[0,1]
    print(l,'|Told|/|Tn| %.3f'%(np.linalg.norm(To)/np.linalg.norm(Tn)),'R2 beta*m %.3f'%r2(b*R1),'R2 free u %.3f'%r2(Lf(u)),'corr(u,m) %.3f'%cu, flush=True)
