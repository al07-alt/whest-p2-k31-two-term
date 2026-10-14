import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'radfit.py')).read().split("res={}")[0])
n=1024
for i in [0]:
    R=np.load(f'data/rec{i}_full.npz')
    for l in [6,9,12,15]:
        T=Ttrue(i,l); Tc=R['T'][l].astype(float); S=R['S'][l].astype(float); m=R['m'][l]; v=np.diag(S); D=T-Tc
        Lf=lambda u: 2*u[:,None]*S+u[None,:]*v[:,None]
        Lt=lambda X: 2*np.sum(X*S,1)+X.T@v
        u,_=cg(LinearOperator((n,n),matvec=lambda u: Lt(Lf(u))),Lt(D),maxiter=400,rtol=1e-10)
        r2u=1-np.sum((D-Lf(u))**2)/np.sum(D**2)
        # u + m^3
        R2=np.outer(m*m,m)
        X=np.stack([Lf(u).ravel(),R2.ravel()],1); b=np.linalg.lstsq(X,D.ravel(),rcond=None)[0]; r2um=1-np.sum((D.ravel()-X@b)**2)/np.sum(D**2)
        g=np.sum(R2*D)/np.sum(R2*R2); r2m=1-np.sum((D-g*R2)**2)/np.sum(D**2)
        print(l,'deficit R2: m^3 %.3f | free Sym(u x S) %.3f | both %.3f | corr(u,m) %.3f'%(r2m,r2u,r2um,np.corrcoef(u,m)[0,1]),flush=True)
