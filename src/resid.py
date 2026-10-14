import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
Z=np.load('data/old0_A3.npz')
for l in [8,15]:
    To=Z[f'Told_{l}'].astype(float); m=Z[f'm_{l}']; S=Z[f'S_{l}'].astype(float); v=np.diag(S); n=len(m); s=np.sqrt(v)
    Lf=lambda u,S=S,v=v: 2*u[:,None]*S+u[None,:]*v[:,None]
    Lt=lambda R,S=S,v=v: 2*np.sum(R*S,1)+R.T@v
    u,_=cg(LinearOperator((n,n),matvec=lambda u: Lt(Lf(u))),Lt(To),maxiter=500,rtol=1e-10)
    R=To-Lf(u)
    d=np.diag(R); off=R-np.diag(d)
    print(l,'resid frac %.3f'%(np.linalg.norm(R)/np.linalg.norm(To)),'diag share %.3f'%(np.linalg.norm(d)/np.linalg.norm(R)))
    sv=np.linalg.svd(off/np.outer(s**2,s),compute_uv=False); e=np.cumsum(sv**2)/np.sum(sv**2)
    print('  normalized resid energy in top k:',{k:round(float(e[k-1]),3) for k in (1,4,16,64,256)})
    Rn=off/np.outer(s**2,s); Sn=S/np.outer(s,s)
    for nm,F in [('rho',Sn),('rho^2',Sn**2),('rho^3',Sn**3),('m_i rho',(m/s)[:,None]*Sn),('m_j',np.outer(np.ones(n),m/s)),('m_i m_j',np.outer(m/s,m/s)),('m_i^2 m_j',np.outer((m/s)**2,m/s)),('rho*m_j^2',Sn*np.outer(np.ones(n),(m/s)**2))]:
        F=F-np.diag(np.diag(F)); print('   corr with %-10s %.3f'%(nm,np.sum(F*Rn)/np.linalg.norm(F)/np.linalg.norm(Rn)))
