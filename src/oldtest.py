import numpy as np, fac3
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam,diag_old=5)
fac3.run_fac3(W(0),opts=EX)
for d in fac3.OLD:
    l=d['l']; To=d['Told'].astype(float); Tn=d['Tn'].astype(float); m=d['m']; S=d['S'].astype(float); v=np.diag(S); s=np.sqrt(v)
    if not np.any(To): continue
    R1=2*m[:,None]*S+m[None,:]*v[:,None]   # sym(m x S) (2,1)-slice: k(i,i,j)
    R2=(m*m)[:,None]*m[None,:]
    off=~np.eye(len(m),dtype=bool)
    X=np.stack([R1[off],R2[off]],1); y=To[off]
    b=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-np.sum((y-X@b)**2)/np.sum(y**2)
    Xd=np.stack([np.diag(R1),np.diag(R2)],1); yd=np.diag(To); bd=np.linalg.lstsq(Xd,yd,rcond=None)[0]; r2d=1-np.sum((yd-Xd@bd)**2)/np.sum(yd**2)
    print(l,'|Told|/|Tn| off %.3f diag %.3f'%(np.linalg.norm(y)/np.linalg.norm(Tn[off]),np.linalg.norm(yd)/np.linalg.norm(np.diag(Tn))),'R2off %.3f'%r2,np.round(b,5),'R2diag %.3f'%r2d,np.round(bd,5),flush=True)
