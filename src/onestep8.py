import numpy as np, pcp3
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
PM=np.load('data/kp/M11_000.npy',mmap_mode='r')
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=True,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,edgecoef=True))
o_=~np.eye(1024,dtype=bool)
for l in [4,7,10,13]:
    m,S,T,Q,K31=state(l)
    mu,C,G,Mx,K31h,aux=pcp3.transfer3(m,S,T,Q,K31,6,dict(base,_layer=l))
    w=Wt[l+1]; Q1,K4w=pcp3.wick4(w,S,aux,True)
    m2,S2,T2,Q2,K312=state(l+1); v2=np.diag(S2)
    RK=v2[:,None]*S2
    X=np.stack([RK[o_],K4w[o_]],1); y=K312[o_]
    b=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-np.sum((y-X@b)**2)/np.sum(y**2)
    b1=np.sum(RK[o_]*y)/np.sum(RK[o_]**2); r21=1-np.sum((y-b1*RK[o_])**2)/np.sum(y**2)
    cw=np.corrcoef(K4w[o_],y-b1*RK[o_])[0,1]
    print(l+1,'K31 R2: radial %.3f | radial+wick %.3f  coef %s | corr(wick, radial-resid) %.3f'%(r21,r2,np.round(b,4),cw),flush=True)
    # one-step var error at l+1 -> l+2 using model K31
    pm=K['mean'][l+1]; Ct=PM[l+1].astype(float)-np.outer(pm,pm); w2_=Wt[l+2]; vv=np.einsum('ji,jk,ki->i',w2_,Ct,w2_)
    for nm,KK in [('true',K312),('radial',b1*RK),('rad+wick',b[0]*RK+b[1]*K4w)]:
        KK=KK.copy(); np.fill_diagonal(KK,np.diag(K312))
        mu,C,G,Mx,_,_=pcp3.transfer3(m2,S2,T2,Q2,KK,6,dict(base,_layer=l+1))
        dv=np.einsum('ji,jk,ki->i',w2_,C-Ct,w2_); print('    %-9s next var rel %.3e'%(nm,np.sqrt(np.mean((dv/vv)**2))),flush=True)
