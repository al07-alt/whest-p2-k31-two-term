import numpy as np, pcp3, sys
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
lam=np.load('data/lam_table.npy')
PM=np.load('data/kp/M11_000.npy',mmap_mode='r')
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=True,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,edgecoef=True))
for l in [5,8,11,14]:
    m,S,T,Q,K31=state(l); pm=K['mean'][l]; Ct=PM[l].astype(float)-np.outer(pm,pm)
    ok=np.diag(Ct)>1e-3*np.median(np.diag(Ct)); sd=np.sqrt(np.maximum(np.diag(Ct),1e-12)); off=~np.eye(1024,dtype=bool)&np.outer(ok,ok)
    w=Wt[l+1]; v2=np.einsum('ji,jk,ki->i',w,Ct,w); v=np.diag(S)
    Qr=lam[l]/3*(np.outer(v,v)+2*S*S); np.fill_diagonal(Qr,np.diag(Q)); Kr=lam[l]*v[:,None]*S; np.fill_diagonal(Kr,np.diag(K31))
    # fitted lambda for off-diagonals (oracle scalar)
    o_=~np.eye(1024,dtype=bool); RQ=(np.outer(v,v)+2*S*S)/3; lq=np.sum(Q[o_]*RQ[o_])/np.sum(RQ[o_]**2); RK=v[:,None]*S; lk=np.sum(K31[o_]*RK[o_])/np.sum(RK[o_]**2)
    r2q=1-np.sum((Q[o_]-lq*RQ[o_])**2)/np.sum(Q[o_]**2); r2k=1-np.sum((K31[o_]-lk*RK[o_])**2)/np.sum(K31[o_]**2)
    print(l,'lam table %.4f  fitted lamQ %.4f (R2 %.3f) lamK %.4f (R2 %.3f)'%(lam[l],lq,r2q,lk,r2k))
    for nm,(QQ,KK) in [('true',(Q,K31)),('regen tab',(Qr,Kr)),('noQK off',(np.diag(np.diag(Q)),np.diag(np.diag(K31))))]:
        mu,C,G,Mx,K31h,aux=pcp3.transfer3(m,S,T,QQ,KK,6,dict(base,_layer=l))
        E=(C-Ct)/np.outer(sd,sd); dv=np.einsum('ji,jk,ki->i',w,C-Ct,w)
        print('  %-10s off err %.3e  next var rel %.3e'%(nm,np.sqrt(np.mean(E[off]**2)),np.sqrt(np.mean((dv/v2)**2))),flush=True)
