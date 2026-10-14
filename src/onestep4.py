import numpy as np, pcp3, sys
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
PM=np.load('data/kp/M11_000.npy',mmap_mode='r')
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False))
V=[('base',{}),('k31x',dict(k31_cross=True)),('edge',dict(edgecoef=True)),('edge+k31x',dict(edgecoef=True,k31_cross=True)),('edge.5+k31x',dict(edgecoef=True,k31_cross=True,edge_alpha=0.5)),('K8+edge+k31x',dict(edgecoef=True,k31_cross=True,_K=8))]
for l in [2,5,8,11,14]:
    m,S,T,Q,K31=state(l); pm=K['mean'][l]; Ct=PM[l].astype(float)-np.outer(pm,pm)
    ok=np.diag(Ct)>1e-3*np.median(np.diag(Ct)); sd=np.sqrt(np.maximum(np.diag(Ct),1e-12)); off=~np.eye(1024,dtype=bool)&np.outer(ok,ok)
    w=Wt[l+1]; v2=np.einsum('ji,jk,ki->i',w,Ct,w)
    for nm,o in V:
        oo=dict(base,_layer=l,**o); Kh=oo.pop('_K',6)
        mu,C,G,Mx,K31h,aux=pcp3.transfer3(m,S,T,Q,K31,Kh,oo)
        E=(C-Ct)/np.outer(sd,sd); dv=np.einsum('ji,jk,ki->i',w,C-Ct,w)
        print(l,'%-14s off err %.3e  next var rel %.3e'%(nm,np.sqrt(np.mean(E[off]**2)),np.sqrt(np.mean((dv/v2)**2))),flush=True)
