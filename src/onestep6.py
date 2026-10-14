import numpy as np, pcp3, sys
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
lam=np.load('data/lam_table.npy')
base=dict(pcp3.OPT); base.update(dict(mixT=True,edgecoef=True))
o_=~np.eye(1024,dtype=bool)
for l in [4,8,12,14]:
    m,S,T,Q,K31=state(l); m2,S2,T2,Q2,K312=state(l+1); w=Wt[l+1]; v2=np.diag(S2)
    nrm=np.outer(v2**1.5,v2**0.5)
    for nm,o in [('lin',{}),('lin+wick',dict(q4wick=True,k31wick=True))]:
        oo=dict(base,_layer=l,**o)
        mu,C,G,Mx,K31h,aux=pcp3.transfer3(m,S,T,Q,K31,6,oo)
        Qn,Kn=pcp3.linear_q_k31(w,Mx,K31h,oo,S,aux)
        RK=v2[:,None]*S2
        e=lambda X: np.sqrt(np.mean(((X-K312)/nrm)[o_]**2))/np.sqrt(np.mean((K312/nrm)[o_]**2))
        lk=np.sum(K312[o_]*RK[o_])/np.sum(RK[o_]**2)
        # best combo: a*Kn + b*RK
        X=np.stack([Kn[o_],RK[o_]],1); b=np.linalg.lstsq(X,K312[o_],rcond=None)[0]
        print(l,'%-9s K31 rel err: chain-closure %.3f | radial-only(fit lam %.4f) %.3f | combo %s %.3f'%(nm,e(Kn),lk,e(lk*RK),np.round(b,3),e(b[0]*Kn+b[1]*RK)),flush=True)
