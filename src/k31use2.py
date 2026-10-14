import numpy as np, pcp3, sys
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
BK=np.load('data/bk/marg_000.npz'); l=9
Kb=np.load(f'data/k31b_{l}.npy').astype(float); Kt=np.load(f'data/tree31_{l}.npy').astype(float)
m,S,T,Q,K31=state(l); v=np.diag(S); o_=~np.eye(1024,dtype=bool); RK=v[:,None]*S; y=K31[o_]
b1=np.sum(RK[o_]*y)/np.sum(RK[o_]**2)
X=np.stack([RK[o_],Kt[o_],Kb[o_]],1); b=np.linalg.lstsq(X,y,rcond=None)[0]
# radial coef refit with fixed theory coefs 1,1
r=y-Kt[o_]-Kb[o_]; bth=np.sum(RK[o_]*r)/np.sum(RK[o_]**2)
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=True,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,edgecoef=True,_layer=l))
w=Wt[l+1]; vi=BK['pre_var'][l+1]
for nm,KK in [('true',K31),('radial',b1*RK),('rad+tree+births fit',b[0]*RK+b[1]*Kt+b[2]*Kb),('rad(refit)+tree+births theory',bth*RK+Kt+Kb)]:
    KK=KK.copy(); np.fill_diagonal(KK,np.diag(K31))
    mu,C,G,Mx,_,_=pcp3.transfer3(m,S,T,Q,KK,6,base); vp=np.einsum('ji,jk,ki->i',w,C,w)
    print('%-32s next var err %.3e'%(nm,np.sqrt(np.mean((vp/vi-1)**2))),flush=True)
print('radial coef alone %.4f, with theory births %.4f'%(b1,bth))
