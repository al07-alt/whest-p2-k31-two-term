import numpy as np, pcp3, sys
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
BK=np.load('data/bk/marg_000.npz')
l=int(sys.argv[1])   # K31 births file for layer l
Kb=np.load(f'data/k31b_{l}.npy').astype(float)
m,S,T,Q,K31=state(l); v=np.diag(S); o_=~np.eye(1024,dtype=bool)
# wick births from layer l-1 true state
mp,Sp,Tp,Qp,Kp=state(l-1)
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False))
_,_,_,_,_,auxp=pcp3.transfer3(mp,Sp,Tp,Qp,Kp,6,dict(base,_layer=l-1)); _,K4w=pcp3.wick4(Wt[l],Sp,auxp,True)
RK=v[:,None]*S; y=K31[o_]
X=np.stack([RK[o_],Kb[o_],K4w[o_]],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-np.sum((y-X@b)**2)/np.sum(y**2)
print(l,'R2 radial+births+wick %.3f coef %s'%(r2,np.round(b,4)))
X2=np.stack([RK[o_],Kb[o_]],1); b2=np.linalg.lstsq(X2,y,rcond=None)[0]
b1=np.sum(RK[o_]*y)/np.sum(RK[o_]**2)
w=Wt[l+1]; v_ind=BK['pre_var'][l+1]
opt=dict(base,edgecoef=True,k31_cross=True,_layer=l)
for nm,KK in [('true',K31),('radial',b1*RK),('rad+births(fit)',b2[0]*RK+b2[1]*Kb),('rad(tab)+births(1.0)',b1*RK+Kb)]:
    KK=KK.copy(); np.fill_diagonal(KK,np.diag(K31))
    mu,C,G,Mx,_,_=pcp3.transfer3(m,S,T,Q,KK,6,opt); vp=np.einsum('ji,jk,ki->i',w,C,w)
    print('   %-22s next var err vs independent %.3e'%(nm,np.sqrt(np.mean((vp/v_ind-1)**2))),flush=True)
mu,C,G,Mx,_,_=pcp3.transfer3(m,S,T,Q,K31,6,dict(base,_layer=l)); vp=np.einsum('ji,jk,ki->i',w,C,w)
print('   %-22s next var err vs independent %.3e'%('base (no edge/k31)',np.sqrt(np.mean((vp/v_ind-1)**2))))
