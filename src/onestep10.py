import numpy as np, pcp3
from lib import *
exec(open('onestep.py').read().split("opts=dict")[0])
PM=np.load(f'./data/kp/M11_{i:03d}.npy',mmap_mode='r'); BK=np.load(f'./data/bk/marg_{i:03d}.npz')
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False))
V=[('base',{}),('edge+k31x',dict(edgecoef=True,k31_cross=True))]
o_=~np.eye(1024,dtype=bool)
for l in [2,5,8,11,14]:
    m,S,T,Q,K31=state(l); pm=K['mean'][l]; Ct=PM[l].astype(float)-np.outer(pm,pm)
    w=Wt[l+1]; v_same=np.einsum('ji,jk,ki->i',w,Ct,w); v_ind=BK['pre_var'][l+1]
    print(l,'same-sample vs independent next-var truth: rel diff %.2e'%np.sqrt(np.mean((v_same/v_ind-1)**2)))
    v=np.diag(S)
    lk=np.sum(K31[o_]*(v[:,None]*S)[o_])/np.sum(((v[:,None]*S)[o_])**2); Krad=lk*v[:,None]*S; np.fill_diagonal(Krad,np.diag(K31))
    for nm,o in V+[('edge+k31rad',dict(edgecoef=True,k31_cross=True,_Krad=True))]:
        oo=dict(base,_layer=l,**o); kr=oo.pop('_Krad',False)
        mu,C,G,Mx,_,_=pcp3.transfer3(m,S,T,Q,Krad if kr else K31,6,oo)
        vp=np.einsum('ji,jk,ki->i',w,C,w)
        print('   %-12s err vs same-sample %.3e | vs independent %.3e'%(nm,np.sqrt(np.mean((vp/v_same-1)**2)),np.sqrt(np.mean((vp/v_ind-1)**2))),flush=True)
