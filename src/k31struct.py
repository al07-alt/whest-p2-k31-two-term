import numpy as np
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
o_=~np.eye(1024,dtype=bool)
for l in [4,8,12,14]:
    m,S,T,Q,K31=state(l); v=np.diag(S); s=np.sqrt(v)
    Ko=K31*o_; So=S*o_; To=T*o_
    tot=np.sum(Ko**2)
    c=np.sum(Ko*So,1)/np.sum(So*So,1); r2row=1-np.sum((Ko-c[:,None]*So)**2)/tot
    d=np.sum(Ko*So,0)/np.sum(So*So,0); r2col=1-np.sum((Ko-So*d[None,:])**2)/tot
    k4=np.diag(Q)  # diag of Q = marginal kappa4
    cc=np.corrcoef(c,k4/v)[0,1]
    lam_i=np.sum(c*k4/v)/np.sum((k4/v)**2)
    r2k4=1-np.sum((Ko-(lam_i*k4/v)[:,None]*So)**2)/tot
    # T-based structures
    feats={'m_i T_ij':m[:,None]*To,'v_i T_ij/s?':To*s[:,None],'T_ji*s_j':To.T*s[None,:],'S_ij*k3_i/s_i':So*(np.diag(T)/s)[:,None],'S_ij*k3_j/s_j':So*(np.diag(T)/s)[None,:]}
    X=np.stack([(c[:,None]*So)[o_]]+[f[o_] for f in feats.values()],1); b=np.linalg.lstsq(X,Ko[o_],rcond=None)[0]
    r2all=1-np.sum((Ko[o_]-X@b)**2)/tot
    print(l,'R2 row-scale c_i S_ij %.3f | col-scale %.3f | c_i vs k4_i/v_i corr %.3f, R2 using k4_i/v_i %.3f | +T feats %.3f'%(r2row,r2col,cc,r2k4,r2all),flush=True)
