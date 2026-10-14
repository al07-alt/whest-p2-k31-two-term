import numpy as np, pcp3
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
PM=np.load('data/kp/M11_000.npy',mmap_mode='r')
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=True,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,edgecoef=True))
o_=~np.eye(1024,dtype=bool)
for l in [5,8,11,14]:
    m,S,T,Q,K31=state(l); pm=K['mean'][l]; Ct=PM[l].astype(float)-np.outer(pm,pm)
    w=Wt[l+1]; v2=np.einsum('ji,jk,ki->i',w,Ct,w); v=np.diag(S); s=np.sqrt(v)
    Ko=K31*o_; So=S*o_; To=T*o_; k4=np.diag(Q)
    lam_i=np.sum(np.sum(Ko*So,1)/np.sum(So*So,1)*k4/v)/np.sum((k4/v)**2)
    Kk4=(lam_i*k4/v)[:,None]*So
    feats=[Kk4,m[:,None]*To,To*s[:,None],To.T*s[None,:],So*(np.diag(T)/s)[:,None],So*(np.diag(T)/s)[None,:]]
    X=np.stack([f[o_] for f in feats],1); b=np.linalg.lstsq(X,Ko[o_],rcond=None)[0]; KT=sum(bb*f for bb,f in zip(b,feats))
    lq=np.sum(Q[o_]*((np.outer(v,v)+2*S*S)/3)[o_])/np.sum((((np.outer(v,v)+2*S*S)/3)[o_])**2); Qr=lq*(np.outer(v,v)+2*S*S)/3; np.fill_diagonal(Qr,k4)
    for nm,QQ,KK in [('trueQK',Q,K31),('trueQ,K=k4row',Q,Kk4),('trueQ,K=k4row+T',Q,KT),('radQ,K=k4row+T',Qr,KT),('radQ,K=k4row',Qr,Kk4),('radQ,K=0',Qr,0*K31)]:
        KK=KK.copy(); np.fill_diagonal(KK,np.diag(K31))
        mu,C,G,Mx,K31h,aux=pcp3.transfer3(m,S,T,QQ,KK,6,dict(base,_layer=l))
        dv=np.einsum('ji,jk,ki->i',w,C-Ct,w)
        print(l,'%-18s next var rel %.3e'%(nm,np.sqrt(np.mean((dv/v2)**2))),flush=True)
