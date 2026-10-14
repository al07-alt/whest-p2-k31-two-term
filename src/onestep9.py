import numpy as np, pcp3
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
PM=np.load('data/kp/M11_000.npy',mmap_mode='r'); R=np.load('data/rec0_full.npz')
base=dict(pcp3.OPT); base.update(dict(mixT=True,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False))
o_=~np.eye(1024,dtype=bool)
for l in [4,8,12,14]:
    m,S,T,Q,K31=state(l); Tc=R['T'][l].astype(float); Sc=R['S'][l].astype(float); mc=R['m'][l]
    e=np.linalg.norm((Tc-T)[o_])/np.linalg.norm(T[o_]); ed=np.linalg.norm(np.diag(Tc-T))/np.linalg.norm(np.diag(T))
    eS=np.linalg.norm((Sc-S)[o_])/np.linalg.norm(S[o_])
    pm=K['mean'][l]; Ct=PM[l].astype(float)-np.outer(pm,pm); w=Wt[l+1]; vv=np.einsum('ji,jk,ki->i',w,Ct,w)
    print(l,'chain T rel err off %.3f diag %.3f | chain S off rel err %.4f'%(e,ed,eS))
    for nm,TT,SS,mm in [('true',T,S,m),('chainT',Tc,S,m),('chainT-offonly',T*(~o_)+Tc*o_,S,m),('chainS',T,Sc,m),('chainS,T,m',Tc,Sc,mc)]:
        mu,C,G,Mx,_,_=pcp3.transfer3(mm,SS,TT,Q,K31,6,dict(base,_layer=l))
        dv=np.einsum('ji,jk,ki->i',w,C-Ct,w); print('   %-15s next var rel %.3e'%(nm,np.sqrt(np.mean((dv/vv)**2))),flush=True)
