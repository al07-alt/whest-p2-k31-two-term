import numpy as np
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
R=np.load('data/rec0_full.npz'); o_=~np.eye(1024,dtype=bool)
for l in [2,4,6,8,10,12,14,15]:
    m,S,T,Q,K31=state(l); Tc=R['T'][l].astype(float); Sc=R['S'][l].astype(float)
    y=T[o_]; x=Tc[o_]; b=np.sum(x*y)/np.sum(x*x); r=np.corrcoef(x,y)[0,1]
    yd=np.diag(T); xd=np.diag(Tc); bd=np.sum(xd*yd)/np.sum(xd*xd)
    bs=np.sum(Sc[o_]*S[o_])/np.sum(Sc[o_]**2); bv=np.sum(np.diag(Sc)*np.diag(S))/np.sum(np.diag(Sc)**2)
    print(l,'T_true ~ b*T_chain: off b %.4f corr %.4f | diag b %.4f | S off b %.4f diag b %.5f'%(b,r,bd,bs,bv))
