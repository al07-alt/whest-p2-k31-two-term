import numpy as np
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
R=np.load('data/rec0_full.npz'); o_=~np.eye(1024,dtype=bool)
for l in [4,6,8,10,12,14,15]:
    m,S,T,Q,K31=state(l); Tc=R['T'][l].astype(float); Sc=R['S'][l].astype(float); mc=R['m'][l]; vc=np.diag(Sc)
    D=(T-Tc)[o_]; R1=(2*mc[:,None]*Sc+mc[None,:]*vc[:,None])[o_]; R2=np.outer(mc*mc,mc)[o_]
    tot=np.sum(D**2)
    for nm,X in [('scale T',Tc[o_][:,None]),('radial',R1[:,None]),('radial+m3',np.stack([R1,R2],1)),('scale+radial',np.stack([Tc[o_],R1],1))]:
        b=np.linalg.lstsq(X,D,rcond=None)[0]; r2=1-np.sum((D-X@b)**2)/tot
        print(l,'%-13s explains %.3f of chain T error energy, coef %s'%(nm,r2,np.round(b,5)))
