import numpy as np, sys
from scipy.special import ndtr
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
o_=~np.eye(1024,dtype=bool)
for l in [int(x) for x in sys.argv[1].split(',')]:
    m,S,T,Q,K31=state(l); s=np.sqrt(np.diag(S)); a=m/s; p=pdf(a)
    f1=ndtr(a); f2=p/s; f3=-a*p/s**2
    w=Wt[l+1]; Co=S.copy(); np.fill_diagonal(Co,0)
    G1=Co@(f1[:,None]*w); V=f2[:,None]*w
    H=Co@(V*G1)
    P1=6*(H*V).T@G1; P2=6*(H*G1).T@V; S1=3*(f3[:,None]*w*G1*G1).T@G1; S2=(G1**3).T@(f3[:,None]*w)
    Kb=P1+P2+S1+S2
    m2,S2_,T2,Q2,K2=state(l+1); v2=np.diag(S2_); y=K2[o_]; RK=(v2[:,None]*S2_)[o_]
    b1=np.sum(RK*y)/np.sum(RK*RK); r1=1-np.sum((y-b1*RK)**2)/np.sum(y**2)
    out=[]
    for nm,F in [('tree',Kb),('paths',P1+P2),('stars',S1+S2)]:
        X=np.stack([RK,F[o_]],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=1-np.sum((y-X@b)**2)/np.sum(y**2)
        out.append('%s R2 %.3f coef %.3f |F|/|K| %.3f'%(nm,r,b[1],np.linalg.norm(F[o_])/np.linalg.norm(y)))
    # also relative to radial residual: does tree explain resid?
    extra=''
    try:
        if i!=0: raise FileNotFoundError
        Kc=np.load(f'data/k31b_{l+1}.npy').astype(float)
        X=np.stack([RK,Kb[o_],Kc[o_]],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=1-np.sum((y-X@b)**2)/np.sum(y**2)
        extra=' | radial+tree+k3births R2 %.3f coef %s'%(r,np.round(b,3))
    except FileNotFoundError: pass
    print(l+1,'radial R2 %.3f |'%r1,' | '.join(out),extra,flush=True)
    np.save(f'data/tree31_{l+1}.npy' if i==0 else f'data/tree31_{i:03d}_{l+1}.npy',Kb.astype(np.float32))
