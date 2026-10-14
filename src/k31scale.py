import numpy as np
from lib import *
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
o_=~np.eye(1024,dtype=bool)
LAM=[4.9895e-03, 8.0876e-03, 9.8291e-03, 1.0549e-02, 1.0851e-02, 1.0828e-02, 1.0589e-02, 1.0048e-02, 9.6483e-03, 9.1720e-03, 8.7730e-03, 8.3938e-03, 8.0555e-03, 7.6770e-03, 7.2588e-03, 7.2588e-03]
for l in [2,4,6,8,10,12,14,15]:
    m,S,T,Q,K31=state(l); v=np.diag(S)
    # wk431[i,j] = kappa(z_i,z_j,z_j,z_j) = K31[j,i]
    Y=K31.T[o_]; X=S[o_]; lam_eff=np.sum(X*Y)/np.sum(X*X); r2=1-np.sum((Y-lam_eff*X)**2)/np.sum(Y**2)
    print(l,'mean var %.3f | K31 ~ lam*C_off: lam_eff %.4f (R2 %.3f) | v29 LAM[l-1]*.95 %.4f LAM[l]*.95 %.4f'%(v.mean(),lam_eff,r2,LAM[l-1]*.95,LAM[l]*.95))
