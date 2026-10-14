# Partial correlation of the v3 single-latent skewness prediction with the truth, given alpha (report §4.3).
import numpy as np
for i,tag in [(0,''),(1,'1_'),(2,'2_')]:
    K=np.load(f'data/kp/marg_{i:03d}.npz')
    for l in (8,15):
        d=np.load(f'data/latent_{tag}{l}.npz')
        m=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]; v=X2-m*m; k3=X3-3*m*X2+2*m**3
        g=k3/v**1.5; a=m/np.sqrt(v); gp=d['pred3']/v**1.5
        def res(y,x): X=np.stack([np.ones_like(x),x],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; return y-X@b
        print('net',i,'l',l,'corr latent %.3f alpha %.3f partial(latent|alpha) %.3f'%(np.corrcoef(g,gp)[0,1],np.corrcoef(g,a)[0,1],np.corrcoef(res(g,a),res(gp,a))[0,1]))
