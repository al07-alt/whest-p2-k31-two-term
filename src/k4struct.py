import numpy as np
V=np.load('data/normR2_000.npy'); Vv=V.var(0)/V.mean(0)**2
for i in [0,1,2]:
    z=np.load(f'data/kp/marg_{i:03d}.npz')
    r1,r2,r3,r4=z['pre_mean'],z['pre_m2'],z['pre_m3'],z['pre_m4']
    c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
    for l in ([4,8,12,15] if i==0 else [8,15]):
        v=c2[l]; m=r1[l]; y=c4[l]
        X=np.stack([v*v, v*m*m, m**4],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; r2_=1-np.sum((y-X@b)**2)/np.sum(y*y)
        b1=np.sum(v*v*y)/np.sum(v**4); r1_=1-np.sum((y-b1*v*v)**2)/np.sum(y*y)
        g3=c3[l]/v**1.5
        X2=np.stack([v*v, v*v*g3, v*v*g3**2],1); b2=np.linalg.lstsq(X2,y,rcond=None)[0]; r22=1-np.sum((y-X2@b2)**2)/np.sum(y*y)
        print(i,l,'R2 lam*v^2 %.3f (lam %.4f, 3V_prev %.4f) | +v m^2, m^4: %.3f %s | v^2(1,g3,g3^2): %.3f %s'%(r1_,b1,3*Vv[l],r2_,np.round(b,4),r22,np.round(b2,4)))
