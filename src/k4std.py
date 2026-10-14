import numpy as np
for i in [0,1,2]:
    z=np.load(f'data/kp/marg_{i:03d}.npz')
    r1,r2,r3,r4=z['pre_mean'],z['pre_m2'],z['pre_m3'],z['pre_m4']
    c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
    out=[]
    for l in [4,8,12,15]:
        g3=c3[l]/c2[l]**1.5; g4=c4[l]/c2[l]**2
        X=np.stack([np.ones(1024),g3**2],1); b=np.linalg.lstsq(X,g4,rcond=None)[0]
        r2=1-np.var(g4-X@b)/np.var(g4)
        out.append('L%d exkurt mean %.4f sd %.4f | fit lam %.4f + c %.2f skew^2: R2(across neurons) %.2f'%(l,g4.mean(),g4.std(),b[0],b[1],r2))
    print('net',i); print('\n'.join(out))
