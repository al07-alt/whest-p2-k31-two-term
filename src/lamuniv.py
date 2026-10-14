import numpy as np
L=[]
for i in range(30):
    z=np.load(f'data/kp/marg_{i:03d}.npz')
    r1,r2,r3,r4=z['pre_mean'],z['pre_m2'],z['pre_m3'],z['pre_m4']
    c2=r2-r1**2; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
    x=c2**2; L.append(np.sum(x*c4,1)/np.sum(x*x,1))
L=np.array(L)
print('mean lam per layer', np.round(L.mean(0),4))
print('sd   lam per layer', np.round(L.std(0),4))
print('rel sd', np.round(L.std(0)/L.mean(0),3))
np.save('data/lam_table.npy',L.mean(0))
