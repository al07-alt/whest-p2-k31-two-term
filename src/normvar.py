import numpy as np, time, sys
from lib import *
i=int(sys.argv[1]) if len(sys.argv)>1 else 0
Wt=W(i,np.float32); n=1024; N=int(sys.argv[2]) if len(sys.argv)>2 else 60000; B=10000
rng=np.random.default_rng(7)
R2=np.zeros((N,17))  # ||h_l||^2/n for l=-1(input)..15
t=time.time()
for b in range(N//B):
    x=rng.standard_normal((B,n)).astype(np.float32); h=x
    R2[b*B:(b+1)*B,0]=np.mean(x.astype(np.float64)**2,1)
    for l in range(16):
        h=np.maximum(h@Wt[l],0); R2[b*B:(b+1)*B,l+1]=np.mean(h.astype(np.float64)**2,1)
np.save(f'data/normR2_{i:03d}.npy',R2)
V=R2.var(0)/R2.mean(0)**2; Vl=np.log(R2).var(0)
print('%.0fs'%(time.time()-t)); print('V=Var(r2)/E^2 :',np.round(V,4)); print('Var log r2    :',np.round(Vl,4))
