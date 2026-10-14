import numpy as np
from scipy.special import ndtr
from lib import *
Ws=W(0); F=np.load('data/feat/f000.npz'); Phi=ndtr(F['m']/np.sqrt(F['v']))
D=np.diag(Phi[15])
for l in range(14,-1,-1):
    D=(D@Ws[l+1].T)*Phi[l][None,:]
    if l in (12,8,4,1):
        sv=np.linalg.svd(D,compute_uv=False); e=np.cumsum(sv**2)/np.sum(sv**2)
        print('l',l,'energy in top k:',{k:round(float(e[k-1]),3) for k in (8,32,128,256,384,512)})
