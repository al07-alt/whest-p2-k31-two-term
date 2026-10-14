# V29's (3,1) coefficient relative to the least-squares radial coefficient of the true slice (net 0).
# diag slopes: regression slope of V29's slot on 0.5*K^T (EF_K31OR_PRINT run, li 1..14); the slope on K^T is half.
import numpy as np
from truek31 import k31
diag=[0.275,0.474,0.625,0.713,0.758,0.765,0.757,0.757,0.743,0.727,0.709,0.697,0.665,0.657]
K=np.load('data/kp/marg_000.npz')
for l in range(1,15):
    Kt=k31(0,l); M11=np.load('data/kp/pre_M11_000.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); off=~np.eye(1024,dtype=bool)
    y=Kt.T[off]; x=S[off]          # V29 form: K(j^3,i) ~ c * C_ij  (no v_j)
    b=(x@y)/(x@x); R2=1-np.sum((y-b*x)**2)/np.sum((y-y.mean())**2)
    sv=diag[l-1]/2
    print('l=%2d V29 slope on K %.3f | R2 of c*C form %.3f | V29 coef / LS coef %.2f'%(l,sv,R2,sv/R2),flush=True)
