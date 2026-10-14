# b1 refit with b2 fixed at 1: K - kappa3_a T_aab / v_a ~ b1 v_a S_ab (off-diagonal LS), per layer, nets 0-2
import numpy as np
from truek31 import k31, t21
B=np.zeros((3,14))
for i in (0,1,2):
    K=np.load(f'/home/claude/data/kp/marg_{i:03d}.npz')
    for l in range(1,15):
        Kt=k31(i,l); T,k3,v=t21(i,l)
        M11=np.load(f'/home/claude/data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
        m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); off=~np.eye(len(m),dtype=bool)
        y=(Kt-(k3/v)[:,None]*T)[off]; x=(v[:,None]*S)[off]; B[i,l-1]=(x@y)/(x@x)
    print(i,np.round(B[i],5),flush=True)
print('mean',[round(float(b),5) for b in B.mean(0)])
