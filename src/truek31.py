import numpy as np
_cache={}
def k31(i,l):
    """True pre-activation K31[i,j]=kappa(z_i,z_i,z_i,z_j) of mini net i at layer l (N=1e8 dumps), float64."""
    K=np.load(f'./data/kp/marg_{i:03d}.npz')
    M11=np.load(f'./data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    M21=np.load(f'./data/kp/pre_M21_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    M31=np.load(f'./data/kp/pre_M31_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    a=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]
    A=a[:,None]; B=a[None,:]; S=M11-A*B; v=np.diag(S)
    c31=M31-B*X3[:,None]-3*A*M21+3*A*B*X2[:,None]+3*A*A*M11-3*A**3*B
    return c31-3*v[:,None]*S

def t21(i,l):
    """True T[a,b]=kappa(z_a,z_a,z_b) and per-neuron kappa3, var of mini net i at pre-activation layer l."""
    K=np.load(f'./data/kp/marg_{i:03d}.npz')
    M11=np.load(f'./data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    M21=np.load(f'./data/kp/pre_M21_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]
    A=m[:,None]; B=m[None,:]; S=M11-A*B; v=np.diag(S).copy()
    T=M21-B*X2[:,None]-2*A*M11+2*A*A*B
    k3=X3-3*m*X2+2*m**3
    return T,k3,v
