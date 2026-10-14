import numpy as np, sys
from truek31 import k31, t21
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz')
for l in Ls:
    Kt=k31(i,l); T,k3,v=t21(i,l)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m)
    off=~np.eye(len(m),dtype=bool); Rd=v[:,None]*S
    b=(Kt[off]*Rd[off]).sum()/(Rd[off]**2).sum(); Rz=Kt-b*Rd; np.fill_diagonal(Rz,0)
    sv=np.linalg.svd(Rz,compute_uv=False); e=np.cumsum(sv**2)/np.sum(sv**2)
    Ks=Kt.copy(); np.fill_diagonal(Ks,0); svk=np.linalg.svd(Ks,compute_uv=False); ek=np.cumsum(svk**2)/np.sum(svk**2)
    # symmetric/antisymmetric split of residual
    sym=(Rz+Rz.T)/2; asym=(Rz-Rz.T)/2
    # random-matrix reference: energy share of top-k for iid noise of same shape
    print('l=%d |R|/|K| %.2f | R top-1/4/16/64/256 energy %.2f %.2f %.2f %.2f %.2f | K: %.2f %.2f %.2f %.2f %.2f | |sym|^2/|R|^2 %.2f'%(
        l,np.linalg.norm(Rz)/np.linalg.norm(Ks),e[0],e[3],e[15],e[63],e[255],ek[0],ek[3],ek[15],ek[63],ek[255],np.sum(sym**2)/np.sum(Rz**2)))
