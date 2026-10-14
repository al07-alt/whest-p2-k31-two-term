import numpy as np, sys
from truek31 import k31
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz')
for l in Ls:
    Kt=k31(i,l)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); v=np.diag(S).copy()
    off=~np.eye(len(m),dtype=bool); Rd=v[:,None]*S
    b=(Kt[off]*Rd[off]).sum()/(Rd[off]**2).sum(); R=Kt-b*Rd; np.fill_diagonal(R,0)
    U,s,Vt=np.linalg.svd(R)
    ev,EV=np.linalg.eigh(S); top=EV[:,::-1][:,:16]
    mh=m/np.linalg.norm(m); one=np.ones(len(m))/np.sqrt(len(m))
    def frac(vec,Mat): return (vec@Mat)**2
    print('l=%d s1^2 share %.2f | right sv1 . mhat %.2f . 1 %.2f | left sv1 . mhat %.2f . 1 %.2f | right sv1 energy in top-16 eig of S %.2f | sv2: right.mhat %.2f left.mhat %.2f'%(
        l,s[0]**2/np.sum(s**2),abs(Vt[0]@mh),abs(Vt[0]@one),abs(U[:,0]@mh),abs(U[:,0]@one),np.sum((top.T@Vt[0])**2),abs(Vt[1]@mh),abs(U[:,1]@mh)))
    # energy of R in the rank-2 space spanned by m on the right (R m m^T/|m|^2) and on the left
    Pm=np.outer(mh,mh)
    print('     |R Pm|^2/|R|^2 %.2f  |Pm R|^2/|R|^2 %.2f  | R energy in top-16 eig of S (right) %.2f (left) %.2f'%(np.sum((R@Pm)**2)/np.sum(R**2),np.sum((Pm@R)**2)/np.sum(R**2),np.sum((R@top)**2)/np.sum(R**2),np.sum((top.T@R)**2)/np.sum(R**2)))
