import numpy as np, sys
from scipy.special import ndtr
from truek31 import k31
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz')
pdf=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
for l in Ls:
    Kt=k31(i,l)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); S=M11-np.outer(m,m); v=np.diag(S).copy(); a=m/np.sqrt(v)
    np.fill_diagonal(S,0); Kd=Kt.copy(); np.fill_diagonal(Kd,0)
    R=v[:,None]*S
    rowslope=(Kd*R).sum(1)/(R*R).sum(1)
    bins=[-4,-2.5,-1.5,-0.75,0,0.75,1.5,2.5,4]
    out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        sel=(a>=lo)&(a<hi)
        if sel.sum()<5: out.append('  [%+.1f,%+.1f) n=%d'%(lo,hi,sel.sum())); continue
        sl=(Kd[sel]*R[sel]).sum()/(R[sel]**2).sum()
        out.append('[%+.2f,%+.2f) n=%3d slope %.4f'%(lo,hi,sel.sum(),sl))
    gl=(Kd*R).sum()/(R*R).sum()
    # fit row slope ~ c0 + c1 a + c2 a^2 (weighted by row energy)
    wr=(R*R).sum(1); X=np.stack([np.ones_like(a),a,a*a],1); b=np.linalg.lstsq(X*np.sqrt(wr)[:,None],rowslope*np.sqrt(wr),rcond=None)[0]
    print('l=%d global slope %.4f | row-slope fit c0 %.4f c1 %.4f c2 %.4f'%(l,gl,*b))
    for o in out: print('    ',o)
