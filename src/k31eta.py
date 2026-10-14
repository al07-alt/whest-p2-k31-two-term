import numpy as np, sys
from truek31 import k31
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
K=np.load(f'data/kp/marg_{i:03d}.npz')
for l in Ls:
    Kt=k31(i,l)
    M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')[l].astype(float)
    m=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]; X4=K['pre_m4'][l]
    S=M11-np.outer(m,m); v=np.diag(S).copy(); a=m/np.sqrt(v)
    k4=X4-4*m*X3+6*m*m*X2-3*m**4-3*v*v; eta=k4/v**2
    np.fill_diagonal(S,0); Kd=Kt.copy(); np.fill_diagonal(Kd,0)
    R=v[:,None]*S; wr=(R*R).sum(1); rs=(Kd*R).sum(1)/wr
    X=np.stack([np.ones_like(a),eta],1); b=np.linalg.lstsq(X*np.sqrt(wr)[:,None],rs*np.sqrt(wr),rcond=None)[0]
    lam_perp=np.median(eta[np.abs(a)<0.5])
    pred=(lam_perp+eta)/2
    print('l=%d corr(rowslope,eta) %.3f fit rowslope = %.4f + %.3f eta | theory (lam_perp+eta)/2: lam_perp %.4f, weighted R2 of theory %.3f, of const %.3f'%(
        l,np.corrcoef(rs,eta)[0,1],b[0],b[1],lam_perp,
        1-np.sum(wr*(rs-pred)**2)/np.sum(wr*(rs-np.average(rs,weights=wr))**2),
        1-np.sum(wr*(rs-np.average(rs,weights=wr))**2)/np.sum(wr*(rs-np.average(rs,weights=wr))**2)))
    # entry-level R2 on K for models: global radial, eta-radial, (lam_perp+eta)/2 radial
    off=~np.eye(len(m),dtype=bool); y=Kd[off]
    for nm,Mk in [('glob',R*((Kd*R).sum()/(R*R).sum())),('eta',eta[:,None]*R),('half',pred[:,None]*R)]:
        print('    %-5s entry R2 %.3f  slope %.3f'%(nm,1-np.mean((y-Mk[off])**2)/np.var(y),(y*Mk[off]).sum()/(Mk[off]**2).sum()))
