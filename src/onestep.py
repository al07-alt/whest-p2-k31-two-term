import numpy as np, pcp3, sys
from lib import *
import os
i=int(os.environ.get('NET','0')); Wt=W(i)
K=np.load(f'data/kp/marg_{i:03d}.npz')
M11=np.load(f'data/kp/pre_M11_{i:03d}.npy',mmap_mode='r'); M21=np.load(f'data/kp/pre_M21_{i:03d}.npy',mmap_mode='r')
M22=np.load(f'data/kp/pre_M22_{i:03d}.npy',mmap_mode='r'); M31=np.load(f'data/kp/pre_M31_{i:03d}.npy',mmap_mode='r')
def state(l):
    a=K['pre_mean'][l].astype(float); X2=K['pre_m2'][l]; X3=K['pre_m3'][l]
    m11=M11[l].astype(float); m21=M21[l].astype(float); m22=M22[l].astype(float); m31=M31[l].astype(float)
    A=a[:,None]; B=a[None,:]
    S=m11-A*B
    T=m21-B*X2[:,None]-2*A*m11+2*A*A*B
    c22=m22-2*B*m21-2*A*m21.T+B*B*X2[:,None]+A*A*X2[None,:]+4*A*B*m11-3*A*A*B*B
    v=np.diag(S); Q=c22-np.outer(v,v)-2*S*S
    c31=m31-B*X3[:,None]-3*A*m21+3*A*B*X2[:,None]+3*A*A*m11-3*A**3*B
    K31=c31-3*v[:,None]*S
    return a,S,T,Q,K31
opts=dict(pcp3.OPT); opts.update(dict(mixT=True,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False))
variants=sys.argv[1].split(';') if len(sys.argv)>1 else ['']
for l in [2,5,8,11,14]:
    m,S,T,Q,K31=state(l); m2,S2,T2,Q2,K312=state(l+1); w=Wt[l+1]
    for vnm in variants:
        o=dict(opts,_layer=l); TT,QQ,KK=T,Q,K31
        if 'noToff' in vnm: TT=np.diag(np.diag(T))
        if 'noQoff' in vnm: QQ=np.diag(np.diag(Q))
        if 'noKoff' in vnm: KK=np.diag(np.diag(K31))
        if 'noT' == vnm: TT=0*T
        mu,C,G,Mx,K31h,aux=pcp3.transfer3(m,S,TT,QQ,KK,6,o)
        Sp=w.T@C@w; v2=np.diag(S2); mp=w.T@mu
        rv=np.sqrt(np.mean((np.diag(Sp)/v2-1)**2)); off=~np.eye(1024,dtype=bool)
        ro=np.sqrt(np.mean(((Sp-S2)/np.sqrt(np.outer(v2,v2)))[off]**2))
        rm=np.sqrt(np.mean(((mp-m2)/np.sqrt(v2))**2))
        print(l,'->',l+1,'%-8s'%vnm,'mean err/s %.2e  var rel %.2e  offdiag rho err %.2e'%(rm,rv,ro),flush=True)
