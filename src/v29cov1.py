# Offline replica of V29's (1,1) post-activation covariance program, fed with the TRUE pre-activation state,
# to settle the convention of the (3,1) slot (wk431 = f * K^T) against the true post-activation covariance.
import numpy as np, sys, ast, re
from truek31 import k31, t21
from lib import W
src=open('./v29_experiments/estimator_v29_ef.py').read()
def grab(name):
    i=src.index(name+' = '); j=i+len(name)+3; depth=0; k=j
    while True:
        c=src[k]
        if c in '[{(': depth+=1
        elif c in ']})': depth-=1
        k+=1
        if depth==0 and c in ']})': break
    return ast.literal_eval(src[j:k])
WP=grab('WICK_PAIRS'); WU=grab('WICK_UNIFIED'); TS=grab('TERM_SPECS')
I=WP.index
TS[(1,1)].append(('wk431','ones2',I((1,1)),I((3,1)),1/3))
from scipy.special import ndtr
pdf=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
i=int(sys.argv[1]); Ls=[int(x) for x in sys.argv[2].split(',')]
Kz=np.load(f'./data/kp/marg_{i:03d}.npz'); Wt=W(i)
M11p=np.load(f'./data/kp/M11_{i:03d}.npy',mmap_mode='r'); M11=np.load(f'./data/kp/pre_M11_{i:03d}.npy',mmap_mode='r')
M21=np.load(f'./data/kp/pre_M21_{i:03d}.npy',mmap_mode='r'); M22=np.load(f'./data/kp/pre_M22_{i:03d}.npy',mmap_mode='r')
for l in Ls:
    m=Kz['pre_mean'][l].astype(float); X2=Kz['pre_m2'][l]; X3=Kz['pre_m3'][l]; X4=Kz['pre_m4'][l]
    A=m[:,None]; B=m[None,:]; S=M11[l].astype(float)-A*B; v=np.diag(S).copy(); s=np.sqrt(v); a=m/s
    m21=M21[l].astype(float); m22=M22[l].astype(float); m11=M11[l].astype(float)
    T=m21-B*X2[:,None]-2*A*m11+2*A*A*B
    c22=m22-2*B*m21-2*A*m21.T+B*B*X2[:,None]+A*A*X2[None,:]+4*A*B*m11-3*A*A*B*B
    Q=c22-np.outer(v,v)-2*S*S
    k3=X3-3*m*X2+2*m**3; k4=X4-4*m*X3+6*m*m*X2-3*m**4-3*v*v
    K=k31(i,l)
    ph=pdf(a); Ph=ndtr(a)
    Wv=[]
    for (const,e,p1,p2) in WU:
        Wv.append(const*s**e*(sum(c*a**k for k,c in enumerate(p1))*ph+sum(c*a**k for k,c in enumerate(p2))*Ph))
    Wv=np.array(Wv)
    off=~np.eye(len(m),dtype=bool)
    Coff=S.copy(); np.fill_diagonal(Coff,0); Qo=Q.copy(); np.fill_diagonal(Qo,0)
    base={'ones2':np.ones_like(S),'c_off':Coff,'d21':T,'d21T':T.T,'d3col':np.repeat(k3[:,None],len(m),1),'d3row':np.repeat(k3[None,:],len(m),0),
          'g4col':np.repeat(k4[:,None],len(m),1),'g4row':np.repeat(k4[None,:],len(m),0),'wk4m':Qo}
    mp=Kz['mean'][l].astype(float); Ct=M11p[l].astype(float)-np.outer(mp,mp)
    w=Wt[l+1] if l<15 else None
    res=[]
    for f in (0.0,0.5,1.0):
        b=dict(base); Kt=K.T.copy(); np.fill_diagonal(Kt,0); b['wk431']=f*Kt
        PK=np.zeros_like(S)
        for (x,y,wl,wr,coef) in TS[(1,1)]:
            Ax=b[x]*b[y] if (x!='ones2' and y!='ones2') else (b[y] if x=='ones2' else b[x])
            PK+=coef*Ax*Wv[wl][:,None]*Wv[wr][None,:]
        pk=0.5*(PK+PK.T); np.fill_diagonal(pk,np.diag(Ct))   # true diagonal: isolate the off-diagonal program
        e=pk-Ct
        r1=np.linalg.norm(e[off])/np.linalg.norm(Ct[off])
        if w is not None:
            vt=np.einsum('ji,jk,ki->i',w,Ct,w); vp=np.einsum('ji,jk,ki->i',w,pk,w); r2=np.sqrt(np.mean((vp/vt-1)**2))
        else: r2=np.nan
        res.append('f=%.1f: offdiag rel err %.3e | next-var rel err %.3e'%(f,r1,r2))
    print('net %d l=%d | '%(i,l)+' | '.join(res),flush=True)
