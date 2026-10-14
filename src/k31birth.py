import numpy as np, fac3, sys, time
from lib import *
from scipy.special import ndtr
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'onestep.py')).read().split("opts=dict")[0])
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
l=int(sys.argv[1]); t0=time.time()
fac3.OLD.clear(); fac3.run_fac3(W(0),opts=dict(EX,dump_srcs_at=l)); d=fac3.OLD[-1]
m=d['m']; S=d['S']; s=np.sqrt(np.diag(S)); a=m/s; f1=ndtr(a); f2=pdf(a)/s
w=Wt[l+1]; Wt_=w*f1[:,None]; G=Wt_.T@S        # G[i,a]
Hm=f2[:,None]*w*G.T                            # Hm[a,i] = f''_a W_ai G_ia
T1=np.zeros((1024,1024)); Lam=np.zeros((1024,1024))   # Lam[a,i] = Lambda_a(i,i)
for sr in d['srcs']:
    J,A,B,U,q=sr['J'],sr['A'],sr['B'],sr['U'],sr['q']; V=J*q[None,:]
    trip=[(J,J,B,1/3),(J,B,J,1/3),(B,J,J,1/3),(A,A,V,1.),(A,V,A,1.),(V,A,A,1.)]
    if U is not None: trip+=[(J,U,A,1.),(J,A,U,1.),(U,J,A,1.),(U,A,J,1.),(A,J,U,1.),(A,U,J,1.)]
    cache={}
    tr=lambda X: cache.setdefault(id(X), Wt_.T@X)
    for X,Y,Z,c in trip:
        Yp=tr(Y); Zp=tr(Z); R=Hm.T@X
        T1+=c*((R*Yp)@Zp.T); Lam+=c*(X@(Yp*Zp).T)
N=f2[:,None]*w*Lam
K31b=6*T1+3*(N.T@G.T)+3*((G*Lam.T)@(f2[:,None]*w))
m2,S2,T2,Q2,K312=state(l+1); v2=np.diag(S2); o_=~np.eye(1024,dtype=bool)
RK=(v2[:,None]*S2)[o_]; y=K312[o_]; Bb=K31b[o_]
b1=np.sum(RK*y)/np.sum(RK*RK); r1=1-np.sum((y-b1*RK)**2)/np.sum(y**2)
X=np.stack([RK,Bb],1); b=np.linalg.lstsq(X,y,rcond=None)[0]; r2=1-np.sum((y-X@b)**2)/np.sum(y**2)
print(l+1,'%.0fs'%(time.time()-t0),'|births|/|K31| %.3f'%(np.linalg.norm(Bb)/np.linalg.norm(y)),'R2 radial %.3f | radial+births %.3f coef %s | corr(births, radial-resid) %.3f'%(r1,r2,np.round(b,4),np.corrcoef(Bb,y-b1*RK)[0,1]),flush=True)
np.save(f'data/k31b_{l+1}.npy',K31b.astype(np.float32))
