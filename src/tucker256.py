import numpy as np, fac3, sys
from lib import *
n=256; L=16; rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
Ws=rng.standard_normal((L,n,n))*np.sqrt(2/n)
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False)
tl=int(sys.argv[2]) if len(sys.argv)>2 else 12; A0=int(sys.argv[3]) if len(sys.argv)>3 else 4
fac3.OLD.clear(); fac3.run_fac3(Ws,opts=dict(EX,dump_srcs_at=tl))
d=fac3.OLD[-1]; m=d['m']; S=d['S']
def full(sr):
    J,A,B,U,q=sr['J'],sr['A'],sr['B'],sr['U'],sr['q']; V=J*q[None,:]
    E=lambda X,Y,Z: ((X[:,None,:]*Y[None,:,:]).reshape(n*n,-1)@Z.T).reshape(n,n,n)
    K=(E(J,J,B)+E(J,B,J)+E(B,J,J))/3+(E(A,A,V)+E(A,V,A)+E(V,A,A))
    if U is not None:
        K+=E(J,U,A)+E(J,A,U)+E(U,J,A)+E(U,A,J)+E(A,J,U)+E(A,U,J)
    return K
def sl(K):  # (2,1) slice K[a,a,b]
    return np.einsum('aab->ab',K)
Kold=sum(full(sr) for sr in d['srcs'] if sr['age']>=A0); Kall=sum(full(sr) for sr in d['srcs'])
print('check slice vs chain Tn: rel %.2e'%(np.linalg.norm(sl(Kall)-d['Tn'])/np.linalg.norm(d['Tn'])))
# radial tensors
I3=lambda x,M: x[:,None,None]*M[None,:,:]+x[None,:,None]*M[:,None,:]+x[None,None,:]*M[:,:,None]
R1=I3(m,S); R2=np.einsum('a,b,c->abc',m,m,m)
X=np.stack([R1.ravel(),R2.ravel()],1); bg=np.linalg.lstsq(X,Kold.ravel(),rcond=None)[0]
Krad=bg[0]*R1+bg[1]*R2; Kres=Kold-Krad
print('layer',tl,'old ages>=',A0,' |Kold|/|Kall| %.3f  radial R2 (full tensor) %.3f'%(np.linalg.norm(Kold)/np.linalg.norm(Kall),1-np.linalg.norm(Kres)**2/np.linalg.norm(Kold)**2))
def hosvd(K,r):
    M1=K.reshape(n,-1); U_,s_,_=np.linalg.svd(M1,full_matrices=False); Q=U_[:,:r]
    P=Q@Q.T; K2=np.tensordot(K,P,axes=([2],[0])); K2=np.tensordot(K2,P,axes=([1],[0])).transpose(0,2,1); K2=np.tensordot(P,K2,axes=([1],[0])); return K2
nS=np.linalg.norm(sl(Kold))
for r in (16,32,48,64,96,128):
    e1=np.linalg.norm(sl(Kold-hosvd(Kold,r)))/nS
    e2=np.linalg.norm(sl(Kres-hosvd(Kres,r)))/nS
    print('  r=%3d (n/%.0f): slice err  tucker-only %.3f | radial+tucker(resid) %.3f'%(r,n/r,e1,e2),flush=True)
