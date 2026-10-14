"""Pairwise Cumulant Propagation (PCP), numpy float64 research version.
State per pre-activation layer: m (mean), S (cov), T (T_il = k(y_i,y_i,y_l), diag = k3), Q (Q_il = k(y_i,y_i,y_l,y_l), diag = k4)."""
import numpy as np, sys, time
from lib import *
from math import factorial
def Hes(a,K):
    H=[np.ones_like(a), a.copy()]
    for m in range(2,K+1): H.append(a*H[-1]-(m-1)*H[-2])
    return H
def series(u,v,R,K):
    out=np.zeros_like(R); Rk=np.ones_like(R)
    for k in range(1,K+1): Rk=Rk*R; out+=np.outer(u[k],v[k])*Rk/factorial(k)
    return out
def transfer(m,S,T,Q,K=10,opts={}):
    """ReLU transfer: returns mu, C, G (k(h_i,h_i,h_l)), M (k(h_i,h_i,h_l,h_l)) with diagonals = var, k3, k4 of h."""
    s=np.sqrt(np.diag(S)); a=m/s; P=ndtr(a); p=pdf(a); KK=K+4 if opts.get('edgecoef',False) else K; H=Hes(a,KK+2)
    muG=m*P+s*p
    f=[None, s*P]+[s*p*(-1)**k*H[k-2] for k in range(2,KK+1)]
    g=[None, 2*s*((m-muG)*P+s*p), 2*s*s*P-2*s*muG*p]+[2*s*s*(-1)**(k-3)*H[k-3]*p-2*s*muG*(-1)**(k-2)*H[k-2]*p for k in range(3,KK+1)]
    if opts.get('edgecoef',False):
        k3_=np.diag(T); k4_=np.diag(Q); c3_=k3_/(6*s**3); c4_=k4_/(24*s**4)
        f=[None]+[f[k]+c3_*f[k+3]+c4_*f[k+4] for k in range(1,K+1)]
        g=[None]+[g[k]+c3_*g[k+3]+c4_*g[k+4] for k in range(1,K+1)]
    R=S/np.outer(s,s)
    # 1D: raw moments of relu under Gaussian + Edgeworth corrections
    k3=np.diag(T).copy(); k4=np.diag(Q).copy()
    r1=m*P+s*p; r2=(m*m+s*s)*P+m*s*p; r3=(m**3+3*m*s*s)*P+(m*m+2*s*s)*s*p; r4=(m**4+6*m*m*s*s+3*s**4)*P+(m**3+5*m*s*s)*s*p
    if opts.get('edge1d',True):
        r1=r1+k3/6*(-a*p/s**2)+k4/24*p*(a*a-1)/s**3
        r2=r2+k3/6*(2*p/s)+k4/24*(-2*a*p/s**2)
        r3=r3+k3/6*(6*P)+k4/24*(6*p/s)
        r4=r4+k3/6*(24*muG)+k4/24*(24*P)
    mu=r1; c2=r2-mu**2; c3=r3-3*mu*r2+2*mu**3; c4=r4-4*mu*r3+6*mu*mu*r2-3*mu**4
    kh4=c4-3*c2**2
    # expectations of derivatives under Gaussian
    F1=P; F2=p/s; A1=g[1]/s; A2=g[2]/s**2
    C=series(f,f,R,K); G=series(g,f,R,K); Mc=series(g,g,R,K)
    if opts.get('cross',True):
        Tt=T.T
        C+=0.5*T*np.outer(F2,F1)+0.5*Tt*np.outer(F1,F2)+0.25*Q*np.outer(F2,F2)
        G+=0.5*T*np.outer(A2,F1)+0.5*Tt*np.outer(A1,F2)+0.25*Q*np.outer(A2,F2)
        Mc+=0.5*T*np.outer(A2,A1)+0.5*Tt*np.outer(A1,A2)+0.25*Q*np.outer(A2,A2)
    np.fill_diagonal(C,c2)
    M=Mc-2*C*C
    np.fill_diagonal(G,c3); np.fill_diagonal(M,kh4)
    return mu,C,G,M,dict(s=s,P=P,q=f[2]/s**2)
def linear(w,mu,C,G,M,aux,S_prev,opts={}):
    n=w.shape[0]; w2=w*w
    m=w.T@mu; S=w.T@C@w
    # T_new[j,k] = k(y_j,y_j,y_k)
    X=G@w                         # X[a,j] = sum_b G_ab w_bj
    Gd=np.diag(G)
    T=w2.T@X + 2*(w*X).T@w        # pattern a=b ; patterns a=c,b=c
    # overcount of a=b=c: appears in all three patterns; should appear once
    T-= 2*(w2*Gd[:,None]).T@w     # a=b=c counted three times above; keep once
    if opts.get('wick3off',False):
        P=aux['P']; q=aux['q']; s=aux['s']
        U=S_prev@(P[:,None]*w)
        Tw=2*(w*U*q[:,None]).T@U + (q[:,None]*U*U).T@w
        T2=2*(q*P*s**2)[:,None]*(P[:,None]*S_prev)+(P**2)[:,None]*(q[None,:]*S_prev**2)   # F(a,a,c) [a,c]
        coinc = w2.T@(T2@w) + 2*(w*(T2@w)).T@w
        T+= Tw-coinc
    if opts.get('wickoff',False):
        P_=aux['P']; q_=aux['q']
        St=S_prev.copy(); np.fill_diagonal(St,0)
        Ut=St@(P_[:,None]*w); V=(St*St)@(q_[:,None]*w)
        T=T+2*(w*q_[:,None]*Ut).T@Ut + (q_[:,None]*Ut*Ut - 2*w*(P_*P_)[:,None]*V).T@w - (w2*(P_*P_)[:,None]).T@V
    # fix diagonal exactly as before (triple coincidence counted once)
    Td = 3*np.einsum('ij,ij->j',w2,X)-2*np.einsum('ij,i->j',w2*w,Gd)
    if not opts.get('toff',True): T=np.zeros_like(T)
    if opts.get('wick3',True) and opts.get('layer',0) < opts.get('wick_until',99):
        P=aux['P']; q=aux['q']; s=aux['s']; U=S_prev@(P[:,None]*w)
        T2=2*(q*P*s**2)[:,None]*(P[:,None]*S_prev)+(P**2)[:,None]*(q[None,:]*S_prev**2)
        Tw_d=3*np.einsum('ij,i,ij->j',w,q,U**2)
        co_d=3*np.einsum('ij,ij->j',w2,T2@w)-3*np.einsum('ij,i->j',w2*w,np.diag(T2))+np.einsum('ij,i->j',w2*w,3*q*P**2*s**4)
        Td+=Tw_d-co_d
    np.fill_diagonal(T,Td)
    # Q_new[j,k] = k(y_j,y_j,y_k,y_k)
    Md=np.diag(M).copy()
    Qn=w2.T@(M@w2) if opts.get('qoff',True) else np.zeros_like(M)
    Qd=3*np.einsum('ij,ij->j',w2,M@w2)-2*np.einsum('ij,i->j',w2*w2,Md)
    np.fill_diagonal(Qn,Qd)
    return m,S,T,Qn
def run(Ws,K=10,opts={},rec=None):
    n=Ws.shape[1]; rows=[]
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); Q=np.zeros((n,n))
    for l in range(len(Ws)):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        rows.append(mu)
        if rec is not None: rec.append(dict(m=m,v=np.diag(S).copy(),k3=np.diag(T).copy(),k4=np.diag(Q).copy()))
        if l+1<len(Ws):
            opts['layer']=l+1
            m,S,T,Q=linear(Ws[l+1],mu,C,G,M,aux,S,opts)
    return np.array(rows)
def perlayer(fn,ids):
    E=[];B=[]
    for i in ids:
        p=fn(W(i)); t=TRUTH[list(IDS).index(i)]
        E.append(np.mean((p-t)**2,1)); B.append(np.mean(t-p,1))
    return np.mean(E,0),np.mean(B,0)
if __name__=='__main__':
    ids=IDS[:int(sys.argv[1])]
    A=(np.load('data/momacc_0_11.npy')+np.load('data/momacc_0_12.npy'))/2
    confs=[('full',{}),('no cross',dict(cross=False)),('no wick3',dict(wick3=False))]
    for name,o in confs:
        rec=[]; run(W(0),opts=o,rec=rec); out=[]
        for l in [1,2,3]:
            r1,r2,r3,r4=A[l]; c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2; d=rec[l]
            out.append(f"L{l+1} var {np.mean(c2/d['v']-1):+.1e} ku {np.mean(c4/d['v']**2):.4f}/{np.mean(d['k4']/d['v']**2):.4f}")
        t=time.time(); e,b=perlayer(lambda Ws: run(Ws,opts=o),ids)
        print(name,'|'.join(out),'| final',e[-1],'bias',np.array2string(b[[2,5,10,15]],precision=2),f'{time.time()-t:.0f}s',flush=True)

from scipy.special import gammaln
def run_sphere(Ws,K=10,opts={},rec=None):
    n=Ws.shape[1]; rows=[]
    Er=np.exp(0.5*np.log(2/n)+gammaln((n+1)/2)-gammaln(n/2))
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); d=np.diag(S)
    Q=-2/(n+2)*(np.outer(d,d)+2*S*S)
    for l in range(len(Ws)):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        rows.append(Er*mu)
        if l+1<len(Ws):
            m,S,T,Q=linear(Ws[l+1],mu,C,G,M,aux,S,opts)
    return np.array(rows)

def local_k3(Z,G,aux,S,wick):
    s,P,q=aux['s'],aux['P'],aux['q']; Gd=np.diag(G); Z2=Z*Z
    out=3*np.einsum('ij,ij->j',Z2,G@Z)-2*np.einsum('ij,i->j',Z2*Z,Gd)
    if wick:
        U=S@(P[:,None]*Z); out+=3*np.einsum('ij,i,ij->j',Z,q,U**2)
        T2m=2*(q*P*s**2)[:,None]*(P[:,None]*S)+(P**2)[:,None]*(q[None,:]*S**2)
        out-=3*np.einsum('ij,ij->j',Z2,T2m@Z)-3*np.einsum('ij,i->j',Z2*Z,np.diag(T2m))+np.einsum('ij,i->j',Z2*Z,3*q*P**2*s**4)
    return out
def run_t(Ws,K=6,opts={},levels=1,rec=None,wick_trans=None):
    """PCP with transported (long-range) third cumulants, `levels` layers back."""
    n=Ws.shape[1]; rows=[]; L=len(Ws)
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); Q=np.zeros((n,n))
    hist=[]   # per pre-activation layer index k (0-based): dict(G=h-level G, aux, S, T)
    for l in range(L):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        hist.append(dict(G=G,aux=dict(s=aux['s'],P=aux['P'],q=aux['q']),S=S,T=T))
        rows.append(mu)
        if rec is not None: rec.append(dict(m=m,v=np.diag(S).copy(),k3=np.diag(T).copy(),k4=np.diag(Q).copy()))
        if l+1<L:
            opts['layer']=l+1
            m,S,T,Q=linear(Ws[l+1],mu,C,G,M,aux,S,opts)
            # transported third cumulant for y_{l+1}
            extra=np.zeros(n); U=hist[l]['aux']['P'][:,None]*Ws[l+1]; k=l
            for lev in range(levels):
                if k==0: break              # y_0 (first pre-activation) is Gaussian
                Z=Ws[k]@U
                wk = (k-1==0) if wick_trans is None else wick_trans
                full=local_k3(Z,hist[k-1]['G'],hist[k-1]['aux'],hist[k-1]['S'],wk)
                Tk=hist[k]['T']
                coinc=3*np.einsum('ij,ij->j',U*U,Tk@U)-2*np.einsum('ij,i->j',U**3,np.diag(Tk))
                extra+=full-coinc
                U=hist[k-1]['aux']['P'][:,None]*Z; k-=1
            T=T.copy(); T[np.diag_indices(n)]+=extra
    return np.array(rows)

def pair3(Z,Gm,Gd):
    """pair slice k(z_j,z_j,z_l) of Z^T x given pair third cumulants Gm (off-diag) and diag Gd of x. Returns (full matrix, diag)."""
    Z2=Z*Z; X=Gm@Z
    Tm=Z2.T@X + (2*(Z*X-Z2*Gd[:,None])).T@Z
    Td=3*np.einsum('ij,ij->j',Z2,X)-2*np.einsum('ij,i->j',Z2*Z,Gd)
    return Tm,Td
def run_tfull(Ws,K=6,opts={},levels=99,offdiag=True,rec=None):
    n=Ws.shape[1]; rows=[]; L=len(Ws)
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); Q=np.zeros((n,n))
    hist=[]
    for l in range(L):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        hist.append(dict(G=G,aux=dict(s=aux['s'],P=aux['P'],q=aux['q']),S=S,T=T))
        rows.append(mu)
        if rec is not None: rec.append(dict(m=m,v=np.diag(S).copy(),k3=np.diag(T).copy(),k4=np.diag(Q).copy()))
        if l+1<L:
            opts['layer']=l+1
            m,S,T,Q=linear(Ws[l+1],mu,C,G,M,aux,S,opts)
            extraM=np.zeros((n,n)); extraD=np.zeros(n)
            U=hist[l]['aux']['P'][:,None]*Ws[l+1]; k=l
            for lev in range(levels):
                if k==0: break
                Z=Ws[k]@U
                Gk=hist[k-1]['G']; Gm,Gd=Gk,np.diag(Gk)
                lm,ld=pair3(Z,Gm,Gd)
                if k-1==0:  # Wick at the Gaussian first layer (diag only, as validated)
                    ld=ld+local_k3(Z,Gk,hist[k-1]['aux'],hist[k-1]['S'],True)-local_k3(Z,Gk,hist[k-1]['aux'],hist[k-1]['S'],False)
                Tk=hist[k]['T']
                cm,cd=pair3(U,Tk,np.diag(Tk))
                extraM+=lm-cm; extraD+=ld-cd
                U=hist[k-1]['aux']['P'][:,None]*Z; k-=1
            T=T.copy()
            if offdiag: T+=extraM
            T[np.diag_indices(n)]=np.diag(T)-np.diag(extraM if offdiag else 0*extraM)+extraD if offdiag else np.diag(T)+extraD
    return np.array(rows)

DIAGMODE={'mode':'validated'}
LEVNORM=None
def k3_slices(Z,G,S,aux,wick=True):
    """Full pair slices k(z_j,z_j,z_l) of Z^T h for h = relu(y), y ~ closure with cov S, h-pair slices G.
    Includes coincident patterns (G) and the distinct-index Wick path term. Returns (matrix with correct diag)."""
    Z2=Z*Z; Gd=np.diag(G); X=G@Z
    Tm=Z2.T@X + (2*(Z*X-Z2*Gd[:,None])).T@Z
    Td=3*np.einsum('ij,ij->j',Z2,X)-2*np.einsum('ij,i->j',Z2*Z,Gd)
    if wick:
        P=aux['P']; q=aux['q']; s=aux['s']
        St=S.copy(); np.fill_diagonal(St,0)
        Ut=St@(P[:,None]*Z); V=(St*St)@(q[:,None]*Z)
        Tm=Tm+2*(Z*q[:,None]*Ut).T@Ut + (q[:,None]*Ut*Ut - 2*Z*(P*P)[:,None]*V).T@Z - (Z2*(P*P)[:,None]).T@V
        if DIAGMODE['mode']=='validated':
            Td=Td+local_k3(Z,G,aux,S,True)-local_k3(Z,G,aux,S,False)
    if DIAGMODE['mode']=='fromoff':
        # diag from the off-diagonal expression evaluated at j=l, then fix triple coincidence (counted 3x -> 1x)
        Td=np.diag(Tm).copy()
    np.fill_diagonal(Tm,Td)
    return Tm
def run_v2(Ws,K=6,opts={},levels=99,rec=None,wick_levels=99,trans_G=True,trans_off=True,off_levels=99):
    n=Ws.shape[1]; rows=[]; L=len(Ws)
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); Q=np.zeros((n,n))
    hist=[]
    for l in range(L):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        a_=dict(s=aux['s'],P=aux['P'],q=aux['q'])
        hist.append(dict(G=G,aux=a_,S=S,T=T))
        rows.append(mu)
        if rec is not None: rec.append(dict(m=m,v=np.diag(S).copy(),k3=np.diag(T).copy(),k4=np.diag(Q).copy(),T=T,Q=Q,C=C))
        if l+1<L:
            w=Ws[l+1]; w2=w*w
            m_new=w.T@mu; S_new=w.T@C@w
            Tn=k3_slices(w,G,S,a_,wick=True)
            Md=np.diag(M).copy(); Qn=w2.T@(M@w2); Qd=3*np.einsum('ij,ij->j',w2,M@w2)-2*np.einsum('ij,i->j',w2*w2,Md); np.fill_diagonal(Qn,Qd)
            # transport of distinct part of y-level third cumulants
            U=a_['P'][:,None]*w; k=l
            for lev in range(levels):
                if k==0: break
                Z=Ws[k]@U
                Gk=hist[k-1]['G'] if trans_G else np.zeros((n,n))
                full=k3_slices(Z,Gk,hist[k-1]['S'],hist[k-1]['aux'],wick=(lev<wick_levels))
                Tk=hist[k]['T']
                coinc=pair3(U,Tk,np.diag(Tk)); cm=coinc[0]; np.fill_diagonal(cm,coinc[1])
                add=full-cm
                if LEVNORM is not None and l+1 in (10,15): LEVNORM.append((l+1,lev,np.diag(add).copy(),add.copy())) if False else None
                if LEVNORM is not None and l+1 in (10,15): LEVNORM.append((l+1,lev,np.diag(add).copy(),add[:64,:64].copy()))
                if (not trans_off) or lev>=off_levels: add=np.diag(np.diag(add))
                Tn=Tn+add
                U=hist[k-1]['aux']['P'][:,None]*Z; k-=1
            m,S,T,Q=m_new,S_new,Tn,Qn
    return np.array(rows)

# ---------------- forward factored transport ----------------
class Blocks:
    """Symmetric 3-tensor X = sum_blocks w * sum_r Sym3(A_r, A_r, B_r); A,B are (n, rank)."""
    def __init__(self,n): self.n=n; self.A=np.zeros((n,0)); self.B=np.zeros((n,0)); self.w=np.zeros(0)
    def add(self,A,B,w):
        self.A=np.concatenate([self.A,A],1); self.B=np.concatenate([self.B,B],1); self.w=np.concatenate([self.w,np.full(A.shape[1],float(w)) if np.ndim(w)==0 else w])
    def slices(self):
        A=self.A*self.w[None,:]; B=self.B
        Sm=((self.A*self.A*self.w[None,:])@B.T + 2*((self.A*B*self.w[None,:])@self.A.T))/3
        d=np.einsum('jr,jr,r->j',self.A*self.A,B,self.w); np.fill_diagonal(Sm,d)
        return Sm
    def contract(self,W): self.A=W.T@self.A; self.B=W.T@self.B
    def scale_rows(self,v): self.A=self.A*v[:,None]; self.B=self.B*v[:,None]
    def copy(self):
        b=Blocks(self.n); b.A=self.A.copy(); b.B=self.B.copy(); b.w=self.w.copy(); return b
def coin_block(S):
    n=S.shape[0]; return np.eye(n), 3*S.T-2*np.diag(np.diag(S))
def run_fac(Ws,K=6,opts={},s=1.0,seed=0,rec=None,exact_local=True):
    rng=np.random.default_rng(seed)
    n=Ws.shape[1]; rows=[]; L=len(Ws)
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); Q=np.zeros((n,n))
    Y=Blocks(n)      # transported tensor at current y level (excluding what is exact-local)
    def sub(A,B,w):
        if s>=1.0: return A,B,w
        k=max(1,int(round(s*A.shape[1]))); idx=rng.choice(A.shape[1],k,replace=False)
        return A[:,idx],B[:,idx],w*A.shape[1]/k
    for l in range(L):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        a_=dict(s=aux['s'],P=aux['P'],q=aux['q'])
        rows.append(mu)
        if l+1<L:
            w=Ws[l+1]; w2=w*w; P=a_['P']; q=a_['q']
            m_new=w.T@mu; S_new=w.T@C@w
            Md=np.diag(M).copy(); Qn=w2.T@(M@w2); Qd=3*np.einsum('ij,ij->j',w2,M@w2)-2*np.einsum('ij,i->j',w2*w2,Md); np.fill_diagonal(Qn,Qd)
            # H = Coin_h(G) + ZR_h(Wick) + P^3 o ZR_y(Y)
            H=Blocks(n)
            # transported part
            if Y.A.shape[1]>0:
                Ysl=Y.slices(); Ai,Bi=coin_block(Ysl)
                Z=Y.copy(); Z.add(Ai,Bi,-1.0); Z.scale_rows(P)
                H.add(Z.A,Z.B,Z.w)
            Hexact=Blocks(n)
            St=S.copy(); np.fill_diagonal(St,0)
            Aw=St*P[:,None]            # A_ab = S~_ab P_a, column b
            Bw=np.diag(q)
            Ag,Bg=coin_block(G)
            Awc,Bwc=coin_block((Aw*Aw)*q[None,:])
            local=[(Ag,Bg,1.0),(Aw,Bw,3.0),(Awc,Bwc,-1.0)]
            for (A_,B_,w_) in local: Hexact.add(A_,B_,w_)
            # slices for T_{l+1}: exact local + transported
            Hc=H.copy(); Hc.contract(w); Tn=Hc.slices() if Hc.A.shape[1]>0 else np.zeros((n,n))
            if exact_local:
                He=Hexact.copy(); He.contract(w); Tn=Tn+He.slices()
            # next transported tensor: all H blocks contracted, with local blocks subsampled
            Ynew=Hc
            for (A_,B_,w_) in local:
                A2,B2,w2_=sub(A_,B_,w_); Ynew.add(w.T@A2,w.T@B2,w2_)
            if not exact_local:
                Tn=Ynew.slices()
            Y=Ynew
            m,S,T,Q=m_new,S_new,Tn,Qn
    return np.array(rows)

def run_fac2(Ws,K=6,opts={},s=1.0,seed=0,exact_local=True,sample_mode='rand'):
    """Forward factored transport without zero-repeat blocks: H = [Coin_h(G) - P^3 o Coin_y(T)] + ZR_h(Wick) + P^3 o Y."""
    rng=np.random.default_rng(seed)
    n=Ws.shape[1]; rows=[]; L=len(Ws)
    w=Ws[0]; m=np.zeros(n); S=w.T@w; T=np.zeros((n,n)); Q=np.zeros((n,n))
    Y=Blocks(n)
    def sub(A,B,w_):
        if s>=1.0: return A,B,w_
        k=max(1,int(round(s*A.shape[1])))
        idx=rng.choice(A.shape[1],k,replace=False)
        return A[:,idx],B[:,idx],w_*A.shape[1]/k
    for l in range(L):
        mu,C,G,M,aux=transfer(m,S,T,Q,K,opts)
        a_=dict(s=aux['s'],P=aux['P'],q=aux['q'])
        rows.append(mu)
        if l+1<L:
            w=Ws[l+1]; w2=w*w; P=a_['P']; q=a_['q']
            m_new=w.T@mu; S_new=w.T@C@w
            Md=np.diag(M).copy(); Qn=w2.T@(M@w2); Qd=3*np.einsum('ij,ij->j',w2,M@w2)-2*np.einsum('ij,i->j',w2*w2,Md); np.fill_diagonal(Qn,Qd)
            # local blocks at h level
            St=S.copy(); np.fill_diagonal(St,0)
            Aw=St*P[:,None]; Bw=np.diag(q)
            Gc=G-(P*P)[:,None]*T*P[None,:]          # coincident nonlinear correction: G - P^3 o T (entry (i,i,l): P_i^2 P_l T_il)
            Ag,Bg=coin_block(Gc)
            Awc,Bwc=coin_block((Aw*Aw)*q[None,:])
            local=[(Ag,Bg,1.0),(Aw,Bw,3.0),(Awc,Bwc,-1.0)]
            # transported Y (full, no ZR), scaled by P
            Yt=Y.copy(); Yt.scale_rows(P); Yt.contract(w)
            Tn=Yt.slices() if Yt.A.shape[1]>0 else np.zeros((n,n))
            He=Blocks(n)
            for (A_,B_,w_) in local: He.add(w.T@A_,w.T@B_,w_)
            Tn=Tn+He.slices()
            Ynew=Yt
            for (A_,B_,w_) in local:
                A2,B2,wq=sub(A_,B_,w_); Ynew.add(w.T@A2,w.T@B2,wq)
            Y=Ynew
            m,S,T,Q=m_new,S_new,Tn,Qn
    return np.array(rows)
