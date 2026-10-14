import numpy as np
def mix_tensor(T,S,P,F2,F3):
    n=len(P); K=np.zeros((n,n,n))
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if len({a,b,c})<3: continue
                v=0
                for (x,y,z) in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
                    v+=0.5*F3[x]*P[y]*P[z]*T[x,y]*S[x,z]+0.5*F2[x]*F2[y]*P[z]*T[x,y]*S[y,z]
                K[a,b,c]=v
    return K
def mix_slices(W,T,S,P,F2,F3):
    """T_il += sum_{abc distinct} W_ai W_bi W_cl kmix(a,b,c)."""
    Tt=T.copy(); np.fill_diagonal(Tt,0); St=S.copy(); np.fill_diagonal(St,0)
    PW=P[:,None]*W; F2W=F2[:,None]*W; F3W=F3[:,None]*W; W2=W*W
    TP=Tt@PW; SP=St@PW; TtF=Tt.T@F2W
    out=(TP*SP).T@F3W                       # type1 center on l-side
    out+=(F3W*TP).T@SP + (F3W*SP).T@TP      # type1 center on i-side
    out+=(TtF*SP).T@F2W                     # type2 middle on l-side
    out+=(F2W*TtF).T@SP + (F2W*SP).T@TtF    # type2 middle on i-side
    # remove a==b coincidences (a,b both on i side) that were counted
    TS=Tt*St
    out-=(W2*(P*P)[:,None]).T@(TS.T@F3W)          # 1: a=b in type1 center l
    X=F3W.T@TS
    out-=2*((X*((P*P)[:,None]*W).T)@W)            # 2,3: b=c in type1 center i
    out-=(W2*(F2*P)[:,None]).T@(TS@F2W)           # 4: a=b in type2 middle l
    Y=F2W.T@TS.T
    out-=2*((Y*((F2*P)[:,None]*W).T)@W)           # 5,6
    return out
def brute(W,K):
    return np.einsum('ai,bi,cl,abc->il',W,W,W,K)
if __name__=='__main__':
    rng=np.random.default_rng(0); n=7
    W=rng.standard_normal((n,n)); A=rng.standard_normal((n,n)); S=A@A.T/n; T=rng.standard_normal((n,n))
    P=rng.random(n); F2=rng.random(n); F3=rng.standard_normal(n)
    K=mix_tensor(T,S,P,F2,F3); b=brute(W,K); m=mix_slices(W,T,S,P,F2,F3)
    off=~np.eye(n,dtype=bool)
    print('offdiag maxdiff',np.abs(b-m)[off].max(),'scale',np.abs(b).max())
