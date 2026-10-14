import numpy as np
def path_slices(W, E1, E2, ce1, cm, ce2):
    """sum_{a,b,c distinct} W_ai W_bi W_cl sum_{path (e1,m,e2) -> perm of (a,b,c)} E1[e1,m] E2[m,e2] ce1[e1] cm[m] ce2[e2].
    E1, E2 must have zero diagonal."""
    W2 = W * W
    A1 = E1.T @ (ce1[:, None] * W)      # A1[m,i] = sum_e1 E1[e1,m] ce1 W_e1,i
    A2 = E2 @ (ce2[:, None] * W)        # A2[m,i] = sum_e2 E2[m,e2] ce2 W_e2,i
    CW = cm[:, None] * W
    E12 = E1.T * E2                     # E12[m,e] = E1[e,m] E2[m,e]
    # mid on l-side (c): ends a,b on i-side
    out = (A1 * A2).T @ CW
    out -= (W2 * (ce1 * ce2)[:, None]).T @ (E12.T @ CW)          # a==b
    # mid on i-side: mid=a or b (x2 handled by summing both orientations below), ends one i-side one l-side
    # (e1 on i-side, e2 on l-side)
    out += (CW * A1).T @ A2
    X = CW.T @ E12                                               # X[i,e] = sum_m cm W_mi E12[m,e]
    out -= (X * ((ce1 * ce2)[:, None] * W).T) @ W               # e1==e2 coincidence
    # (e2 on i-side, e1 on l-side)
    out += (CW * A2).T @ A1
    out -= (X * ((ce1 * ce2)[:, None] * W).T) @ W
    return 2 * out
def brute_path(W,E1,E2,ce1,cm,ce2):
    n=len(cm); K=np.zeros((n,n,n))
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if len({a,b,c})<3: continue
                v=0
                for (e1,m,e2) in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
                    v+=E1[e1,m]*E2[m,e2]*ce1[e1]*cm[m]*ce2[e2]
                K[a,b,c]=v
    return np.einsum('ai,bi,cl,abc->il',W,W,W,K)
def mix_all(W, T, S, Q, K31, P, F2, F3, F4, which=('T','Q','K')):
    Z = lambda X: X - np.diag(np.diag(X))
    St = Z(S); out = np.zeros((W.shape[1],) * 2)
    if 'T' in which:
        Tt = Z(T)
        out += 0.5 * path_slices(W, Tt.T, St, P, F3, P) + 0.5 * path_slices(W, Tt, St, F2, F2, P)
    if 'Q' in which and Q is not None:
        out += 0.5 * path_slices(W, Z(Q), St, F2, F3, P)
    if 'K' in which and K31 is not None:
        Kt = Z(K31)
        out += (path_slices(W, Kt.T, St, P, F4, P) + path_slices(W, Kt, St, F3, F2, P)) / 6
    return out
if __name__ == '__main__':
    import mix
    rng=np.random.default_rng(1); n=7
    W=rng.standard_normal((n,n)); A=rng.standard_normal((n,n)); S=A@A.T/n; T=rng.standard_normal((n,n))
    E=rng.standard_normal((n,n)); np.fill_diagonal(E,0); St=S-np.diag(np.diag(S))
    c1,c2,c3=rng.random(n),rng.standard_normal(n),rng.random(n)
    off=~np.eye(n,dtype=bool)
    print('path', np.abs(path_slices(W,E,St,c1,c2,c3)-brute_path(W,E,St,c1,c2,c3))[off].max())
    P=rng.random(n); F2=rng.random(n); F3=rng.standard_normal(n)
    print('T-mix vs mix.py', np.abs(mix_all(W,T,S,None,None,P,F2,F3,None,('T',))-mix.mix_slices(W,T,S,P,F2,F3))[off].max())
