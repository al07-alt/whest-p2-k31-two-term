"""PCP v3: v2 (transported third cumulants) + (3,1) pair fourth cumulant K31_il = k(y_i,y_i,y_i,y_l)."""
import numpy as np
from math import factorial
from lib import *
import pcp
pcp.DIAGMODE['mode'] = 'fromoff'
OPT = dict(k31_cross=True, k31_h=True, k31_lin=True, k31_q=True, k31_uc=True, k31_qd=True)

def Edelta(j, a, p, s):
    """E[delta^{(j)}(y)] for y ~ N(m, s^2), a = m/s, p = phi(a)."""
    return (-1) ** j * pcp.Hes(a, max(j, 1))[j] * p / s ** (j + 1)

def transfer3(m, S, T, Q, K31, K=6, opts={}):
    s = np.sqrt(np.diag(S)); a = m / s; P = ndtr(a); p = pdf(a); H = pcp.Hes(a, K + 2)
    muG = m * P + s * p
    f = [None, s * P] + [s * p * (-1) ** k * H[k - 2] for k in range(2, K + 1)]
    g = [None, 2 * s * ((m - muG) * P + s * p), 2 * s * s * P - 2 * s * muG * p] + \
        [2 * s * s * (-1) ** (k - 3) * H[k - 3] * p - 2 * s * muG * (-1) ** (k - 2) * H[k - 2] * p for k in range(3, K + 1)]
    # u(y) = (relu(y)-muG)^3 : E[u^{(k)}]
    mu_ = muG; r1g = muG; r2g = (m * m + s * s) * P + m * s * p
    Ed = lambda j: Edelta(j, a, p, s)
    Eu = [None, 3 * (r2g - 2 * mu_ * r1g + mu_ ** 2 * P), 6 * (r1g - mu_ * P) + 3 * mu_ ** 2 * Ed(0),
          6 * P - 6 * mu_ * Ed(0) + 3 * mu_ ** 2 * Ed(1)]
    for k in range(4, K + 1): Eu.append(6 * Ed(k - 4) - 6 * mu_ * Ed(k - 3) + 3 * mu_ ** 2 * Ed(k - 2))
    u = [None] + [s ** k * Eu[k] for k in range(1, K + 1)]
    R = S / np.outer(s, s)
    if opts.get('qoffscale', 1.0) != 1.0 or opts.get('k4scale', 1.0) != 1.0:
        qd = np.diag(Q).copy(); Q = Q * opts.get('qoffscale', 1.0); np.fill_diagonal(Q, qd * opts.get('k4scale', 1.0))
    if opts.get('k3scale', 1.0) != 1.0:
        td = np.diag(T).copy(); T = T.copy(); np.fill_diagonal(T, td * opts['k3scale'])
    for nm, key in (('Q','qnoise'),('T','tnoise'),('K31','knoise')):
        fr = opts.get(key, 0)
        if fr and (K31 is not None or nm != 'K31'):
            X = {'Q':Q,'T':T,'K31':K31}[nm]; d = np.diag(X).copy(); o = X - np.diag(d)
            rng = np.random.default_rng(opts.get('seed', 0) + int(1e3 * float(np.sum(np.abs(d)))) % 100000)
            Z = rng.standard_normal(X.shape); Z = (Z + Z.T) / np.sqrt(2) if nm == 'Q' else Z
            ss_ = np.sqrt(np.diag(S)); sc = {'Q':np.outer(ss_**2, ss_**2), 'T':np.outer(ss_**2, ss_), 'K31':np.outer(ss_**3, ss_)}[nm]
            o = o + fr * np.std(o / sc) * Z * sc; np.fill_diagonal(o, d)
            if nm == 'Q': Q = o
            elif nm == 'T': T = o
            else: K31 = o
    if opts.get('oracle_diag') is not None:
        od = opts['oracle_diag']; L_ = opts.get('_layer', 0)
        r1_, r2_, r3_, r4_ = od['pre_mean'][L_], od['pre_m2'][L_], od['pre_m3'][L_], od['pre_m4'][L_]
        c2_ = r2_ - r1_ ** 2; c3_ = r3_ - 3 * r1_ * r2_ + 2 * r1_ ** 3; c4_ = r4_ - 4 * r1_ * r3_ + 6 * r1_ ** 2 * r2_ - 3 * r1_ ** 4 - 3 * c2_ ** 2
        which = opts.get('oracle_which', 'vk3k4')
        S = S.copy(); T = T.copy(); Q = Q.copy()
        if 'v' in which:
            sc_ = np.sqrt(c2_ / np.diag(S)); S = S * np.outer(sc_, sc_)
        if 'm' in which: m = r1_.copy()
        if 'k3' in which: np.fill_diagonal(T, c3_)
        if 'k4' in which: np.fill_diagonal(Q, c4_)
        s = np.sqrt(np.diag(S)); a = m / s; P = ndtr(a); p = pdf(a); H = pcp.Hes(a, K + 2)
        muG = m * P + s * p
        f = [None, s * P] + [s * p * (-1) ** k * H[k - 2] for k in range(2, K + 1)]
        g = [None, 2 * s * ((m - muG) * P + s * p), 2 * s * s * P - 2 * s * muG * p] + \
            [2 * s * s * (-1) ** (k - 3) * H[k - 3] * p - 2 * s * muG * (-1) ** (k - 2) * H[k - 2] * p for k in range(3, K + 1)]
    if opts.get('k4lam') is not None:
        lam_ = opts['k4lam'][opts.get('_layer', 0)]
        v_ = np.diag(S)
        if opts.get('k4lam_off', False):
            Q = (lam_ / 3) * (np.outer(v_, v_) + 2 * S * S)
        Q = Q.copy(); np.fill_diagonal(Q, lam_ * v_ * v_ + opts.get('k4k3c', 0.0) * np.diag(T) ** 2 / v_)
    if opts.get('k3beta') is not None:
        b_ = opts['k3beta'][opts.get('_layer', 0)]
        if opts.get('k3beta_off', False):
            v_ = np.diag(S); T = T + b_ * (2 * m[:, None] * S + m[None, :] * v_[:, None]) / 3
        else:
            T = T.copy(); np.fill_diagonal(T, np.diag(T) + b_ * m * np.diag(S))
    km_ = opts.get('k31_mode')
    if km_ is not None:
        L_ = opts.get('_layer', 0); v_ = np.diag(S)
        if km_ == 'oracle' and L_ in opts.get('k31_layers', range(16)):
            K31 = opts['k31_true'](L_).copy()
        else:
            K31 = opts['k4lam'][L_] * v_[:, None] * S
        np.fill_diagonal(K31, np.diag(Q))
    k3 = np.diag(T).copy(); k4 = np.diag(Q).copy()
    r1 = m * P + s * p; r2 = (m * m + s * s) * P + m * s * p; r3 = (m ** 3 + 3 * m * s * s) * P + (m * m + 2 * s * s) * s * p
    r4 = (m ** 4 + 6 * m * m * s * s + 3 * s ** 4) * P + (m ** 3 + 5 * m * s * s) * s * p
    r1 = r1 + k3 / 6 * (-a * p / s ** 2) + k4 / 24 * p * (a * a - 1) / s ** 3
    r2 = r2 + k3 / 6 * (2 * p / s) + k4 / 24 * (-2 * a * p / s ** 2)
    r3 = r3 + k3 / 6 * (6 * P) + k4 / 24 * (6 * p / s)
    r4 = r4 + k3 / 6 * (24 * muG) + k4 / 24 * (24 * P)
    if opts.get('edge6'):
        e6 = opts['edge6'] * k3 * k3 / (72 * s ** 6)
        r1 = r1 + e6 * s * p * (a ** 4 - 6 * a * a + 3)
        r2 = r2 - e6 * 2 * s * s * p * (a ** 3 - 3 * a)
        r3 = r3 + e6 * 6 * s ** 3 * p * (a * a - 1)
        r4 = r4 - e6 * 24 * s ** 4 * a * p
    mu = r1; c2 = r2 - mu ** 2; c3 = r3 - 3 * mu * r2 + 2 * mu ** 3; c4 = r4 - 4 * mu * r3 + 6 * mu * mu * r2 - 3 * mu ** 4
    kh4 = c4 - 3 * c2 ** 2
    if opts.get('edgecoef', False):
        KK = K + 4
        H2 = pcp.Hes(a, KK + 2)
        fx = [None, s * P] + [s * p * (-1) ** k * H2[k - 2] for k in range(2, KK + 1)]
        gx = [None, 2 * s * ((m - muG) * P + s * p), 2 * s * s * P - 2 * s * muG * p] + \
             [2 * s * s * (-1) ** (k - 3) * H2[k - 3] * p - 2 * s * muG * (-1) ** (k - 2) * H2[k - 2] * p for k in range(3, KK + 1)]
        c3n = k3 / (6 * s ** 3); c4n = (k4 / (24 * s ** 4)) if opts.get('edge_c4', True) else 0 * k4
        c3n = c3n * opts.get('edge_alpha', 1.0); c4n = c4n * opts.get('edge_alpha', 1.0)
        f0, g0 = f, g
        f = [None] + [fx[k] + c3n * fx[k + 3] + c4n * fx[k + 4] for k in range(1, K + 1)]
        g = [None] + [gx[k] + c3n * gx[k + 3] + c4n * gx[k + 4] for k in range(1, K + 1)]
        F1 = f[1] / s; F2 = f[2] / s ** 2; F3 = f[3] / s ** 3
        EG = opts.get('edgewhich', 'CGM')
    else:
        F1 = P; F2 = p / s; F3 = f[3] / s ** 3
    A1 = g[1] / s; A2 = g[2] / s ** 2; A3 = g[3] / s ** 3
    U1, U2, U3 = Eu[1], Eu[2], Eu[3]
    if opts.get('edgecoef', False) and EG != 'CGM':
        fC = f if 'C' in EG else f0; fG = f if 'G' in EG else f0; gG = g if 'G' in EG else g0; gM = g if 'M' in EG else g0
        C = pcp.series(fC, fC, R, K); G = pcp.series(gG, fG, R, K); Mc = pcp.series(gM, gM, R, K)
    else:
        C = pcp.series(f, f, R, K); G = pcp.series(g, f, R, K); Mc = pcp.series(g, g, R, K)
    D = pcp.series(u, f, R, K) if opts.get('k31_h', True) else None
    Tt = T.T
    C += 0.5 * T * np.outer(F2, F1) + 0.5 * Tt * np.outer(F1, F2) + 0.25 * Q * np.outer(F2, F2)
    G += 0.5 * T * np.outer(A2, F1) + 0.5 * Tt * np.outer(A1, F2) + 0.25 * Q * np.outer(A2, F2)
    Mc += 0.5 * T * np.outer(A2, A1) + 0.5 * Tt * np.outer(A1, A2) + 0.25 * Q * np.outer(A2, A2)
    if D is not None and opts.get('k31_uc', True):
        D += 0.5 * T * np.outer(U2, F1) + 0.5 * Tt * np.outer(U1, F2) + 0.25 * Q * np.outer(U2, F2)
    if K31 is not None and opts.get('k31_cross', True):
        Kt = K31.T
        C += K31 * np.outer(F3, F1) / 6 + Kt * np.outer(F1, F3) / 6
        G += K31 * np.outer(A3, F1) / 6 + Kt * np.outer(A1, F3) / 6
        Mc += K31 * np.outer(A3, A1) / 6 + Kt * np.outer(A1, A3) / 6
        if D is not None and opts.get('k31_uc', True):
            D += K31 * np.outer(U3, F1) / 6 + Kt * np.outer(U1, F3) / 6
    np.fill_diagonal(C, c2)
    M = Mc - 2 * C * C
    np.fill_diagonal(G, c3); np.fill_diagonal(M, kh4)
    K31h = None
    if D is not None:
        K31h = D - 3 * c2[:, None] * C
        np.fill_diagonal(K31h, kh4)
    if opts.get('edgecoef', False) and opts.get('edge_auxgauss', False):
        return mu, C, G, M, K31h, dict(s=s, P=P, q=p / s, A1=g0[1] / s, A2=g0[2] / s ** 2)
    return mu, C, G, M, K31h, dict(s=s, P=F1, q=f[2] / s ** 2, A1=A1, A2=A2)

def wick4(w, S, aux, want_k31):
    P = aux['P']; q = aux['q']; A1 = aux['A1']; A2 = aux['A2']
    w2 = w * w; w3 = w2 * w
    St = S.copy(); np.fill_diagonal(St, 0); St2 = St * St
    Pw = P[:, None] * w; qw = q[:, None] * w
    U = St @ Pw
    V2 = St2 @ (Pw * Pw)
    X1 = St @ (qw * U)
    Z = St2 @ qw
    B = A2 - 2 * P * P
    inner = B[:, None] * (U * U - V2) + A1[:, None] * (2 * X1 - 2 * Pw * Z)
    Q1 = w2.T @ inner
    K = None
    if want_k31:
        K = (w2 * B[:, None] * U).T @ U - ((w2.T @ (B[:, None] * St2)) * (Pw * P[:, None]).T) @ w
        WA = w2.T @ (A1[:, None] * St)
        K += (WA * qw.T) @ U - (w2 * A1[:, None] * Z).T @ Pw + (WA * U.T) @ qw - (w3 * (A1 * P)[:, None]).T @ Z
        K *= 3
    return Q1, K

def linear_q_k31(w, M, K31h, opts, S=None, aux=None):
    w2 = w * w; w3 = w2 * w
    Md = np.diag(M).copy()
    MW2 = M @ w2
    Qn = w2.T @ MW2
    Qd = 3 * np.einsum('ij,ij->j', w2, MW2) - 2 * np.einsum('ij,i->j', w2 * w2, Md)
    K31n = None
    if K31h is not None:
        K31o = K31h.copy(); np.fill_diagonal(K31o, 0)
        Mo = M.copy(); np.fill_diagonal(Mo, 0)
        KW = K31o @ w
        if opts.get('k31_lin', True):
            K31n = w3.T @ KW + 3 * (w2 * KW).T @ w + 3 * ((w2.T @ Mo) * w.T) @ w + (w3 * Md[:, None]).T @ w
        if opts.get('k31_q', True):
            V = w * KW
            Qn = Qn + 2 * (w2.T @ V + V.T @ w2)
        if opts.get('k31_qd', True):
            Qd = Qd + 4 * np.einsum('ij,ij->j', w3, KW)
    if opts.get('q4wick', False) and S is not None:
        Q1, K4w = wick4(w, S, aux, opts.get('k31wick', False) and K31n is not None)
        Qn = Qn + opts.get('q4scale', 1.0) * (Q1 + Q1.T)
        Qd = Qd + opts.get('q4dscale', 1.0) * 6 * np.diag(Q1)
        if K4w is not None: K31n = K31n + K4w
    np.fill_diagonal(Qn, Qd)
    if K31n is not None: np.fill_diagonal(K31n, Qd)
    return Qn, K31n

def mixterm(Z, T, S, Q, K31, m, aux, opts):
    import mix2
    s_ = aux['s']; a_ = m / s_; p_ = pdf(a_)
    F3 = -a_ * p_ / s_ ** 2; F4 = (a_ * a_ - 1) * p_ / s_ ** 3
    return mix2.mix_all(Z, T, S, Q, K31, aux['P'], aux['q'], F3, F4, opts.get('mixwhich', ('T',)))

def kxterm(Z, Wsrc, Mprev, aux):
    P3 = aux['P']; F2 = aux['q']; W3s = Wsrc * Wsrc
    A = W3s.T @ Mprev
    Gm = Wsrc @ (P3[:, None] * Z); Hm = W3s @ ((P3 * P3)[:, None] * Z * Z)
    kx = 0.5 * ((A @ (Gm * Gm - Hm)).T @ (F2[:, None] * Z))
    B = (F2[:, None] * Z).T @ A
    kx += (B * Gm.T) @ Gm - ((B @ W3s) * ((P3 * P3)[:, None] * Z).T) @ Z
    return kx

def run_v3(Ws, K=6, opts=None, rec=None):
    opts = dict(OPT, **(opts or {}))
    n = Ws.shape[1]; rows = []; L = len(Ws)
    w = Ws[0]; m = np.zeros(n); S = w.T @ w; T = np.zeros((n, n)); Q = np.zeros((n, n)); K31 = np.zeros((n, n))
    hist = []
    Er = 1.0
    if opts.get('sphere', False):
        from scipy.special import gammaln
        Er = np.exp(0.5 * np.log(2 / n) + gammaln((n + 1) / 2) - gammaln(n / 2))
        d = np.diag(S).copy(); c = -2.0 / (n + 2)
        Q = c * (np.outer(d, d) + 2 * S * S); np.fill_diagonal(Q, 3 * c * d * d)
        K31 = 3 * c * d[:, None] * S; np.fill_diagonal(K31, 3 * c * d * d)
    for l in range(L):
        if opts.get('qsched') is not None:
            sl, sl4 = opts['qsched']
            opts['qoffscale'] = 1 + sl * max(0, l + 1 - 4); opts['k4scale'] = 1 + sl4 * max(0, l + 1 - 4)
        opts['_layer'] = l
        mu, C, G, M, K31h, aux = TRANSFER[opts.get('tr', 'v3')](m, S, T, Q, K31, K, opts)
        hist.append(dict(G=G, aux=aux, S=S, T=T, Q=Q, K31=K31, m=m, M=M))
        rows.append(mu)
        if rec is not None: rec.append(dict(m=m, S=S, T=T, Q=Q, K31=K31))
        if l + 1 < L:
            w = Ws[l + 1]
            m_new = w.T @ mu; S_new = w.T @ C @ w
            Tn = pcp.k3_slices(w, G, S, aux, wick=True)
            if opts.get('mixT', False): Tn = Tn + mixterm(w, T, S, Q, K31, m, aux, opts)
            if opts.get('kx', False) and l >= 1: Tn = Tn + kxterm(w, Ws[l], hist[l - 1]['M'], aux)
            Qn, K31n = linear_q_k31(w, M, K31h, opts, S, aux)
            U = aux['P'][:, None] * w; k = l
            while k > 0:
                Z = Ws[k] @ U
                full = pcp.k3_slices(Z, hist[k - 1]['G'], hist[k - 1]['S'], hist[k - 1]['aux'], wick=True)
                if opts.get('mixT', False) and not opts.get('mixlocal', False):
                    h = hist[k - 1]; full = full + mixterm(Z, h['T'], h['S'], h.get('Q'), h.get('K31'), h['m'], h['aux'], opts)
                if opts.get('kx', False) and k - 1 >= 1 and not opts.get('mixlocal', False) and not opts.get('kxlocal', False):
                    full = full + kxterm(Z, Ws[k - 1], hist[k - 2]['M'], hist[k - 1]['aux'])
                Tk = hist[k]['T']
                cm, cd = pcp.pair3(U, Tk, np.diag(Tk)); np.fill_diagonal(cm, cd)
                Tn = Tn + full - cm
                U = hist[k - 1]['aux']['P'][:, None] * Z; k -= 1
            m, S, T, Q, K31 = m_new, S_new, Tn, Qn, (K31n if K31n is not None else np.zeros((n, n)))
    return Er * np.array(rows)

# ---------------- bivariate Edgeworth transfer (v4) ----------------
def coeffs(m, s, a, P, p, KK):
    """Hermite coefficient lists c_k = s^k E[phi^{(k)}(y)], k=0..KK for relu (f), (relu-muG)^2 (g), (relu-muG)^3 (u)."""
    H = pcp.Hes(a, KK + 2)
    muG = m * P + s * p
    Ed = lambda j: (-1) ** j * H[j] * p / s ** (j + 1)
    f = [muG, s * P] + [s ** k * Ed(k - 2) for k in range(2, KK + 1)]
    r2g = (m * m + s * s) * P + m * s * p
    Eg = [None, 2 * ((m - muG) * P + s * p), 2 * P - 2 * muG * Ed(0)] + [2 * Ed(k - 3) - 2 * muG * Ed(k - 2) for k in range(3, KK + 1)]
    g = [None] + [s ** k * Eg[k] for k in range(1, KK + 1)]
    Eu = [None, 3 * (r2g - 2 * muG * muG + muG ** 2 * P), 6 * (muG - muG * P) + 3 * muG ** 2 * Ed(0),
          6 * P - 6 * muG * Ed(0) + 3 * muG ** 2 * Ed(1)] + \
         [6 * Ed(k - 4) - 6 * muG * Ed(k - 3) + 3 * muG ** 2 * Ed(k - 2) for k in range(4, KK + 1)]
    u = [None] + [s ** k * Eu[k] for k in range(1, KK + 1)]
    return f, g, u

def pairE(A, B, R, tau, om, rho, K, second=True):
    out = np.zeros_like(R); Rk = np.ones_like(R); tt = tau.T; rt = None if rho is None else rho.T
    for k in range(0, K + 1):
        if k >= 1: Rk = Rk * R / k
        term = tau * np.outer(A[k + 2], B[k + 1]) + tt * np.outer(A[k + 1], B[k + 2]) + om * np.outer(A[k + 2], B[k + 2])
        if rho is not None:
            term += rho * np.outer(A[k + 3], B[k + 1]) + rt * np.outer(A[k + 1], B[k + 3])
        if second:
            term += 0.5 * tau * tau * np.outer(A[k + 4], B[k + 2]) + 0.5 * tt * tt * np.outer(A[k + 2], B[k + 4]) + tau * tt * np.outer(A[k + 3], B[k + 3])
        if k >= 1: term = term + np.outer(A[k], B[k])
        out += Rk * term
    return out

def transfer4(m, S, T, Q, K31, K=6, opts={}):
    s = np.sqrt(np.diag(S)); a = m / s; P = ndtr(a); p = pdf(a)
    KK = K + 10
    f, g, u = coeffs(m, s, a, P, p, KK)
    muG = f[0]
    if opts.get('k4lam') is not None:
        lam_ = opts['k4lam'][opts.get('_layer', 0)]
        v_ = np.diag(S)
        if opts.get('k4lam_off', False):
            Q = (lam_ / 3) * (np.outer(v_, v_) + 2 * S * S)
        Q = Q.copy(); np.fill_diagonal(Q, lam_ * v_ * v_ + opts.get('k4k3c', 0.0) * np.diag(T) ** 2 / v_)
    if opts.get('k3beta') is not None:
        b_ = opts['k3beta'][opts.get('_layer', 0)]
        if opts.get('k3beta_off', False):
            v_ = np.diag(S); T = T + b_ * (2 * m[:, None] * S + m[None, :] * v_[:, None]) / 3
        else:
            T = T.copy(); np.fill_diagonal(T, np.diag(T) + b_ * m * np.diag(S))
    k3 = np.diag(T).copy(); k4 = np.diag(Q).copy()
    r1 = m * P + s * p; r2 = (m * m + s * s) * P + m * s * p; r3 = (m ** 3 + 3 * m * s * s) * P + (m * m + 2 * s * s) * s * p
    r4 = (m ** 4 + 6 * m * m * s * s + 3 * s ** 4) * P + (m ** 3 + 5 * m * s * s) * s * p
    r1 = r1 + k3 / 6 * (-a * p / s ** 2) + k4 / 24 * p * (a * a - 1) / s ** 3
    r2 = r2 + k3 / 6 * (2 * p / s) + k4 / 24 * (-2 * a * p / s ** 2)
    r3 = r3 + k3 / 6 * (6 * P) + k4 / 24 * (6 * p / s)
    r4 = r4 + k3 / 6 * (24 * muG) + k4 / 24 * (24 * P)
    mu = r1; c2 = r2 - mu ** 2; c3 = r3 - 3 * mu * r2 + 2 * mu ** 3; c4 = r4 - 4 * mu * r3 + 6 * mu * mu * r2 - 3 * mu ** 4
    kh4 = c4 - 3 * c2 ** 2
    c3n = k3 / (6 * s ** 3) if opts.get('marg', True) else 0 * s
    c4n = k4 / (24 * s ** 4) if opts.get('marg', True) else 0 * s
    hat = lambda c: [None if c[k] is None else (c[k] + (c3n * c[k + 3] + c4n * c[k + 4] if k + 4 <= KK else 0)) for k in range(0, KK - 3)]
    fh, gh, uh = hat(f), hat(g), hat(u)
    R = S / np.outer(s, s); np.fill_diagonal(R, 0)
    ss = np.outer(s, s)
    tau = 0.5 * T / (ss * s[:, None]); om = 0.25 * Q / (ss * ss)
    rho = K31 / (6 * ss * s[:, None] * s[:, None]) if K31 is not None else None
    sec = opts.get('second', True)
    C = pairE(fh, fh, R, tau, om, rho, K, sec)
    G = pairE(gh, fh, R, tau, om, rho, K, sec)
    Mc = pairE(gh, gh, R, tau, om, rho, K, sec)
    D = pairE(uh, fh, R, tau, om, rho, K, sec)
    np.fill_diagonal(C, c2)
    M = Mc - 2 * C * C
    np.fill_diagonal(G, c3); np.fill_diagonal(M, kh4)
    K31h = D - 3 * c2[:, None] * C
    np.fill_diagonal(K31h, kh4)
    A1 = gh[1] / s; A2 = gh[2] / s ** 2
    return mu, C, G, M, K31h, dict(s=s, P=fh[1] / s, q=fh[2] / s ** 2, A1=A1, A2=A2)

TRANSFER = {'v3': transfer3, 'v4': transfer4}

if __name__ == '__main__':
    import sys, time
    ids = [int(x) for x in sys.argv[1].split(',')]
    confs = sys.argv[2].split(';') if len(sys.argv) > 2 else ['']
    for c in confs:
        o = eval('dict(' + c + ')')
        for i in ids:
            t = time.time(); p = run_v3(W(i), opts=o); tr = TRUTH[list(IDS).index(i)]
            e = np.mean((p - tr) ** 2, 1)
            print(c or 'full', i, 'final %.3e' % e[-1], 'L4 %.2e L8 %.2e L12 %.2e' % (e[3], e[7], e[11]), '%.0fs' % (time.time() - t), flush=True)

