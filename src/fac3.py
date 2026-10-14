"""Forward-factored transport with transfer3 (+K31, Wick4, mixT, sphere). Sources carry legs at the current y-level."""
import numpy as np
from lib import *
import pcp, pcp3
Z0 = lambda X: X - np.diag(np.diag(X))

PR = []
RAD = [0.0, 0.0]
OLD = []
class Src:
    pass

def births(T, S, G, m, aux, opts):
    """Return birth legs (h-level) for the source born at this layer."""
    P, q, s = aux['P'], aux['q'], aux['s']
    St = Z0(S); A = P[:, None] * St                      # Wick A-leg, also mix V-leg
    Gc = G - (P * P)[:, None] * T * P[None, :]
    X = (A * A) * q[None, :]
    U = None
    if opts.get('mixT', True):
        a_ = m / s; F3 = -a_ * pdf(a_) / s ** 2
        Tt = Z0(T)
        U = 0.5 * (P[:, None] * Tt.T * F3[None, :] + q[:, None] * Tt * q[None, :])
        Gc = Gc - 2 * (U * A)                            # remove y=z coincidences of the mix block
    Gx = Gc - X
    B = 3 * Gx.T - 2 * np.diag(np.diag(Gx))
    rr = opts.get('res_rank')
    if rr is not None:
        g1s = 2 * ((m - (m * P + s * pdf(m / s))) * P + s * pdf(m / s))   # g1/s = E[g'] 
        B1 = 3 * A * g1s[None, :]
        Bres = B - B1; dres = np.diag(Bres).copy(); Bres = Z0(Bres)
        B = B1 + np.diag(dres) + lowrank(Bres, rr)
    mr = opts.get('mix_rank')
    if mr is not None and U is not None: U = lowrank(U, mr)
    return A, B, U, q

def lowrank(X, r):
    if r == 0: return np.zeros_like(X)
    Uu, s_, Vt = np.linalg.svd(X, full_matrices=False)
    return (Uu[:, :r] * s_[:r]) @ Vt[:r]

def slices(src):
    J, A, B, U, q = src.J, src.A, src.B, src.U, src.q
    Jq = J * q[None, :]
    T = ((J * J) @ B.T + 2 * (J * B) @ J.T) / 3
    T += (A * A) @ Jq.T + 2 * (A * Jq) @ A.T
    if U is not None:
        T += 2 * ((J * U) @ A.T + (J * A) @ U.T + (U * A) @ J.T)
    return T

def run_fac3(Ws, K=6, opts=None, rec=None):
    opts = dict(pcp3.OPT, **(opts or {}))
    n = Ws.shape[1]; L = len(Ws); rows = []
    w = Ws[0]; m = np.zeros(n); S = w.T @ w; T = np.zeros((n, n)); Q = np.zeros((n, n)); K31 = np.zeros((n, n))
    Er = 1.0
    if opts.get('sphere', False):
        from scipy.special import gammaln
        Er = np.exp(0.5 * np.log(2 / n) + gammaln((n + 1) / 2) - gammaln(n / 2))
        d = np.diag(S).copy(); c = -2.0 / (n + 2)
        Q = c * (np.outer(d, d) + 2 * S * S); np.fill_diagonal(Q, 3 * c * d * d)
        K31 = 3 * c * d[:, None] * S; np.fill_diagonal(K31, 3 * c * d * d)
    srcs = []
    RAD[0] = 0.0; RAD[1] = 0.0
    ufeed = None
    for l in range(L):
        if opts.get('k4sched') is not None:
            opts['k4scale'] = 1 + opts['k4sched'] * max(0, l + 1 - 3)
        if opts.get('k4table') is not None:
            opts['k4scale'] = opts['k4table'][l]
        opts['_layer'] = l
        mu, C, G, M, K31h, aux = pcp3.TRANSFER[opts.get('tr', 'v3')](m, S, T, Q, K31, K, opts)
        rows.append(mu)
        if rec is not None: rec.append(dict(m=m, v=np.diag(S).copy(), k3=np.diag(T).copy(), k4=np.diag(Q).copy(), **({'S': S.astype(np.float32), 'T': T.astype(np.float32)} if opts.get('recfull') else {})))
        if opts.get('oracle_from') is not None and l >= opts['oracle_from']:
            mu = opts['truth'][l] / Er
        if l + 1 == L: break
        w = Ws[l + 1]; P = aux['P']
        if opts.get('Ptrans_edge', False):
            s_ = aux['s']; a_ = m / s_; p_ = pdf(a_)
            k3_ = np.diag(T); k4_ = (np.diag(Q) if opts.get('k4lam') is None else opts['k4lam'][l] * s_ ** 4)
            P = P + (k3_ / (6 * s_ ** 3)) * p_ * (a_ * a_ - 1) - (k4_ / (24 * s_ ** 4)) * p_ * (a_ ** 3 - 3 * a_)
            if opts.get('Pbirth_edge', False): aux = dict(aux, P=P)
        P = P * (1 + opts.get('tboost', 0.0))
        # transport existing sources (y-level l -> y-level l+1)
        if opts.get('k31model') and srcs:
            from scipy.special import ndtr as _ndtr
            s_k = np.sqrt(np.diag(S)); a_k = m / s_k; f1k = _ndtr(a_k); f2k = pdf(a_k) / s_k
            Wtk = w * f1k[:, None]; Gk = Wtk.T @ S; Hmk = f2k[:, None] * w * Gk.T
            T1k = np.zeros((n, n)); Lamk = np.zeros((n, n))
            for sc in srcs:
                J, A_, B_, U_, q_ = sc.J, sc.A, sc.B, sc.U, sc.q; V_ = J * q_[None, :]
                trip = [(J, J, B_, 1 / 3), (J, B_, J, 1 / 3), (B_, J, J, 1 / 3), (A_, A_, V_, 1.), (A_, V_, A_, 1.), (V_, A_, A_, 1.)]
                if U_ is not None: trip += [(J, U_, A_, 1.), (J, A_, U_, 1.), (U_, J, A_, 1.), (U_, A_, J, 1.), (A_, J, U_, 1.), (A_, U_, J, 1.)]
                cache = {}; rc = {}
                tr = lambda X: cache.setdefault(id(X), Wtk.T @ X)
                for X, Y, Z, c in trip:
                    Yp = tr(Y); Zp = tr(Z); R = rc.setdefault(id(X), Hmk.T @ X)
                    T1k += c * ((R * Yp) @ Zp.T); Lamk += c * (X @ (Yp * Zp).T)
            Nk = f2k[:, None] * w * Lamk
            K31B = opts.get('k31b_scale', 1.0) * (6 * T1k + 3 * (Nk.T @ Gk.T) + 3 * ((Gk * Lamk.T) @ (f2k[:, None] * w)))
        else:
            K31B = None
        for sc in srcs:
            for nm in ('J', 'A', 'B', 'U'):
                X = getattr(sc, nm)
                if X is not None: setattr(sc, nm, w.T @ (P[:, None] * X))
            if opts.get('track_G') and getattr(sc, 'G', None) is not None:
                sc.G = w.T @ ((P[:, None] * sc.G * P[None, :]) @ w)
        a0 = opts.get('old_age')
        if a0 is not None:
            ages = [len(srcs) - i + 1 for i in range(len(srcs))]   # age after this transport (y-level l+1): newest existing has age 2
            cand = [sc_ for sc_, ag in zip(srcs, ages) if ag == a0]
            if cand:
                r = opts.get('old_rank', 256)
                if opts.get('sens') is not None:
                    sw = opts['sens'][l + 1]
                    Uu, sv, _ = np.linalg.svd(sw[:, None] * cand[0].J, full_matrices=False)
                else:
                    Uu, sv, _ = np.linalg.svd(cand[0].J, full_matrices=False)
                if opts.get('radcomp'):
                    m_ = w.T @ mu; S_ = w.T @ C @ w; v_ = np.diag(S_)
                    R1_ = 2 * m_[:, None] * S_ + m_[None, :] * v_[:, None]; R2_ = np.outer(m_ * m_, m_)
                    Tb_ = sum((slices(sc_) for sc_, ag in zip(srcs, ages) if ((ag == a0) if opts.get('once', False) else (ag >= a0))), np.zeros((n, n)))
                for sc_, ag in zip(srcs, ages):
                    if (ag == a0) if opts.get('once', False) else (ag >= a0):
                        rr = r(ag) if callable(r) else r
                        Ub = Uu[:, :rr]
                        for nm in ('J', 'A', 'B', 'U'):
                            X = getattr(sc_, nm)
                            if X is None: continue
                            if opts.get('sens') is not None:
                                sw = opts['sens'][l + 1]
                                setattr(sc_, nm, (Ub @ (Ub.T @ (sw[:, None] * X))) / sw[:, None])
                            else:
                                setattr(sc_, nm, Ub @ (Ub.T @ X))
                if opts.get('radcomp'):
                    Ta_ = sum((slices(sc_) for sc_, ag in zip(srcs, ages) if ((ag == a0) if opts.get('once', False) else (ag >= a0))), np.zeros((n, n)))
                    dT = Tb_ - Ta_; Xr = np.stack([R1_.ravel(), R2_.ravel()], 1)
                    bg = np.linalg.lstsq(Xr, dT.ravel(), rcond=None)[0]
                    RAD[0] += bg[0]; RAD[1] += bg[1]
                    PR.append((l + 1, np.linalg.norm(dT) / max(np.linalg.norm(Tb_), 1e-30), 1 - np.sum((dT.ravel() - Xr @ bg) ** 2) / max(np.sum(dT ** 2), 1e-300)))
        A, B, U, q = births(T, S, G, m, aux, opts)
        thr = opts.get('prune_thr')
        if thr is not None:
            aa = m / aux['s']; keep = ((aa > -thr) if opts.get('dead_only') else (np.abs(aa) < thr)).astype(float)
            A = A * keep[None, :]; B = B * keep[None, :]; U = None if U is None else U * keep[None, :]
            Jb = w.T * keep[None, :]
            nk = keep.sum(); PR.append(nk / len(keep))
        else:
            Jb = w.T.copy()
        sc = Src(); sc.G = (w.T @ C @ w) if opts.get('track_G') else None; sc.J = Jb; sc.A = w.T @ A; sc.B = w.T @ B; sc.U = None if U is None else w.T @ U; sc.q = q
        srcs.append(sc)
        if opts.get('final_maxage') is not None and l + 2 == L:
            nsrc = len(srcs)
            Tn = sum(slices(s_) for idx, s_ in enumerate(srcs) if nsrc - idx <= opts['final_maxage'])
        else:
            Tn = sum(slices(s_) for s_ in srcs)
        if opts.get('k4feed'):
            from scipy.special import ndtr as _nd
            s_f = np.sqrt(np.diag(S)); a_f = m / s_f; f2f = pdf(a_f) / s_f; Pf = aux['P']
            lam_f = opts['k4lam'][l] * opts.get('k4feed_scale', 1.0)
            y_f = w.T @ ((lam_f / 6.0) * f2f * np.diag(S))
            ufeed = y_f if ufeed is None else (w.T @ (Pf * ufeed) + y_f)
            S_f = w.T @ C @ w; v_f = np.diag(S_f)
            Tn = Tn + 2 * ufeed[:, None] * S_f + ufeed[None, :] * v_f[:, None] - 2 * np.diag(ufeed * v_f) * 0
        if opts.get('gamma_tab') is not None:
            m_g = w.T @ mu; gt_ = opts['gamma_tab'][l + 1]
            Tn = Tn + gt_[1] * np.outer(m_g * m_g, m_g)
            if gt_[0]:
                S_g = w.T @ C @ w; v_g = np.diag(S_g); Tn = Tn + gt_[0] * (2 * m_g[:, None] * S_g + m_g[None, :] * v_g[:, None])
        if (opts.get('radcomp') or opts.get('absorb') is not None) and (RAD[0] or RAD[1]):
            m_ = w.T @ mu; S_ = w.T @ C @ w; v_ = np.diag(S_)
            Tn = Tn + RAD[0] * (2 * m_[:, None] * S_ + m_[None, :] * v_[:, None]) + RAD[1] * np.outer(m_ * m_, m_)
        if opts.get('old_split') is not None:
            A0, kd, ko, rad = opts['old_split']; nsrc = len(srcs)
            To = sum((slices(s_) for idx, s_ in enumerate(srcs) if nsrc - idx >= A0), np.zeros((n, n)))
            if not (kd and ko):
                m_ = w.T @ mu; S_ = w.T @ C @ w; v_ = np.diag(S_)
                R1_ = 2 * m_[:, None] * S_ + m_[None, :] * v_[:, None]; R2_ = np.outer(m_ * m_, m_)
                Xr = np.stack([R1_.ravel(), R2_.ravel()], 1); bg = np.linalg.lstsq(Xr, To.ravel(), rcond=None)[0]
                Trad = bg[0] * R1_ + bg[1] * R2_ if rad else np.zeros((n, n))
                dm = np.eye(n, dtype=bool)
                repl = np.where(dm, (To if kd else Trad), (To if ko else Trad))
                Tn = Tn - To + repl
        if opts.get('dump_srcs_at') == l + 1:
            nsrc = len(srcs)
            OLD.append(dict(l=l + 1, m=w.T @ mu, S=w.T @ C @ w, srcs=[dict(age=nsrc - idx, J=s_.J.copy(), A=s_.A.copy(), B=s_.B.copy(), U=None if s_.U is None else s_.U.copy(), q=s_.q.copy()) for idx, s_ in enumerate(srcs)], Tn=Tn.copy()))
        if opts.get('absorb') is not None:
            A0 = opts['absorb']; nsrc = len(srcs)
            m_ = w.T @ mu; S_ = w.T @ C @ w; v_ = np.diag(S_)
            R1_ = 2 * m_[:, None] * S_ + m_[None, :] * v_[:, None]; R2_ = np.outer(m_ * m_, m_)
            keep_ = []
            for idx, s_ in enumerate(srcs):
                if nsrc - idx >= A0:
                    Ts_ = slices(s_); Xr = np.stack([R1_.ravel(), R2_.ravel()], 1)
                    bg = np.linalg.lstsq(Xr, Ts_.ravel(), rcond=None)[0]
                    RAD[0] += bg[0]; RAD[1] += bg[1]; Tn = Tn - Ts_ + bg[0] * R1_ + bg[1] * R2_
                else: keep_.append(s_)
            srcs = keep_
        if opts.get('radial') is not None:
            A0 = opts['radial']['A']; nsrc = len(srcs)
            Tn = sum((slices(s_) for idx, s_ in enumerate(srcs) if nsrc - idx < A0), np.zeros((n, n)))
            m_ = w.T @ mu; S_ = w.T @ C @ w; v_ = np.diag(S_)
            bt = opts['radial']['beta'][l + 1]
            if bt: Tn = Tn + bt * (2 * m_[:, None] * S_ + m_[None, :] * v_[:, None])
            srcs = [s_ for idx, s_ in enumerate(srcs) if nsrc - idx < A0]
        if opts.get('diag_old') is not None:
            nsrc = len(srcs); A0 = opts['diag_old']
            Told = sum((slices(s_) for idx, s_ in enumerate(srcs) if nsrc - idx >= A0), np.zeros_like(Tn))
            OLD.append(dict(l=l + 1, Told=Told.astype(np.float32), Tn=Tn.astype(np.float32), m=(w.T @ mu), S=(w.T @ C @ w).astype(np.float32)))
            if l + 1 in opts.get('per_age_layers', ()):
                for idx, s_ in enumerate(srcs):
                    OLD.append(dict(l=l + 1, age=nsrc - idx, Ts=slices(s_).astype(np.float32), G=(s_.G.astype(np.float32) if getattr(s_, 'G', None) is not None else None)))
        Qn, K31n = pcp3.linear_q_k31(w, M, K31h, opts, S, aux)
        if opts.get('k31model'):
            S_k = w.T @ C @ w; v_k = np.diag(S_k); lk = opts['k4lam'][l + 1] * opts.get('k31lam_scale', 1.0)
            K31n = lk * v_k[:, None] * S_k + (K31B if K31B is not None else 0)
            if opts.get('k31tree', 0.0):
                from scipy.special import ndtr as _nd2
                s_t = np.sqrt(np.diag(S)); a_t = m / s_t; p_t = pdf(a_t)
                f1t = _nd2(a_t); f2t = p_t / s_t; f3t = -a_t * p_t / s_t ** 2
                Cot = S - np.diag(np.diag(S))
                G1 = Cot @ (f1t[:, None] * w); Vt = f2t[:, None] * w; Ht = Cot @ (Vt * G1)
                Kt = 6 * (Ht * Vt).T @ G1 + 6 * (Ht * G1).T @ Vt + 3 * (f3t[:, None] * w * G1 * G1).T @ G1 + (G1 ** 3).T @ (f3t[:, None] * w)
                K31n = K31n + opts['k31tree'] * Kt
            np.fill_diagonal(K31n, lk * v_k * v_k)
        m, S, T, Q, K31 = w.T @ mu, w.T @ C @ w, Tn, Qn, K31n
    return Er * np.array(rows)

if __name__ == '__main__':
    import sys, time
    ids = [int(x) for x in sys.argv[1].split(',')]
    for c in (sys.argv[2].split(';') if len(sys.argv) > 2 else ['']):
        o = eval('dict(' + c + ')')
        for i in ids:
            t = time.time(); p = run_fac3(W(i), opts=o); tr = TRUTH[list(IDS).index(i)]
            e = np.mean((p - tr) ** 2, 1)
            print(c or 'default', i, 'final %.3e' % e[-1], 'L4 %.2e L8 %.2e L12 %.2e' % (e[3], e[7], e[11]), '%.0fs' % (time.time() - t), flush=True)
