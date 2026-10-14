import numpy as np, sys, time
from lib import W
# Conditional-on-mean identities: for z_a = w_a.h with w_a = t_a muhat + w_perp, averaging over the direction of w_perp:
#   E[k3_a] = t^3 k3(u) + 3 t om Cov(u,|dp|^2)      E[k4_a] = t^4 k4(u) + 6 t^2 om A + 3 om^2 B
# u = muhat.d, dp = d - u muhat, om = |w_perp|^2/(n-1), A = sum_p k(u,u,dp_p,dp_p), B = sum_pq k(dp_p,dp_p,dp_q,dp_q)
i=int(sys.argv[1]); N=int(sys.argv[2]); SEED=int(sys.argv[3]) if len(sys.argv)>3 else 5; Wt=W(i,np.float32); n=1024; B_=10000
zt=np.load(f'./data/kp/marg_{i:03d}.npz'); r1,r2,r3,r4=zt['pre_mean'],zt['pre_m2'],zt['pre_m3'],zt['pre_m4']
c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1*r1*r2-3*r1**4-3*c2**2
def run(seed):
    rng=np.random.default_rng(seed); S1=np.zeros((17,n))
    hs=[]
    for b in range(N//B_):
        h=rng.standard_normal((B_,n)).astype(np.float32); row=[h]
        for k in range(16): h=np.maximum(h@Wt[k],0); row.append(h)
        for k in range(17): S1[k]+=row[k].sum(0,dtype=np.float64)
    return S1/N
t0=time.time(); mu=run(SEED)
# second pass: scalars
acc={k:[] for k in range(16)}
rng=np.random.default_rng(SEED)
stats=np.zeros((16,12))
for b in range(N//B_):
    h=rng.standard_normal((B_,n)).astype(np.float32)
    for k in range(16):
        hd=h.astype(np.float64); mh=mu[k]/np.linalg.norm(mu[k]); d=hd-mu[k]; u=d@mh; dp2=(d*d).sum(1)-u*u
        Cu=(d.T@u)/1.0  # sum over samples of d_p u
        stats[k]+=np.array([u.sum(),(u**2).sum(),(u**3).sum(),(u**4).sum(),dp2.sum(),(dp2**2).sum(),(u*dp2).sum(),(u*u*dp2).sum(),0,0,0,0])
        acc[k].append(Cu)
        h=np.maximum(h@Wt[k],0)
print('pass %.0fs'%(time.time()-t0))
# need ||C_perp||_F^2 and C_perp muhat: do a covariance pass (exact second moments)
rng=np.random.default_rng(SEED); Cs=[np.zeros((n,n)) for _ in range(16)]
for b in range(N//B_):
    h=rng.standard_normal((B_,n)).astype(np.float32)
    for k in range(16):
        d=(h.astype(np.float64)-mu[k]); Cs[k]+=d.T@d; h=np.maximum(h@Wt[k],0)
SAVE={}
print('layer | k3 identity: corr R2 slope | alpha-baseline corr | k4 identity: corr slope (eta) | lam_pred vs true')
for k in range(1,16):
    s=stats[k]/N; C=Cs[k]/N; mh=mu[k]/np.linalg.norm(mu[k]); nm=np.linalg.norm(mu[k])
    Eu,Eu2,Eu3,Eu4,Edp,Edp2,Eudp,Eu2dp=s[:8]
    varu=Eu2-Eu**2; k3u=Eu3-3*Eu*Eu2+2*Eu**3; k4u=Eu4-3*Eu2**2   # u has ~zero mean
    covudp=Eudp-Eu*Edp
    P=np.eye(n)-np.outer(mh,mh); Cp=P@C@P; trCp=np.trace(Cp)
    Cpu=P@(C@mh)   # E[dp u]
    A=(Eu2dp-Eu2*Edp)-2*(Cpu@Cpu)
    B=(Edp2-Edp**2)-2*np.sum(Cp*Cp)
    w=Wt[k].astype(np.float64)  # z_k = h_k . w ; stored h index k -> z index k
    t=(w.T@mh); om=((w*w).sum(0)-t*t)/(n-1)
    l=k   # z_l = W_l^T h_l (h_l stored at index l)
    k3p=t**3*k3u+3*t*om*covudp; vp=t*t*varu+om*trCp
    k4p=t**4*k4u+6*t*t*om*A+3*om*om*B
    v=c2[l]; g=c3[l]/v**1.5; gp=k3p/v**1.5; e=c4[l]/v**2; ep=k4p/v**2; al=r1[l]/np.sqrt(v)
    SAVE[f'g_{l}']=g; SAVE[f'gp_{l}']=gp; SAVE[f'a_{l}']=al; SAVE[f'e_{l}']=e; SAVE[f'ep_{l}']=ep
    r=np.corrcoef(g,gp)[0,1]; sl=np.sum(g*gp)/np.sum(gp*gp); R2=1-np.mean((g-gp)**2)/np.var(g)
    ra=np.corrcoef(g,al)[0,1]
    re=np.corrcoef(e,ep)[0,1]; se=np.sum(e*ep)/np.sum(ep*ep)
    print('z%-2d k3: corr %.3f R2 %.3f slope %.3f | alpha corr %.3f | k4: corr %.3f slope %.3f R2 %.3f | mean eta %.4f pred %.4f | v corr %.3f'%(l,r,R2,sl,ra,re,se,1-np.mean((e-ep)**2)/np.var(e),e.mean(),ep.mean(),np.corrcoef(v,vp)[0,1]),flush=True)

np.savez(f'./data/skewid_{i:03d}.npz' if SEED==5 else f'./data/skewid_{i:03d}_s{SEED}.npz',**SAVE)
