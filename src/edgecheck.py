import numpy as np
from scipy.special import ndtr
from numpy.polynomial.hermite_e import hermeval
pdf=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
He=lambda k,x: hermeval(x,[0]*k+[1])
for i in range(6):
    d=np.load(f'data/bk/marg_{i:03d}.npz')
    out=[]
    for l in [1,4,8,12,15]:
        s_=d['pre_s'][:,l]; c=d['pre_shift'][l].astype(float)
        m1=s_[0]; cm=[None,0,s_[1]-m1**2, s_[2]-3*m1*s_[1]+2*m1**3, s_[3]-4*m1*s_[2]+6*m1**2*s_[1]-3*m1**4, None, None]
        # central moments up to 6 via binomial
        from math import comb
        mu=[sum(comb(p,q)*s_[q-1]*(-m1)**(p-q) if q>0 else (-m1)**p for q in range(0,p+1)) for p in range(7)]
        k2=mu[2]; k3=mu[3]; k4=mu[4]-3*mu[2]**2; k5=mu[5]-10*mu[3]*mu[2]; k6=mu[6]-15*mu[4]*mu[2]-10*mu[3]**2+30*mu[2]**3
        mean=c+m1; s=np.sqrt(k2); a=mean/s; p=pdf(a); P=ndtr(a)
        g3,g4,g5,g6=k3/s**3,k4/s**4,k5/s**5,k6/s**6
        T=lambda k: s*(-1)**k*He(k-2,a)*p
        base=mean*P+s*p
        e1=base+g3/6*T(3)+g4/24*T(4)
        e2=e1+(g3**2/72+g6/720)*T(6)+g5/120*T(5)+(g3*g4/144)*T(7)+(g4**2/1152+g3*g5/720)*T(8)+(g3**3/1296)*T(9)
        e1b=e1+(g3**2/72)*T(6)
        tru=d['post_mean'][l]
        r=lambda x: np.sqrt(np.mean((x-tru)**2))
        out.append((l,r(base),r(e1),r(e1b),r(e2),np.sqrt(np.mean(g3**2)),np.mean(g4)))
    for o in out: print(i,'L%d gauss %.2e edge1 %.2e +k3^2 %.2e edge2full %.2e | g3rms %.3f g4 %.3f'%o)
