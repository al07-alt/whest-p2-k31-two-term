import numpy as np
from math import comb
for i in (0,1):
    B=np.load(f'data/bk/marg_{i:03d}.npz')
    def truemom(l):
        s_=B['pre_s'][:,l]; c=B['pre_shift'][l].astype(float); m1=s_[0]
        mu=[sum(comb(p,q)*(s_[q-1] if q>0 else 1)*(-m1)**(p-q) for q in range(0,p+1)) for p in range(5)]
        return c+m1, mu[2], mu[3], mu[4]-3*mu[2]**2
    D0=np.load(f'data/v29dbg_base_{i:03d}.npz'); D1=np.load(f'data/v29dbg_orall_{i:03d}.npz')
    print('net',i,'layer | var rel err rms base -> oracle | var rel bias base -> oracle | mean err/s rms base -> oracle | corr(dvar_change, alpha^2) ')
    for l in range(1,16):
        m,v,k3,k4=truemom(l)
        e0=D0[f'var_{l}']/v-1; e1=D1[f'var_{l}']/v-1
        u0=(D0[f'mu_{l}']-m)/np.sqrt(v); u1=(D1[f'mu_{l}']-m)/np.sqrt(v)
        a=m/np.sqrt(v)
        print('  %2d | %.2e -> %.2e | %+.2e -> %+.2e | %.2e -> %.2e | %.3f'%(l,np.sqrt(np.mean(e0**2)),np.sqrt(np.mean(e1**2)),e0.mean(),e1.mean(),np.sqrt(np.mean(u0**2)),np.sqrt(np.mean(u1**2)),np.corrcoef(e1-e0,a*a)[0,1]))
