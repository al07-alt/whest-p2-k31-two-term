import numpy as np, glob, sys
from scipy.special import ndtr
from lib import *
lam=np.load('data/lam_table.npy')
ids=sorted(int(f[-7:-4]) for f in glob.glob('data/feat/f*.npz'))
ids=[i for i in ids if i<30]
rows=[]
for i in ids:
    F=np.load(f'data/feat/f{i:03d}.npz'); K=np.load(f'data/kp/marg_{i:03d}.npz')
    tr=TRUTH[list(IDS).index(i)]
    for l in range(1,16):
        m=F['m'][l]; v=F['v'][l]; s=np.sqrt(v); k3=F['k3'][l]; k4=lam[l]*v*v
        r1,r2,r3,r4=K['pre_mean'][l],K['pre_m2'][l],K['pre_m3'][l],K['pre_m4'][l]
        c2=r2-r1**2; c3=r3-3*r1*r2+2*r1**3; c4=r4-4*r1*r3+6*r1**2*r2-3*r1**4-3*c2**2
        a=m/s; P=ndtr(a); p=pdf(a)
        dm=r1-m; dv=c2-v; d3=c3-k3; d4=c4-k4
        tm=[P*dm, p/(2*s)*dv, -a*p/(6*s*s)*d3, p*(a*a-1)/(24*s**3)*d4]
        e=tr[l]-F['pred'][l]
        rows.append(dict(i=i,l=l,a=a,s=s,g3=k3/s**3,g4=k4/s**4,dm=dm/s,dv=dv/v,d3=d3/s**3,d4=d4/s**4,tm=tm,e=e,m=m))
for l in range(1,16):
    R=[r for r in rows if r['l']==l]
    f=lambda k: np.sqrt(np.mean(np.concatenate([r[k] for r in R])**2))
    tm=[np.sqrt(np.mean(np.concatenate([r['tm'][j] for r in R])**2)) for j in range(4)]
    tot=np.sqrt(np.mean(np.concatenate([r['e'] for r in R])**2))
    print(l,'rms dm/s %.2e dv/v %.2e d3 %.2e d4 %.2e | mean-err terms m %.2e v %.2e k3 %.2e k4 %.2e | post err %.2e'%(f('dm'),f('dv'),f('d3'),f('d4'),*tm,tot))
np.save('data/feat/rows.npy',rows,allow_pickle=True)
