import numpy as np, fac3, pcp3
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
fac3.OLD.clear(); fac3.run_fac3(W(0),opts=dict(EX,dump_srcs_at=10)); d=fac3.OLD[-1]
sr=d['srcs'][-1]  # newest source (age 1), born at layer 9 post
# reconstruct a at birth layer from feat dump
F=np.load('data/feat/f000.npz'); a=F['m'][9]/np.sqrt(F['v'][9])
J,A,B,U,q=sr['J'],sr['A'],sr['B'],sr['U'],sr['q']
def contrib(mask):
    s=dict(sr); s2=fac3.Src(); 
    for k in ('J','A','B','U'): setattr(s2,k,None if sr[k] is None else sr[k]*mask[None,:])
    s2.q=q*mask; return fac3.slices(s2)
Tall=contrib(np.ones(1024)); nrm=np.linalg.norm(Tall)
for lo,hi in [(-99,-3),(-3,-2),(-2,-1),(-1,1),(1,2),(2,3),(3,99)]:
    mk=((a>=lo)&(a<hi)).astype(float)
    print('a in [%g,%g): frac %.3f  |slice contrib|/|all| %.4f'%(lo,hi,mk.mean(),np.linalg.norm(contrib(mk))/nrm))
