import numpy as np, fac3, sys, time
from lib import *
import truek31
lam=np.load('data/lam_table.npy')
i=int(sys.argv[1]); tr=TRUTH[list(IDS).index(i)]
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam,edgecoef=True,k31_cross_=True)
kt=lambda l: truek31.k31(i,l)
for Ls in [list(map(int,x.split('+'))) for x in sys.argv[2].split(',')]:
    t=time.time(); p=fac3.run_fac3(W(i),opts=dict(EX,k31_cross=True,k31_mode='oracle',k31_true=kt,k31_layers=Ls)); e=np.mean((p-tr)**2,1)
    np.save(f'data/orl_{i}_{"-".join(map(str,Ls))}.npy',p[-1])
    print(i,'oracle layers',Ls,'final %.4e'%e[-1],'%.0fs'%(time.time()-t),flush=True)
