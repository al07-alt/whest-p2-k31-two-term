import numpy as np, fac3, sys, time
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
for i in [int(x) for x in sys.argv[1].split(',')]:
    t=time.time(); rec=[]
    p=fac3.run_fac3(W(i),opts=dict(EX),rec=rec)
    np.savez(f'data/feat/f{i:03d}.npz', pred=p, m=np.stack([r['m'] for r in rec]), v=np.stack([r['v'] for r in rec]), k3=np.stack([r['k3'] for r in rec]), k4=np.stack([r['k4'] for r in rec]))
    tr=TRUTH[list(IDS).index(i)]; print(i,'final %.3e'%np.mean((p[-1]-tr[-1])**2),'%.0fs'%(time.time()-t),flush=True)
