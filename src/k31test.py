import numpy as np, fac3, sys, time
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
K31ON=dict(k31_cross=True,k31_h=True,k31_lin=True,k31_q=True,k31_qd=True)
i=int(sys.argv[1]); tr=TRUTH[list(IDS).index(i)]
for c in sys.argv[2].split(';'):
    o=eval('dict('+c+')'); t=time.time()
    p=fac3.run_fac3(W(i),opts=dict(EX,**o)); e=np.mean((p-tr)**2,1); print(i,c,' '.join('%.2e'%x for x in e),'%.0fs'%(time.time()-t),flush=True)
