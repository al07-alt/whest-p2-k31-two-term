import numpy as np, fac3, sys, time
from lib import *
lam=np.load('data/lam_table.npy'); g1=np.load('data/gamma_tab.npy'); g2=np.load('data/gamma_tab2.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
for i in [int(x) for x in sys.argv[1].split(',')]:
    tr=TRUTH[list(IDS).index(i)]
    for nm,o in [('gamma',dict(gamma_tab=g1)),('gamma+beta',dict(gamma_tab=g2))]:
        t=time.time(); p=fac3.run_fac3(W(i),opts=dict(EX,**o)); e=np.mean((p-tr)**2,1)
        print(i,nm,'final %.3e L8 %.2e L12 %.2e'%(e[-1],e[7],e[11]),'%.0fs'%(time.time()-t),flush=True)
