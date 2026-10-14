import numpy as np, fac3, sys, time
from lib import *
lam=np.load('data/lam_table.npy')
i=int(sys.argv[1]); tr=TRUTH[list(IDS).index(i)]
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=True,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam,edgecoef=True,k31model=True)
V={'model:rad':dict(k31b_scale=0.0),'model:rad+tree':dict(k31b_scale=0.0,k31tree=1.0),'model:rad+births':dict(k31b_scale=1.0),'model:rad+tree+births':dict(k31b_scale=1.0,k31tree=1.0)}
for nm in sys.argv[2].split(','):
    t=time.time(); p=fac3.run_fac3(W(i),opts=dict(EX,**V[nm])); e=np.mean((p-tr)**2,1); np.save(f'data/jt_{i}_{nm.replace(":","_").replace("+","-")}.npy',p[-1])
    print(i,'%-24s final %.4e L8 %.2e L12 %.2e %.0fs'%(nm,e[-1],e[7],e[11],time.time()-t),flush=True)
