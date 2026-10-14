import numpy as np, fac3, sys, time, json
from lib import *
import truek31
lam=np.load('data/lam_table.npy')
i=int(sys.argv[1]); tr=TRUTH[list(IDS).index(i)]
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
kt=lambda l: truek31.k31(i,l)
V={'base':{}, 'rad':dict(k31_cross=True,k31_mode='radial'), 'edge+rad':dict(edgecoef=True,k31_cross=True,k31_mode='radial'),
   'true':dict(k31_cross=True,k31_mode='oracle',k31_true=kt), 'edge+true':dict(edgecoef=True,k31_cross=True,k31_mode='oracle',k31_true=kt)}
sel=sys.argv[2].split(',') if len(sys.argv)>2 else list(V)
out={}
for nm in sel:
    t=time.time(); p=fac3.run_fac3(W(i),opts=dict(EX,**V[nm])); e=np.mean((p-tr)**2,1)
    out[nm]=dict(final=float(e[-1]),layers=e.tolist(),pred=None); np.save(f'data/or_{i}_{nm}.npy',p[-1])
    print(i,'%-10s final %.4e  L8 %.2e L12 %.2e  %.0fs'%(nm,e[-1],e[7],e[11],time.time()-t),flush=True)
