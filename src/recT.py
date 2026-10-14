import numpy as np, fac3
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam,recfull=True)
rec=[]; import sys; I=int(sys.argv[1]) if len(sys.argv)>1 else 0; fac3.run_fac3(W(I),opts=EX,rec=rec)
np.savez('data/rec%d_full.npz'%I+'',T=np.stack([r['T'] for r in rec]),S=np.stack([r['S'] for r in rec]),m=np.stack([r['m'] for r in rec]))
