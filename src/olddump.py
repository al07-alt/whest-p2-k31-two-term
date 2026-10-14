import numpy as np, fac3
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam,diag_old=3,per_age_layers=(6,10,15))
fac3.run_fac3(W(0),opts=EX)
np.savez('data/old0_pa.npz', **{(f'Ts_{d["l"]}_{d["age"]}' if 'age' in d else f'{k}_{d["l"]}'):(d['Ts'] if 'age' in d else d[k]) for d in fac3.OLD for k in (('Ts',) if 'age' in d else ('Told','Tn','m','S'))})
