import numpy as np, fac3, sys, time
from scipy.special import ndtr
from lib import *
lam=np.load('data/lam_table.npy')
EX=dict(mixT=True,res_rank=16,mix_rank=16,k31_cross=False,k31_h=False,k31_lin=False,k31_q=False,k31_qd=False,k4lam=lam)
i=int(sys.argv[1]); Ws=W(i); F=np.load(f'data/feat/f{i:03d}.npz'); Phi=ndtr(F['m']/np.sqrt(F['v']))
D=np.diag(Phi[15]); sens=[None]*16; sens[15]=np.sqrt((D**2).sum(0))
for l in range(14,-1,-1):
    D=(D@Ws[l+1].T)*Phi[l][None,:]; sens[l]=np.sqrt((D**2).sum(0))
for l in [4,8,12]:
    s_=sens[l]; print('layer',l,'sens cv %.2f  top10%% share of sens^2 %.2f'%(s_.std()/s_.mean(), np.sort(s_**2)[::-1][:102].sum()/np.sum(s_**2)))
tr=TRUTH[list(IDS).index(i)]
for c in sys.argv[2].split(';'):
    o=eval('dict('+c+')'); 
    if o.pop('use_sens',False): o['sens']=sens
    t=time.time(); p=fac3.run_fac3(Ws,opts=dict(EX,**o)); e=np.mean((p-tr)**2,1); print(i,c,'final %.3e L8 %.2e L12 %.2e'%(e[-1],e[7],e[11]),'%.0fs'%(time.time()-t),flush=True)
