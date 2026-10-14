import numpy as np
from lib import *
rng=np.random.default_rng(1)
def ci(i,a,b,pre='or'):
    tr=TRUTH[list(IDS).index(i)][-1]; pa=np.load(f'data/{pre}_{i}_{a}.npy'); pb=np.load(f'data/{b}' if b.endswith('.npy') else f'data/{pre}_{i}_{b}.npy')
    ea=(pa-tr)**2; eb=(pb-tr)**2; r=eb.mean()/ea.mean()-1
    bs=[]
    for _ in range(2000):
        k=rng.integers(0,1024,1024); bs.append(eb[k].mean()/ea[k].mean()-1)
    return r,np.percentile(bs,2.5),np.percentile(bs,97.5)
for i in [0,1,2]:
    for b in ['edge+rad','edge+true']:
        r,lo,hi=ci(i,'base',b); print('net',i,'%-10s vs base: %+.1f%%  95%% CI [%+.1f, %+.1f]'%(b,100*r,100*lo,100*hi))
for L in ['8-9-10-11-12-13-14-15','1-2-3-4-5-6-7','10']:
    r,lo,hi=ci(0,'edge+rad',f'orl_0_{L}.npy'); print('net 0 oracle layers',L,'vs edge+rad: %+.1f%% [%+.1f, %+.1f]'%(100*r,100*lo,100*hi))
