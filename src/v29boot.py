# Paired bootstrap over the 1024 output neurons of the final-MSE change of V29 variants vs V29 (saved predictions).
import numpy as np, sys
T=np.load('data/truth.npz'); tr=T['means']
tags=sys.argv[1].split(','); B=4000; rng=np.random.default_rng(0)
for tag in tags:
    out=[]
    for i in (0,1,2):
        try:
            b=np.load(f'data/pred/base_{i}.npy')[-1]; v=np.load(f'data/pred/{tag}_{i}.npy')[-1]
        except FileNotFoundError: out.append('  n/a'); continue
        t=tr[i][-1]; eb=(b-t)**2; ev=(v-t)**2
        idx=rng.integers(0,len(t),(B,len(t))); r=ev[idx].mean(1)/eb[idx].mean(1)
        out.append('%+.1f%% [%+.1f, %+.1f]'%(100*(ev.mean()/eb.mean()-1),100*(np.quantile(r,.025)-1),100*(np.quantile(r,.975)-1)))
    print('%-14s'%tag,' | '.join(out),flush=True)
