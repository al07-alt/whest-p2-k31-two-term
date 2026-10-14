import numpy as np
rng=np.random.default_rng(0)
for i in [0,1,2]:
    K=np.load(f'data/kp/marg_{i:03d}.npz'); B=np.load(f'data/bk/marg_{i:03d}.npz')
    out=[]
    for l in [6,9,12,15]:
        v1=K['pre_m2'][l]-K['pre_mean'][l]**2; v2=B['pre_var'][l]
        d=(v1/v2-1); f=np.sqrt(np.mean(d**2))
        bs=[np.sqrt(np.mean(d[rng.integers(0,1024,1024)]**2)) for _ in range(300)]
        out.append('L%d %.2e [%.2e,%.2e]'%(l,f,np.percentile(bs,2.5),np.percentile(bs,97.5)))
    print('net',i,'bake-vs-bake rel var diff:',' | '.join(out))
