import numpy as np
rows=np.load('data/feat/rows.npy',allow_pickle=True)
ids=sorted(set(r['i'] for r in rows)); tr_ids=ids[:len(ids)//2+1]; te_ids=ids[len(ids)//2+1:]
for l in [3,6,9,12,15]:
    R=[r for r in rows if r['l']==l]
    cat=lambda k,S: np.concatenate([r[k] for r in R if r['i'] in S])
    for tgt in ['d4','d3','dv']:
        X=lambda S: np.stack([np.ones_like(cat('a',S)),cat('a',S),cat('a',S)**2,cat('g3',S),cat('g3',S)**2,cat('g3',S)*cat('a',S),cat('g4',S)],1)
        Xtr,ytr=X(tr_ids),cat(tgt,tr_ids); Xte,yte=X(te_ids),cat(tgt,te_ids)
        b=np.linalg.lstsq(Xtr,ytr,rcond=None)[0]; r2=1-np.mean((yte-Xte@b)**2)/np.mean(yte**2)
        print(l,tgt,'rms %.2e'%np.sqrt(np.mean(yte**2)),'g3 rms %.2e g4 %.2e'%(np.sqrt(np.mean(cat('g3',te_ids)**2)),np.mean(cat('g4',te_ids))),'R2 %.3f'%r2, np.round(b,4))
