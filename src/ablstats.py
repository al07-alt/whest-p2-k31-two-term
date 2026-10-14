import numpy as np, scipy.stats as st
rows=[l.split() for f in ('logs/or29_u.log','logs/or29_w.log','logs/or29_x.log') for l in open(f) if ' final ' in l] if True else []
d={}
for r in rows: d.setdefault(r[0],{})[int(r[1])]=float(r[3])
base=d['base']
for tag in ('two1','abl_v29k3t','abl_b1only','abl_k3075','two1r'):
    if tag not in d: continue
    nets=sorted(set(d[tag])&set(base)); ch=np.array([100*(d[tag][i]/base[i]-1) for i in nets])
    if len(ch)<2: continue
    se=ch.std(ddof=1)/np.sqrt(len(ch)); t=st.t.ppf(.975,len(ch)-1)
    print('%-12s n=%2d mean %+.2f%% CI [%+.2f, %+.2f] median %+.2f improved %d/%d'%(tag,len(ch),ch.mean(),ch.mean()-t*se,ch.mean()+t*se,np.median(ch),(ch<0).sum(),len(ch)))
# paired comparisons vs two1
for tag in ('abl_v29k3t','abl_b1only','abl_k3075','two1r'):
    if tag not in d: continue
    nets=sorted(set(d[tag])&set(d['two1'])); 
    if len(nets)<2: continue
    diff=np.array([100*(d[tag][i]/d['two1'][i]-1) for i in nets]); se=diff.std(ddof=1)/np.sqrt(len(diff)); t=st.t.ppf(.975,len(diff)-1)
    print('  %s vs two1: %+.2f%% CI [%+.2f, %+.2f]'%(tag,diff.mean(),diff.mean()-t*se,diff.mean()+t*se))
