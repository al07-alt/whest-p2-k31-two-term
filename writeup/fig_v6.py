import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, re, sys
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
C3=['#2a6f97','#c44536','#6a994e']
LOG='logs/v29boot.log'
def boot(tag):
    for line in open(LOG):
        if line.split()[0]==tag and 'n/a' not in line:
            parts=line.strip().split(None,1)[1].split('|')
            out=[]
            for p in parts:
                m=re.search(r'([+-][\d.]+)% \[([+-][\d.]+), ([+-][\d.]+)\]',p); out.append(tuple(float(m.group(k)) for k in (1,2,3)))
            last=out
    return last
tags=['s2radall','rank1','rank4','rank16','rank64','s2nonradall']; xs=np.arange(len(tags)); labels=['0','1','4','16','64','all']
f,ax=plt.subplots(figsize=(4.6,2.9))
for k in range(3):
    pts=[boot(t)[k] for t in tags]; y=[p[0] for p in pts]; lo=[p[0]-p[1] for p in pts]; hi=[p[2]-p[0] for p in pts]
    ax.errorbar(xs+(k-1)*0.08,y,yerr=[lo,hi],fmt='o-',ms=3.5,lw=1.2,capsize=2,c=C3[k],label=f'net {k}')
ax.axhline(0,c='k',lw=.6); ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_xlabel('k: top singular directions of the non-radial remainder added'); ax.set_ylabel('final MSE change vs V29 (%)')
ax.legend(frameon=False,fontsize=7.5,loc='lower left')
f.tight_layout(); f.savefig('fig/rank_curve.pdf'); f.savefig('fig/rank_curve.png',dpi=220)
# per-network two-term figure, if data present
try:
    rows=[l.split() for l in open('logs/or29_u.log') if ' final ' in l]
    base={int(r[1]):float(r[3]) for r in rows if r[0]=='base'}; two={int(r[1]):float(r[3]) for r in rows if r[0]=='two1'}
    nets=sorted(set(base)&set(two)); ch=np.array([100*(two[i]/base[i]-1) for i in nets])
    T=np.load('data/truth.npz')['means']; rng=np.random.default_rng(0)
    f,ax=plt.subplots(figsize=(4.8,2.6))
    ax.bar(range(len(nets)),ch,color=['#2a6f97' if c<0 else '#c44536' for c in ch]); ax.axhline(0,c='k',lw=.6)
    ax.axhline(ch.mean(),c='gray',ls='--',lw=.9); ax.text(len(nets)-0.5,ch.mean()-0.4,'mean %.1f %%'%ch.mean(),ha='right',fontsize=7.5,color='gray')
    ax.set_xticks(range(len(nets))); ax.set_xticklabels([str(i) for i in nets],fontsize=7); ax.set_xlabel('held-out network'); ax.set_ylabel('final MSE change (%)')
    f.tight_layout(); f.savefig('fig/two_term_nets.pdf'); f.savefig('fig/two_term_nets.png',dpi=220)
    print('nets',len(nets),'mean %.2f'%ch.mean(),'improved %d'%(ch<0).sum())
except Exception as e: print('two-term fig skipped',e)
print('ok')
