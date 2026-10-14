import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, re
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
base=2.4680e-08; d={}
for line in open('src/orl0.log'):
    m=re.match(r'0 oracle layers \[(\d+)\] final ([\d.e+-]+)',line)
    if m: d[int(m.group(1))]=float(m.group(2))
L=sorted(d); v=[100*(d[l]/base-1) for l in L]
f,ax=plt.subplots(figsize=(4.6,2.5)); ax.bar(L,v,color=['#2a6f97' if x<0 else '#b0b0b0' for x in v]); ax.axhline(0,c='k',lw=.6)
ax.set_xlabel('layer ℓ with the true (3,1) slice'); ax.set_ylabel('final MSE change (%)'); ax.set_xticks(L)
ax.text(1,-5.2,'layers 1–7 jointly: +0.1 %\nlayers 8–15 jointly: −20.2 %',fontsize=7.5)
f.tight_layout(); f.savefig('fig/k31_layers.pdf'); f.savefig('fig/k31_layers.png',dpi=220)
# onestep raw with floor
Ls=[5,8,11,14]
raw={'Mehler + first-order slices':[2.811e-4,3.520e-4,4.294e-4,5.080e-4],'+ Edgeworth coefs + radial K':[2.481e-4,3.111e-4,3.924e-4,4.694e-4],'+ Edgeworth coefs + true K':[2.089e-4,2.236e-4,2.481e-4,2.589e-4]}
C=['#2a6f97','#c44536','#6a994e']
f,ax=plt.subplots(figsize=(3.8,2.6))
for k,(nm,vv) in enumerate(raw.items()): ax.plot(Ls,[x*1e4 for x in vv],'o-',c=C[k],label=nm)
ax.axhline(2.0,c='gray',ls='--',lw=.9); ax.text(5,2.05,'noise floor (independent bakes)',fontsize=7,color='gray')
ax.set_xlabel('layer ℓ (exact input state)'); ax.set_ylabel(r'next-layer var. error ($10^{-4}$, rel.)'); ax.set_ylim(1.5,5.4); ax.legend(frameon=False,fontsize=7)
f.tight_layout(); f.savefig('fig/onestep.pdf'); f.savefig('fig/onestep.png',dpi=220); print('ok')
