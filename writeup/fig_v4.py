import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, re
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
C3=['#2a6f97','#c44536','#6a994e']

# ---- Fig: inside V29, what the (3,1) slice is worth and through which channel ----
base=np.array([2.290,2.087,2.611])
rows=[('true slice, both uses',[1.681,1.899,2.306]),
      ('true slice, layers 8–15 only',[1.830,1.831,2.445]),
      ('true non-radial part + radial on V29 Σ',[1.678,1.893,2.302]),
      ('radial part alone, LS scale',[2.292,2.192,2.855]),
      ('true slice, covariance update only',[2.674,3.514,3.471]),
      ('true slice, higher-cumulant updates only',[3.821,3.126,4.189]),
      ('no slice at all',[2.878,2.477,2.818])]
f,ax=plt.subplots(figsize=(5.6,3.2))
y=np.arange(len(rows))[::-1]; h=0.26
for k in range(3):
    vals=[100*(r[1][k]/base[k]-1) for r in rows]
    ax.barh(y+(1-k)*h,vals,height=h,color=C3[k],label=f'net {k}')
ax.axvline(0,c='k',lw=.6); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=7.5)
ax.set_xlabel('final MSE change vs V29 (%)'); ax.legend(frameon=False,fontsize=7,loc='upper right')
f.tight_layout(); f.savefig('fig/v29_anatomy.pdf'); f.savefig('fig/v29_anatomy.png',dpi=220)

# ---- Fig: skewness identity ----
def parse(fn):
    R={}
    for line in open(fn):
        m=re.match(r'z(\d+)\s+k3: corr ([\d.]+) R2 ([\d.-]+) slope ([\d.]+) \| alpha corr ([\d.]+)',line)
        if m: R[int(m.group(1))]=tuple(float(m.group(j)) for j in (2,3,4,5))
    return R
logs=['logs/skewid0b.log','logs/skewid1.log','logs/skewid2.log']
P=[parse(fn) for fn in logs]
d=np.load('data/skewid_000.npz')
f,(a1,a2)=plt.subplots(1,2,figsize=(6.4,2.7))
g=d['g_15']; gp=d['gp_15']
a1.scatter(gp,g,s=2,alpha=.5,c='#2a6f97',rasterized=True); lim=[min(g.min(),gp.min()),max(g.max(),gp.max())]
a1.plot(lim,lim,c='k',lw=.7); a1.set_xlabel('identity prediction (no fitted parameter)'); a1.set_ylabel('true skewness γᵢ (N=10⁸)')
a1.set_title('ℓ = 15, net 0',fontsize=8)
for k in range(3):
    L=sorted(P[k]); a2.plot(L,[P[k][l][1] for l in L],'o-',ms=2.5,c=C3[k],label=f'net {k}: R²')
    a2.plot(L,[P[k][l][2] for l in L],'--',lw=.8,c=C3[k])
a2.set_xlabel('layer ℓ'); a2.set_ylim(0.4,1.08); a2.text(5.5,0.45,'dashed: slope (theory 1)',fontsize=7)
a2.legend(frameon=False,fontsize=7,loc='lower right')
f.tight_layout(); f.savefig('fig/skew_identity.pdf'); f.savefig('fig/skew_identity.png',dpi=220)

# ---- Fig: row scale of the (3,1) slice vs alpha ----
U=np.load('data/k31alpha3.npz'); b=U['bins']; c=(b[:-1]+b[1:])/2
f,ax=plt.subplots(figsize=(3.9,2.6))
for k in range(3):
    for l,ls in ((14,'-'),(8,':')):
        ax.plot(c,U[f's_{k}_{l}']/U[f'glob_{k}_{l}'],ls,marker='o',ms=2.5,c=C3[k],label=f'net {k}' if l==14 else None)
ax.axhline(1,c='gray',lw=.7); ax.set_xlabel('αₐ of the tripled neuron (bin centre)'); ax.set_ylabel('row scale / global scale')
ax.text(1.2,1.16,'solid ℓ = 14\ndotted ℓ = 8',fontsize=7); ax.legend(frameon=False,fontsize=7,loc='upper center')
f.tight_layout(); f.savefig('fig/k31_ushape.pdf'); f.savefig('fig/k31_ushape.png',dpi=220)

# ---- Fig: one-step audit, three networks ----
Ls=[5,8,11,14]
nets={0:{'base':[2.811,3.520,4.294,5.080],'rad':[2.481,3.111,3.924,4.694],'true':[2.089,2.236,2.481,2.589]},
      1:{'base':[2.798,3.671,4.388,5.426],'rad':[2.524,3.132,3.877,4.749],'true':[2.174,2.270,2.384,2.682]},
      2:{'base':[2.873,3.453,4.476,4.973],'rad':[2.570,3.055,3.909,4.530],'true':[2.155,2.244,2.473,2.457]}}
lab={'base':'Mehler + first-order slices','rad':'+ Edgeworth coefs + radial K','true':'+ Edgeworth coefs + true K'}
f,ax=plt.subplots(figsize=(3.9,2.7))
for j,key in enumerate(['base','rad','true']):
    for k in range(3):
        ax.plot(Ls,nets[k][key],'o-' if k==0 else 'o--',ms=2.5,lw=1.2 if k==0 else .7,c=C3[j],label=lab[key] if k==0 else None)
ax.axhline(2.0,c='gray',ls='--',lw=.9); ax.text(5,2.05,'noise floor (independent bakes)',fontsize=7,color='gray')
ax.set_xlabel('layer ℓ (exact input state)'); ax.set_ylabel(r'next-layer var. error ($10^{-4}$, rel.)'); ax.set_ylim(1.5,5.6)
ax.legend(frameon=False,fontsize=6.5); ax.text(10.6,1.62,'solid net 0, dashed nets 1–2',fontsize=6.5)
f.tight_layout(); f.savefig('fig/onestep.pdf'); f.savefig('fig/onestep.png',dpi=220)
print('ok')
