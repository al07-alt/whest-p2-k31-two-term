import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':200})
C=['#2a6f97','#c44536','#6a994e','#8d6a9f','#bc6c25']
# Fig1: scale mixture
L=[1,3,6,9,12,15]; kz=[.0091,.0229,.0359,.0475,.0587,.0667]; kn=[-.0086,-.0086,-.0064,-.0046,.0001,-.0019]
f,ax=plt.subplots(figsize=(3.4,2.4)); ax.plot(L,kz,'o-',c=C[0],label=r'$z_{\ell}$'); ax.plot(L,kn,'s-',c=C[1],label=r'$z_{\ell}/\|h_{\ell-1}\|$')
ax.axhline(0,c='gray',lw=.6); ax.set_xlabel('layer'); ax.set_ylabel('mean excess kurtosis'); ax.legend(frameon=False); f.tight_layout(); f.savefig('fig/scale_mixture.pdf'); f.savefig('fig/scale_mixture.png',dpi=220)
# Fig2: per-age radial R2
d={6:[.243,.228,.320,.352,.497,.317],10:[.127,.270,.147,.325,.393,.338,.420,.536,.568,.402],15:[.113,.131,.190,.249,.295,.315,.485,.340,.489,.583,.531,.615,.729,.741,.620]}
f,ax=plt.subplots(figsize=(3.4,2.4))
for k,(l,v) in enumerate(d.items()): ax.plot(range(1,len(v)+1),v,'o-',ms=3,c=C[k],label=f'layer {l}')
ax.set_xlabel('source age (transports)'); ax.set_ylabel(r'$R^2$ of $\beta R(m,S)+\gamma m^{\otimes3}$'); ax.legend(frameon=False); f.tight_layout(); f.savefig('fig/radial_age.pdf'); f.savefig('fig/radial_age.png',dpi=220)
# Fig3: one-step var error, noise floor removed in quadrature (floor 2.0e-4)
fl=2.0e-4; Ls=[5,8,11,14]
raw={'base closure (Mehler + 1st-order slices)':[2.811e-4,3.520e-4,4.294e-4,5.080e-4],'+ Edgeworth coefs + radial K31':[2.481e-4,3.111e-4,3.924e-4,4.694e-4],'+ Edgeworth coefs + true K31':[2.089e-4,2.236e-4,2.481e-4,2.589e-4]}
f,ax=plt.subplots(figsize=(3.6,2.5))
for k,(nm,v) in enumerate(raw.items()): ax.plot(Ls,[np.sqrt(max(x*x-fl*fl,0))*1e4 for x in v],'o-',c=C[k],label=nm)
ax.set_xlabel(r'layer $\ell$ (state given exactly)'); ax.set_ylabel(r'next-layer var. error ($10^{-4}$)'); ax.legend(frameon=False,fontsize=7); f.tight_layout(); f.savefig('fig/onestep.pdf'); f.savefig('fig/onestep.png',dpi=220)
# Fig4: kink masks in V29
labs=['B1/B3 hubs\n|α|≥3','B1/B3 hubs\n|α|≥2.5','+ κ4 feed hubs\n|α|≥3','slice rows\n|α|≥3','slice rows\n|α|≥2.5']
vals=[3.9,5.7,24.8,12.7,93.8]
f,ax=plt.subplots(figsize=(3.6,2.4)); ax.bar(range(5),vals,color=[C[0],C[0],C[1],C[2],C[2]]); ax.set_xticks(range(5)); ax.set_xticklabels(labs,fontsize=6.5)
ax.set_ylabel('raw MSE change (%)'); ax.set_yscale('log'); f.tight_layout(); f.savefig('fig/kink.pdf'); f.savefig('fig/kink.png',dpi=220)
print('ok')
