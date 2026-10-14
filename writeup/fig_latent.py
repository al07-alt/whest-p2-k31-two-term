import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
C=['#2a6f97','#c44536','#6a994e','#8d6a9f']
d=np.load('data/latent_15.npz'); z=np.load('data/kp/marg_000.npz')
v=z['pre_m2'][15]-z['pre_mean'][15]**2
f,ax=plt.subplots(1,2,figsize=(7.2,2.8))
x=d['pred3']/v**1.5; y=d['k3']/v**1.5
ax[0].scatter(x,y,s=3,alpha=.4,c=C[0]); ax[0].set_xlabel('latent prediction of skewness (X = energy)',fontsize=8); ax[0].set_ylabel('true skewness γᵢ (N=10⁸)',fontsize=8)
ax[0].set_title('layer 15, net 0: corr %.3f'%np.corrcoef(x,y)[0,1],fontsize=8)
L=[4,8,12,15]
sk={'E(ℓ−1)':[[.907,.926,.942,.946],[.882,.921,.933,.937],[.899,.924,.939,.935]],
    'mean act.':[[.916,.930,.951,.953],[.902,.928,.938,.944],[.909,.930,.946,.941]]}
ek=[[.212,.170,.388,.439],[.237,.160,.240,.311],[.142,.231,.280,.281]]
null=[.207,.359,.418,.482]
for k in range(3):
    ax[1].plot(L,sk['E(ℓ−1)'][k],'o-',c=C[0],ms=3,label='skewness, X = energy' if k==0 else None)
    ax[1].plot(L,sk['mean act.'][k],'^:',c=C[3],ms=3,label='skewness, X = mean activation' if k==0 else None)
    ax[1].plot(L,ek[k],'s--',c=C[1],ms=3,label='excess kurtosis, X = energy' if k==0 else None)
ax[1].plot(L,null,'x-',c='gray',label='skewness, X = random projection (null)')
ax[1].set_xlabel('layer'); ax[1].set_ylabel('per-neuron correlation'); ax[1].set_ylim(0,1); ax[1].legend(frameon=False,fontsize=6.5,loc='lower right'); ax[1].set_title('3 networks',fontsize=8)
f.tight_layout(); f.savefig('fig/latent.pdf'); f.savefig('fig/latent.png',dpi=220); print('ok')
