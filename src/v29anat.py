import numpy as np
from scipy.special import ndtr
from math import comb
pdf=lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)
i=0; D=np.load(f'data/v29dbg_{i:03d}.npz'); B=np.load(f'data/bk/marg_{i:03d}.npz')
T=np.load('data/truth.npz')['means'][i]
def truemom(l):
    s_=B['pre_s'][:,l]; c=B['pre_shift'][l].astype(float); m1=s_[0]
    mu=[sum(comb(p,q)*(s_[q-1] if q>0 else 1)*(-m1)**(p-q) for q in range(0,p+1)) for p in range(5)]
    return c+m1, mu[2], mu[3], mu[4]-3*mu[2]**2
def erelu(m,v,k3,k4):
    s=np.sqrt(v); a=m/s; p=pdf(a); P=ndtr(a)
    return m*P+s*p + k3/6*(p/s**0)*(-a*0+1)*0 + k3/(6*s**2)*s*p*(1*0+1)*0 + edge(m,s,a,p,P,k3,k4)
def edge(m,s,a,p,P,k3,k4):
    # E relu = m P + s p + k3/6 * phi(a)/s^? : first-order Edgeworth: E[f] += k3/6 E[f'''] + k4/24 E[f'''']
    f3=-a*p/s**2*s**0; f3=(-a*p)/(s**2)  # E[relu'''] = -a phi / s^2
    f4=(a*a-1)*p/s**3
    return k3/6*f3+k4/24*f4
print('layer | rel err mu/s  var  | k3: corr, slope(V29/true) | g4/k4true ratio')
for l in [1,4,8,12,15]:
    m,v,k3,k4=truemom(l)
    mu=D[f'mu_{l}']; var=D[f'var_{l}']; d3=D[f'D3_{l}']; g4=D[f'g4_{l}']
    sl=np.sum(d3*k3)/np.sum(k3*k3); r=np.corrcoef(d3,k3)[0,1]
    rg=np.sum(g4*k4)/np.sum(k4*k4); rr=np.corrcoef(g4,k4)[0,1]
    print(l,'mu err %.2e var rel %.2e | k3 corr %.3f slope %.3f | g4 vs k4 slope %.3f corr %.3f'%(np.sqrt(np.mean(((mu-m)/np.sqrt(v))**2)),np.sqrt(np.mean((var/v-1)**2)),r,sl,rg,rr))
# last layer: decompose final error
l=15; m,v,k3,k4=truemom(l); mu=D[f'mu_{l}']; var=D[f'var_{l}']; d3=D[f'D3_{l}']; g4=D[f'g4_{l}']
gs=np.sum(g4*k4)/np.sum(g4*g4)  # map g4 -> k4 units
base=lambda M,V,K3,K4: (M*ndtr(M/np.sqrt(V))+np.sqrt(V)*pdf(M/np.sqrt(V)))+edge(M,np.sqrt(V),M/np.sqrt(V),pdf(M/np.sqrt(V)),ndtr(M/np.sqrt(V)),K3,K4)
mse=lambda x: np.mean((x-T[15])**2)
print('V29 pred final MSE %.3e'%mse(D['pred'][15]))
print('Edgeworth(V29 moments, g4 mapped) %.3e  | max diff to V29 pred %.2e'%(mse(base(mu,var,d3,gs*g4)),np.abs(base(mu,var,d3,gs*g4)-D['pred'][15]).max()))
print('Edgeworth(true moments) %.3e'%mse(base(m,v,k3,k4)))
for nm,args in [('true mu',(m,var,d3,gs*g4)),('true var',(mu,v,d3,gs*g4)),('true k3',(mu,var,k3,gs*g4)),('true k4',(mu,var,d3,k4)),('true var+k3+k4',(mu,v,k3,k4))]:
    print('  swap %-16s -> final MSE %.3e'%(nm,mse(base(*args))))
print('--- uniform scalings in the reconstruction (compare: V29 actual g4x1.3 -> +10%, x0.7 -> +6%, d3x1.1 -> +14%)')
b0=mse(base(mu,var,d3,gs*g4))
for nm,args in [('g4 x1.3',(mu,var,d3,1.3*gs*g4)),('g4 x0.7',(mu,var,d3,0.7*gs*g4)),('d3 x1.1',(mu,var,1.1*d3,gs*g4))]:
    print('  %-8s %+.1f%%'%(nm,100*(mse(base(*args))/b0-1)))
e=T[15]-base(mu,var,d3,gs*g4); s=np.sqrt(var); a=mu/s; p=pdf(a)
dk4=(k4-gs*g4)/24*(a*a-1)*p/s**3; dmu=(m-mu)*ndtr(a)
print('corr(err, k4-contribution error) %.3f ; corr(err, mean-propagated part) %.3f'%(np.corrcoef(e,dk4)[0,1],np.corrcoef(e,dmu)[0,1]))
print('gs (g4->k4 mapping) %.3f'%gs)
