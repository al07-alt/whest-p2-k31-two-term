import numpy as np, time
from lib import *
Wt=W(0,np.float32); n=1024; N=60000; B=10000
acc={}
rng=np.random.default_rng(1)
t=time.time()
for b in range(N//B):
    x=rng.standard_normal((B,n)).astype(np.float32)
    h=x
    for l in range(16):
        z=h@Wt[l]; r=np.sqrt(np.mean(h.astype(np.float64)**2,1))[:,None]   # rho^(1/2) of previous layer
        zn=(z/r).astype(np.float64); z=z.astype(np.float64)
        for nm,v in (('z',z),('zn',zn)):
            k=(l,nm); a=acc.setdefault(k,np.zeros((4,n)))
            a[0]+=v.sum(0); a[1]+=(v**2).sum(0); a[2]+=(v**3).sum(0); a[3]+=(v**4).sum(0)
        k=(l,'post'); a=acc.setdefault(k,np.zeros((2,n))); hr=np.maximum(z,0); a[0]+=hr.sum(0)
        k=(l,'rpost'); a=acc.setdefault(k,np.zeros((3,n))); hn=np.maximum(zn,0); a[0]+=hn.sum(0); a[1]+=(r*hn).sum(0); a[2]+=r.sum()
        h=np.maximum(z,0).astype(np.float32)
print('%.0fs'%(time.time()-t))
for l in [1,3,6,9,12,15]:
    out=[]
    for nm in ('z','zn'):
        s1,s2,s3,s4=acc[(l,nm)]/N; mu=s1; c2=s2-mu**2; c3=s3-3*mu*s2+2*mu**3; c4=s4-4*mu*s3+6*mu**2*s2-3*mu**4-3*c2**2
        out.append('%s: rms skew %.4f mean exkurt %.4f rms exkurt %.4f'%(nm,np.sqrt(np.mean((c3/c2**1.5)**2)),np.mean(c4/c2**2),np.sqrt(np.mean((c4/c2**2)**2))))
    a=acc[(l,'rpost')]; Ehn=a[0]/N; Erhn=a[1]/N; Er=a[2]/N
    cov=Erhn-Er*Ehn
    out.append('Cov(r,relu(zn)) rms %.2e (E relu ~%.2f)'%(np.sqrt(np.mean(cov**2)),np.mean(Erhn)))
    print(l,' | '.join(out))
