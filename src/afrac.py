import numpy as np, glob
from lib import *
fs=sorted(glob.glob('data/feat/f*.npz'))
R=[]
for f in fs:
    F=np.load(f); a=F['m']/np.sqrt(F['v']); p=pdf(a)
    row=[]
    for t in (1.5,2.0,2.5,3.0):
        row.append(np.mean(np.abs(a[1:])>t))
    # energy of hub weights p^2 lost
    for t in (2.0,2.5,3.0):
        row.append(np.sum((p**2)[1:][np.abs(a[1:])>t])/np.sum(p[1:]**2))
    R.append(row)
R=np.array(R); print('frac |a|>1.5,2,2.5,3 ; p^2-energy lost at 2,2.5,3')
print('mean',np.round(R.mean(0),4)); print('sd  ',np.round(R.std(0),4))
F=np.load(fs[0]); a=F['m']/np.sqrt(F['v'])
print('per layer frac |a|>2.5:',np.round(np.mean(np.abs(a)>2.5,1),3))
print('per layer frac a<-2.5 (dead):',np.round(np.mean(a<-2.5,1),3),' a>2.5 (linear):',np.round(np.mean(a>2.5,1),3))
