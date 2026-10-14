import numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'v29anat.py')).read().split("print('layer |")[0])
lam=np.load('data/lam_table.npy')
for l in [8,12,15]:
    m,v,k3,k4=truemom(l); var=D[f'var_{l}']; g4=D[f'g4_{l}']
    rad=lam[l]*var**2
    c=lambda x: np.corrcoef(x,k4)[0,1]
    X=np.stack([g4,var**2],1); b=np.linalg.lstsq(X,k4,rcond=None)[0]; comb_=X@b
    print(l,'corr with true k4: V29 g4 %.3f | lam*var^2 %.3f (scale %.3f) | best a*g4+b*v^2 %.3f coefs %s'%(c(g4),c(rad),np.sum(rad*k4)/np.sum(rad*rad),c(comb_),np.round(b,4)))
