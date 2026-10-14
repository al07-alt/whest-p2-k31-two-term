import numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'radfit.py')).read().split("res={}")[0])
for i in [0,1]:
    R=np.load(f'data/rec{i}_full.npz')
    for l in [8,12,15]:
        T=Ttrue(i,l); Tc=R['T'][l].astype(float); Sc=R['S'][l].astype(float); mc=R['m'][l]; vc=np.diag(Sc); D=(T-Tc)
        F={'R1':2*mc[:,None]*Sc+mc[None,:]*vc[:,None],'m2m':np.outer(mc*mc,mc),'v_i m_j':np.outer(vc,mc),'m_i v_j':np.outer(mc,vc),'m_i S_ij':mc[:,None]*Sc,'S_ij m_j':Sc*mc[None,:],'Tc':Tc,'v_i m_i m_j/s?':np.outer(vc*mc/np.sqrt(vc),mc)}
        tot=np.sum(D**2)
        def r2(keys):
            X=np.stack([F[k].ravel() for k in keys],1); b=np.linalg.lstsq(X,D.ravel(),rcond=None)[0]; return 1-np.sum((D.ravel()-X@b)**2)/tot,b
        out=[]
        for keys in [['m2m'],['v_i m_j'],['R1','m2m'],['v_i m_j','m2m'],['m_i S_ij','S_ij m_j','v_i m_j','m_i v_j','m2m'],['m_i S_ij','S_ij m_j','v_i m_j','m_i v_j','m2m','Tc']]:
            r,b=r2(keys); out.append('%s:%.3f'%('+'.join(keys),r))
        print(i,l,' | '.join(out),flush=True)
