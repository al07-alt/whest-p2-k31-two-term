# Report §3: for z_i = g + a(g^2-1), z_j = rho g + sqrt(1-rho^2) e, K_iiij = kappa3_i T_iij / v_i at leading order (b2 = 1);
# if z_j is also perturbed (coefficient c of the same sign), the ratio exceeds 1. Gauss-Hermite quadrature, 60x60 nodes.
import numpy as np
x,w=np.polynomial.hermite_e.hermegauss(60); w=w/w.sum()
G,E=np.meshgrid(x,x,indexing='ij'); Wt=np.outer(w,w); Ex=lambda f:(Wt*f).sum()
for a,c,rho in [(0.01,0,0.2),(0.01,0,0.5),(0.01,0,0.8),(0.02,0.02,0.5),(0.03,0.01,0.3),(0.02,0.04,0.8),(0.02,-0.02,0.5)]:
    g=G; h=rho*g+np.sqrt(1-rho*rho)*E; zi=g+a*(g*g-1); zj=h+c*(h*h-1)
    di=zi-Ex(zi); dj=zj-Ex(zj); v=Ex(di*di); k3=Ex(di**3); T=Ex(di*di*dj); S=Ex(di*dj); K=Ex(di**3*dj)-3*v*S
    print('a=%.2f c=%+.2f rho=%.1f  K/(kappa3*T/v) = %.4f'%(a,c,rho,K/(k3*T/v)))
