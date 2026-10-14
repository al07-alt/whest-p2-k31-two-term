import sys, importlib.util, numpy as np, os
os.environ['EF_DEBUG']='1'
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
spec=importlib.util.spec_from_file_location('est','v29_experiments/estimator_v29_ef.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
E=mod.Estimator(); flops.configure(symmetry_warnings=False)
i=int(sys.argv[1]); w=np.load(f'data/npy/w{i:03d}.npy')
mlp=MLP(width=1024,depth=16,weights=[fnp.asarray(w[l]) for l in range(16)],seed=i)
with flops.BudgetContext(flop_budget=2**41) as ctx: p=np.asarray(E.predict(mlp,2**41),dtype=np.float64)
out={}
for d in mod.EF_DBG:
    for k in ('mu','var','D3','g4'): out[f"{k}_{d['li']}"]=np.asarray(d[k],dtype=np.float64)
out['pred']=p; np.savez(f'data/v29dbg_{i:03d}.npz',**out); print('saved',len(mod.EF_DBG))
