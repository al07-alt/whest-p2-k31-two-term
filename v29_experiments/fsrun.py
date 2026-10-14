import sys, time, json, importlib.util
import numpy as np
import os; REPO=os.environ.get('REPO','.')
T_=np.load(REPO+"/data/truth.npz"); IDS=T_["ids"]; TRUTH=T_["means"]
import flopscope as flops, flopscope.numpy as fnp
from whestbench.domain import MLP
est_path=sys.argv[1]; ids=[int(x) for x in sys.argv[2].split(',')]
spec=importlib.util.spec_from_file_location('est',est_path); mod=importlib.util.module_from_spec(spec); sys.modules['est']=mod; spec.loader.exec_module(mod)
E=mod.Estimator(); flops.configure(symmetry_warnings=False)
try:
    E._build()
except Exception: pass
out={}
for i in ids:
    w=np.load(REPO+f'/data/npy/w{i:03d}.npy')
    mlp=MLP(width=1024,depth=16,weights=[fnp.asarray(w[l]) for l in range(16)],seed=i)
    t=time.time()
    with flops.BudgetContext(flop_budget=2**41) as ctx:
        p=E.predict(mlp,2**41)
        used=ctx.flops_used; res=ctx.residual_wall_time_s
    p=np.asarray(p,dtype=np.float64); tr=TRUTH[list(IDS).index(i)]
    np.save(REPO+f"/data/fspred_{i:03d}.npy",p)
    f=float(np.mean((p[-1]-tr[-1])**2)); a=float(np.mean((p-tr)**2))
    out[i]=dict(final=f,all=a,flops=int(used),frac=used/2**41,resid=res,wall=time.time()-t)
    print(i, f'final {f:.3e} all {a:.3e} flops {used:.3e} ({used/2**41*100:.2f}%) resid {res*1000:.0f}ms wall {time.time()-t:.1f}s', flush=True)
if len(sys.argv)>3: json.dump(out,open(sys.argv[3],'w'))
