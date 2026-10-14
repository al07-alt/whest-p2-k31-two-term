import numpy as np, sys, os, time
from huggingface_hub import HfFileSystem
fs=HfFileSystem(); repo="datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026"
i=int(sys.argv[1]); keys=sys.argv[2].split(',')
for k in keys:
    out=f'data/kp/{k}_{i:03d}.npy'
    if os.path.exists(out): continue
    t=time.time()
    with fs.open(f"{repo}/moments/mlp_{i:05d}.npz","rb") as f:
        z=np.load(f); a=z[k]
    np.save(out,a); print(k,a.shape,a.dtype,'%.0fs'%(time.time()-t),flush=True)
