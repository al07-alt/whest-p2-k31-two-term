import numpy as np, sys, os
from huggingface_hub import HfFileSystem
fs=HfFileSystem(); repo="datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026"
os.makedirs('data/kp',exist_ok=True)
for i in [int(x) for x in sys.argv[1].split(',')]:
    out=f'data/kp/marg_{i:03d}.npz'
    if os.path.exists(out): continue
    with fs.open(f"{repo}/moments/mlp_{i:05d}.npz","rb") as f:
        z=np.load(f); d={k:z[k] for k in ['pre_mean','pre_m2','pre_m3','pre_m4','mean','post_m2','gt_mean','mlp_name']}
    np.savez(out,**d); print(i,d['mlp_name'],flush=True)
