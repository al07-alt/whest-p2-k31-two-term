"""Fetch dense pre-activation moments (pre_M11, pre_M21, pre_M31 [, M11]) for public nets, one npz download per net."""
import numpy as np, sys, os, time, tempfile
from huggingface_hub import hf_hub_download
repo = "keenanpepper/arc-whestbench-p2-higher-moments-2026"
nets = [int(x) for x in sys.argv[1].split(',')]
keys = sys.argv[2].split(',') if len(sys.argv) > 2 else ['pre_M11', 'pre_M21', 'pre_M31']
for i in nets:
    outs = {k: f'./data/kp/{k}_{i:03d}.npy' for k in keys}
    if all(os.path.exists(o) for o in outs.values()):
        continue
    t = time.time()
    with tempfile.TemporaryDirectory(dir='./data') as td:
        p = hf_hub_download(repo, f"moments/mlp_{i:05d}.npz", repo_type="dataset", local_dir=td)
        z = np.load(p)
        for k, o in outs.items():
            if not os.path.exists(o):
                np.save(o, z[k])
    print(i, 'done %.0fs' % (time.time() - t), flush=True)
