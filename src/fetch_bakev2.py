"""Independent N=1e8 per-neuron moments (Keenan Pepper's bakev2, mini split): data/bk/marg_NNN.npz.
Used as the independent reference of the one-step audit (onestep10.py, k31use*.py, edgecheck.py)."""
import os, sys, shutil
from huggingface_hub import hf_hub_download
os.makedirs('data/bk', exist_ok=True)
for i in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2,3,4,5').split(',')]:
    p = hf_hub_download('keenanpepper/whest-p2-bakev2-mini-supp', f's00/net_{i:05d}/marg.npz', repo_type='dataset')
    shutil.copy(p, f'data/bk/marg_{i:03d}.npz'); print(i)
