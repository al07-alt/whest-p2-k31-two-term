"""Download the public `mini` split (100 MLPs, weights + ground-truth means) of the organisers' dataset
and convert it to data/npy/wNNN.npy + data/truth.npz (via convert.py)."""
import os
from huggingface_hub import HfApi, hf_hub_download
os.makedirs('data', exist_ok=True)
rid = 'aicrowd/arc-whestbench-public-2026'
for f in HfApi().list_repo_files(rid, repo_type='dataset'):
    if f.startswith('data/mini-') and f.endswith('.parquet'):
        p = hf_hub_download(rid, f, repo_type='dataset', local_dir='data/_hf')
        os.replace(p, 'data/' + os.path.basename(f).replace('mini-', 'mini-s'))
        print(f)
