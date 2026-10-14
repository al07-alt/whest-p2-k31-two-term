import pyarrow.parquet as pq, numpy as np, glob, json, os
out='data/npy'; os.makedirs(out, exist_ok=True)
truth={}; meta={}
for fn in sorted(glob.glob('data/mini-s*.parquet')):
    f=pq.ParquetFile(fn)
    for g in range(f.metadata.num_row_groups):
        t=f.read_row_group(g)
        for r in range(t.num_rows):
            row=t.slice(r,1)
            mid=row['mlp_id'][0].as_py()
            w=row['weights'].combine_chunks()
            # extension array -> storage
            s=w.storage if hasattr(w,'storage') else w
            arr=s.flatten().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(16,1024,1024)
            np.save(f'{out}/w{mid:03d}.npy', arr)
            a=row['all_layer_means'].combine_chunks(); s=a.storage if hasattr(a,'storage') else a
            truth[mid]=s.flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(16,1024)
            fm=np.array(row['final_means'][0].as_py())
            assert np.allclose(fm, truth[mid][-1], atol=1e-6), mid
            meta[mid]=dict(name=row['mlp_name'][0].as_py(), seed=row['mlp_seed'][0].as_py(), avg_variance=row['avg_variance'][0].as_py())
        print(fn, g, len(truth), flush=True)
ids=sorted(truth)
np.savez('data/truth.npz', ids=np.array(ids), means=np.stack([truth[i] for i in ids]))
json.dump(meta, open('data/meta.json','w'))
print('done', len(ids))
