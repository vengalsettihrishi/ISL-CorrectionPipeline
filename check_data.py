import csv, numpy as np
from pathlib import Path

# Check CSV
rows = list(csv.DictReader(open('data_iSign/iSign_v1.1.csv', encoding='utf-8')))
print(f'CSV rows:  {len(rows)}')
print(f'Columns:   {list(rows[0].keys())}')
print(f'Sample:    {rows[0]}')

# Check poses
poses = list(Path('data_iSign/poses').glob('*.npy'))
print(f'NPY files: {len(poses)}')
sample = np.load(str(poses[0]))
print(f'Shape:     {sample.shape}  dtype: {sample.dtype}')

# Check norm stats
ns = np.load('data_iSign/norm_stats.npz')
print(f'Norm keys: {list(ns.keys())}  mean: {ns["mean"].shape}')

# Vocab estimate
texts = [r['english'] for r in rows]
words = set()
for t in texts:
    words.update(t.lower().split())
print(f'Vocab est: {len(words)} unique words')
print('All checks passed - ready to train!')
