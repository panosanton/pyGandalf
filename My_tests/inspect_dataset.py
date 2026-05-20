"""
Quick inspection of a dataset .npz file.
Usage:  python My_tests/inspect_dataset.py dataset/run_000000.npz
"""
import sys
import numpy as np
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('dataset/run_000000.npz')
d    = np.load(path, allow_pickle=False)

print(f"\nFile: {path}")
print("=" * 60)
for key in d.files:
    arr = d[key]
    if arr.ndim == 0:
        print(f"  {key:20s}  scalar  = {arr.item()}")
    else:
        has_nan = np.isnan(arr.astype(float)).any() if arr.dtype.kind == 'f' else False
        has_inf = np.isinf(arr.astype(float)).any() if arr.dtype.kind == 'f' else False
        extra   = ''
        if has_nan: extra += '  *** NaN ***'
        if has_inf and key != 'spring_break_at': extra += '  *** Inf ***'
        print(f"  {key:20s}  {str(arr.shape):18s}  {str(arr.dtype):8s}"
              f"  min={arr.min():.4g}  max={arr.max():.4g}{extra}")

print()

T  = d['positions'].shape[0]
N  = d['positions'].shape[1]
E  = d['edges'].shape[0]
ns = int(d['n_structural'])
nc = E - ns

print(f"Trajectory length : {T} frames")
print(f"Vertices (N)      : {N}  (n_orig={d['n_orig'].item()}, n_phys={d['n_phys'].item()})")
print(f"Springs  (E)      : {E}  ({ns} structural + {nc} cutting)")

pos   = d['positions']
disp  = pos - pos[0]                     # displacement from initial
mag   = np.linalg.norm(disp, axis=-1)   # (T, N)
print(f"\nMax displacement  : {mag.max():.4f} m  (at frame {mag.max(1).argmax()}, "
      f"vertex {mag[mag.max(1).argmax()].argmax()})")
print(f"Mean final disp   : {mag[-1].mean():.4f} m")

bp = d['blade_progress']
print(f"\nBlade progress    : {bp[0]:.4f} → {bp[-1]:.4f}  "
      f"(first break at ~{bp[bp > bp[0]].min():.4f} if any)")

n_broken = np.isfinite(d['spring_break_at']).sum()
print(f"Cutting springs   : {nc} total, {n_broken} have a break distance recorded")
