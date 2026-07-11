"""petrecon reconstruction runner with external per-LOR scatter support.

Subprocess entry for danvas_compare.py. Mirrors run_reconstruction's
semantics (geometric streamed sensitivity, backprojection-normalized init)
but additionally accepts a precomputed per-event scatter estimate (e.g. from
openSSS via opensss_bridge), which petrecon's osem_gpu consumes natively.

Usage: python petrecon_runner.py <config.pkl>
  config: lors_npy, out_dir, grid, voxel, iterations, subsets,
          sensitivity (bool), scatter_npy ("" = none), sens_npy ("" = compute)

Prints tqdm progress (parsed by the dashboard) and CORE_SECONDS=<t>.
"""
import os
import pickle
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from petrecon.helpers.backprojection import backproject_gpu, backproject_gpu_streaming
from petrecon.helpers.osem import osem_gpu
from petrecon.helpers.sensitivity import iter_scanner_lors_chunked, total_scanner_lors

cfg = pickle.load(open(sys.argv[1], "rb"))
lors = np.load(cfg["lors_npy"]).astype(np.float64)
grid = (int(cfg["grid"]),) * 3
voxel = float(cfg["voxel"])
out = cfg["out_dir"]
os.makedirs(out, exist_ok=True)
scatter = np.load(cfg["scatter_npy"]).astype(np.float64) if cfg.get("scatter_npy") else None
print(f"petrecon: {len(lors)} LORs, grid {grid[0]}^3, scatter "
      f"{'ON' if scatter is not None else 'off'}")

t0 = time.time()
if cfg.get("sens_npy"):
    sens = np.load(cfg["sens_npy"]).astype(np.float64)
elif cfg.get("sensitivity"):
    n_scanner = total_scanner_lors(40, 400)
    sens = backproject_gpu_streaming(
        iter_scanner_lors_chunked(40, 400, 235.422, 296.0,
                                  chunk_size=5_000_000, balance_directions=True),
        grid_size=grid, voxel_size=voxel, batch_size=5_000_000, total_lors=n_scanner)
    sens = gaussian_filter(sens, sigma=5.0, radius=5)
else:
    sens = np.ones(grid)
np.save(os.path.join(out, "sensitivity_image.npy"), sens)

bp = backproject_gpu(lors=lors, grid_size=grid, voxel_size=voxel, batch_size=1_000_000)
init = bp.copy()
mask = sens > 0
init[mask] /= sens[mask]
init[~mask] = 0

recon = osem_gpu(lors=lors, grid_size=grid, voxel_size=voxel,
                 sensitivity_image=sens, init_image=init,
                 n_iterations=int(cfg["iterations"]), n_subsets=int(cfg["subsets"]),
                 batch_size=1_000_000, scatter=scatter)
core = time.time() - t0
np.save(os.path.join(out, "reconstruction_image.npy"), recon)
print(f"petrecon: max {recon.max():.4g} at "
      f"{np.unravel_index(np.argmax(recon), recon.shape)}")
print(f"CORE_SECONDS={core:.3f}", flush=True)
