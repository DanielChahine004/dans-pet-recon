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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scanner as SC
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
if scatter is not None:
    if not (cfg.get("sensitivity") or cfg.get("sens_npy")):
        sys.exit("petrecon runner: scatter correction requires geometric "
                 "sensitivity — with sens=ones the forward model is not at "
                 "count scale and the correction is a silent no-op")
    if not (1e-4 < scatter.mean() < 2.0):
        print(f"petrecon WARNING: mean scatter {scatter.mean():.3g} counts/event "
              f"outside the sane band (1e-4, 2) — check the scale", flush=True)
print(f"petrecon: {len(lors)} LORs, grid {grid[0]}^3, scatter "
      f"{'ON' if scatter is not None else 'off'}")

t0 = time.time()
if cfg.get("sens_npy"):
    sens = np.load(cfg["sens_npy"]).astype(np.float64)
elif cfg.get("sensitivity"):
    n_scanner = total_scanner_lors(SC.RINGS, SC.DETS_PER_RING)
    sens = backproject_gpu_streaming(
        iter_scanner_lors_chunked(SC.RINGS, SC.DETS_PER_RING, SC.INNER_RADIUS_MM,
                                  SC.AXIAL_MM,
                                  chunk_size=5_000_000, balance_directions=True),
        grid_size=grid, voxel_size=voxel, batch_size=5_000_000, total_lors=n_scanner)
    sens = gaussian_filter(sens, sigma=5.0, radius=5)
else:
    sens = np.ones(grid)
np.save(os.path.join(out, "sensitivity_image.npy"), sens)

bp = backproject_gpu(lors=lors, grid_size=grid, voxel_size=voxel, batch_size=1_000_000)

if scatter is not None:
    # Count-scale normalization: the geometric sensitivity is H^T 1 over ALL
    # scanner LORs (~128M), so at the MLEM fixed point fp << 1 count/event —
    # smaller than the scatter term, making fp + s = m infeasible and
    # collapsing the image to ~0. Rescale sensitivity to the measured-event
    # backprojection's magnitude so the equilibrium fp is ~1 count/event and
    # the additive scatter (in counts) is commensurate. Scale is arbitrary in
    # MLEM without scatter, so this only matters (and only runs) with scatter.
    m = sens > sens.max() * 1e-3
    k = float(np.median(bp[m] / sens[m]))
    sens = sens * k
    print(f"petrecon: sensitivity count-scale normalization x{k:.3g} "
          f"(scatter correction active)", flush=True)

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

# provenance manifest next to the output
import hashlib, json, subprocess as sp
def _ver(mod):
    try:
        import importlib.metadata as md
        return md.version(mod)
    except Exception:
        return "?"
try:
    sha = sp.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
                  "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    sha = "?"
manifest = {"engine": "petrecon", "config": {k: str(v) for k, v in cfg.items()},
            "core_seconds": round(core, 3),
            "lors_md5": hashlib.md5(open(cfg["lors_npy"], "rb").read()).hexdigest(),
            "versions": {m: _ver(m) for m in ("petrecon", "numba", "numpy")},
            "pet_recon_git": sha, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
(Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=1))
