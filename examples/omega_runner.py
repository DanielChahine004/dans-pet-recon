"""OMEGA list-mode OSEM runner (subprocess entry for danvas_compare.py).

Reconstructs the same LOR endpoint data petrecon uses, with OMEGA's Siddon
projector (projector_type=1) through the custom-algorithm CuPy path — the
prebuilt `reconstructions_main` libraries need the ArrayFire native runtime,
which this machine doesn't have; the CuPy path compiles OMEGA's kernels at
runtime instead.

Usage: python omega_runner.py <config.pkl>
  config dict: lors_npy, out_npy, grid, iterations, subsets,
               sens_npy ("" = none), scatter_npy ("" = none)

With scatter, the update runs in measurement-count units: fp is converted
from OMEGA's fraction-of-LOR-length normalization to mm units, the external
sensitivity is unit-matched to OMEGA's backprojector response (one fixed-
point probe), and events whose LOR misses the image are skipped.

Prints tqdm-style progress lines ("OMEGA OSEM:  33%|") that danvas_compare's
ProgressTap parses, and a final "CORE_SECONDS=<t>" line.
"""
import pickle
import sys
import time
import types

# The pip 'arrayfire' wrapper raises RuntimeError without ArrayFire's native
# libraries, and OMEGA imports it even on the CuPy path (device-info display
# only). Stub just enough that those code paths no-op.
_af = types.ModuleType("arrayfire")
_af.get_active_backend = lambda: "cuda"
_af.set_backend = lambda *a, **k: None
_af.info_string = lambda: "[0] NVIDIA 0 MB"
_dev = types.ModuleType("arrayfire.device")
_dev.info_str = lambda: "[0] NVIDIA (Compute)"
_dev.set_device = lambda n: None
_af.device = _dev
_af.BackendType = type("BackendType", (), {"opencl": "opencl", "cuda": "cuda"})
sys.modules["arrayfire"] = _af
sys.modules["arrayfire.device"] = _dev

import numpy as np
from omegatomo.projector import proj

cfg = pickle.load(open(sys.argv[1], "rb"))
lors = np.load(cfg["lors_npy"]).astype(np.float32)
grid = int(cfg["grid"])
print(f"OMEGA: {len(lors)} LORs, grid {grid}^3, "
      f"{cfg['iterations']} iterations x {cfg['subsets']} subsets")

options = proj.projectorClass()
# Scanner: uniform cylinder matching petrecon's defaults (40 rings x 400 dets)
options.blocks_per_ring = 40
options.cryst_per_block = 10
options.linear_multip = 4
options.cryst_per_block_axial = 10
options.diameter = 2 * 235.422                # inner radius, mm
options.cr_p = np.pi * 2 * 235.422 / 400      # transaxial crystal pitch
options.cr_pz = 296.0 / 40                    # axial pitch
options.det_per_ring = 400
options.det_w_pseudo = 400
options.rings = 40
options.detectors = 16000
# Image: 256 mm cube, same as petrecon's default FOV
options.FOVa_x = 256.0
options.FOVa_y = 256.0
options.axial_fov = 256.0
options.Nx = options.Ny = options.Nz = grid
# List-mode events: 6 x N endpoint coordinates in mm
options.x = np.asfortranarray(lors.T)
options.SinM = np.ones(len(lors), dtype=np.float32)
options.Niter = int(cfg["iterations"])
options.subsets = int(cfg["subsets"])
options.subsetType = 1                        # every-nth; listmode supports 0/1/3
options.projector_type = 1                    # Siddon, same family as petrecon
options.useCUDA = True
options.useCuPy = True
options.verbose = 0

t0 = time.time()
options.addProjector()
options.initProj()
import cupy as cp

nvox = grid ** 3
d_f = cp.ones(nvox, dtype=cp.float32)

# Per-event LOR lengths: OMEGA's fp returns intersection as a fraction of the
# full LOR length; multiplying by L puts it in mm units (matching its bp).
L_np = np.linalg.norm(lors[:, 3:] - lors[:, :3], axis=1).astype(np.float32)
subs = int(cfg["subsets"])
order = [np.arange(i, len(lors), subs) for i in range(subs)]   # subsetType 1
d_L = [cp.asarray(L_np[o]) for o in order]

scatter = None
if cfg.get("scatter_npy"):
    s_np = np.load(cfg["scatter_npy"]).astype(np.float32)
    scatter = [cp.asarray(s_np[o]) for o in order]
    print("OMEGA: external scatter estimate ON")

# Sensitivity: either shared (petrecon's full-scanner geometric sensitivity,
# loaded from its output) or none (ones) — same normalisation petrecon uses.
if cfg.get("sens_npy"):
    sv = cp.asarray(np.load(cfg["sens_npy"]).astype(np.float32).ravel(order="F"))
    sens_mask = sv > float(sv.max()) * 1e-3
    # unit-match petrecon's sensitivity to OMEGA's bp response: at the MLEM
    # fixed point bp(m/fp) ~ sens, so probe once with a flat image
    _f0 = cp.full(nvox, 1.0 / 235.0, dtype=cp.float32)
    _r = cp.zeros(nvox, dtype=cp.float32)
    for _i in range(subs):
        options.subset = _i
        _fp = options.forwardProject(_f0, _i) * d_L[_i]
        _r += options.backwardProject(
            cp.where(_fp > 1e-6, 1.0 / _fp, 0.0).astype(cp.float32), _i).ravel()
    sv = sv * cp.float32(float(cp.median(_r[sens_mask] / sv[sens_mask])))
    d_sens = cp.where(sens_mask, sv, cp.float32(1.0))
    d_f = cp.full(nvox, 1.0 / 235.0, dtype=cp.float32)
    print("OMEGA: using shared sensitivity image (unit-matched)")
else:
    d_sens = cp.ones(nvox, dtype=cp.float32)
    sens_mask = None

n_total = options.Niter * subs
step = 0
for it in range(options.Niter):
    for i in range(subs):
        options.subset = i
        fp = options.forwardProject(d_f, i) * d_L[i]     # mm units
        if scatter is not None:
            denom = fp + scatter[i]
        else:
            denom = fp
        # zero fp = LOR misses the image box entirely -> skip those events
        # (fp/bp kernels clip boundaries slightly differently; 1/eps would
        # inject garbage along partially-clipped paths).
        ratio = cp.where(fp > 1e-6, 1.0 / cp.maximum(denom, 1e-8), 0.0).astype(cp.float32)
        bp = options.backwardProject(ratio, i)
        d_f = d_f / d_sens * bp.ravel()
        if sens_mask is not None:
            d_f = cp.where(sens_mask, d_f, cp.float32(0.0))
        step += 1
        print(f"OMEGA OSEM: {100 * step // n_total}%|", flush=True)
cp.cuda.runtime.deviceSynchronize()
core = time.time() - t0

img = cp.asnumpy(d_f).reshape((grid, grid, grid), order="F")
np.save(cfg["out_npy"], img)
print(f"OMEGA: max {img.max():.4g} at {np.unravel_index(np.argmax(img), img.shape)}")
print(f"CORE_SECONDS={core:.3f}", flush=True)
