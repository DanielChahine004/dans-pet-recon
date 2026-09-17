"""OMEGA binned (sinogram) reconstruction runner — the conventional baseline.

Bins continuous LOR endpoints onto OMEGA's own crystal grid (via its
detectorCoordinates table, so binning and system matrix share one convention),
forms span-3 sinograms with OMEGA's native saveSinogram, and reconstructs with
the fully-supported turnkey path (reconstructions_main OSEM) — the well-trodden,
published route through OMEGA. This is the "what a conventional scanner +
reconstruction would do" counterpart to the unbinned engines.

Validated against known point sources: positions reproduce within ~±3 mm
axially and ~1 crystal pitch transaxially (= the binning resolution loss the
unbinned pipeline avoids — which is the comparison's point). Sources within
~15 mm of the transaxial FOV edge are not reliable targets.

VALIDITY DOMAIN: point-like / localized sources at research count levels are
solid (global peak lands on the source). Extended objects at ultra-low counts
(e.g. the 200k-event IEC: ~10 counts per active bin) come out artifact-
dominated — sparse-sinogram MLEM noise plus a suspected span-3 oblique-
segment z issue in the Python port. Use the unbinned engines for extended
low-count data until that is resolved; see RECON_ENGINES_NOTES.md.

The image is reconstructed over the full 296 mm scanner axial extent and then
cropped to the central 256 mm cube: this matches the unbinned legs' grid AND
removes the low-sensitivity boundary slabs where sinogram OSEM blows up.

Usage: python omega_sinogram_runner.py <config.pkl>
  config: lors_npy, out_npy, grid, iterations, subsets
"""
import pickle
import sys
import time
import types
from pathlib import Path

# stub the broken pip arrayfire wrapper (device-info display only)
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

# OMEGA's prebuilt CUDA_matrixfree_lib.so is linked against libnvrtc.so.13,
# which is not on the loader path (the env's conda CUDA stack is 12.8; the 13
# runtime comes from the nvidia-cuda-nvrtc pip wheel). Preloading it by soname
# lets the dynamic linker satisfy the NEEDED entry when omegatomo dlopens.
def _preload_nvrtc13():
    import ctypes, glob, site, sysconfig
    roots = [sysconfig.get_paths()["purelib"], *site.getsitepackages()]
    for root in roots:
        for lib in glob.glob(str(Path(root) / "nvidia" / "cu13" / "lib" / "libnvrtc.so.13")):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return lib
    print("OMEGA-binned WARNING: libnvrtc.so.13 not found in site-packages "
          "(pip package nvidia-cuda-nvrtc); OMEGA's CUDA library may fail to load",
          flush=True)
_preload_nvrtc13()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scanner as SC
from omegatomo.projector import proj
from omegatomo.projector.detcoord import detectorCoordinates
from omegatomo.util.sinogram import saveSinogram
from omegatomo.reconstruction import reconstructions_main

FOV_MM = 256.0

cfg = pickle.load(open(sys.argv[1], "rb"))
lors = np.load(cfg["lors_npy"]).astype(np.float32)
grid = int(cfg["grid"])
voxel = FOV_MM / grid
# virtual-scanner overrides: bin onto a coarser/finer crystal grid than the
# physical scanner (the unbinned legs are invariant to this by construction)
D = int(cfg.get("bin_dets") or SC.DETS_PER_RING)
R = int(cfg.get("bin_rings") or SC.RINGS)
if D % 4:
    sys.exit(f"OMEGA-binned: transaxial crystal count {D} must be divisible "
             f"by 4 — the sinogram angular interleave needs an even Nang (= D/2)")
PITCH_Z = SC.AXIAL_MM / R
nz_native = 2 * R - 1     # canonical sinogram-recon slice count (79):
# reconstructing on a finer z grid starves the slices between the 3.7 mm
# sinogram planes (zebra artifact); reconstruct natively, resample after
print(f"OMEGA-binned: {len(lors)} LORs, grid {grid}^3, virtual scanner "
      f"{D}x{R} (native Nz {nz_native}, resampled), span 1, MLEM", flush=True)

options = proj.projectorClass()
if D == SC.DETS_PER_RING and R == SC.RINGS:
    # physical scanner: keep the validated block-faceted factorization
    options.blocks_per_ring = SC.SECTORS_TRANS
    options.cryst_per_block = SC.CRYSTALS_TRANS
    options.linear_multip = SC.MODULES_AXIAL
    options.cryst_per_block_axial = SC.CRYSTALS_AXIAL
else:
    # virtual scanner: uniform polygon ring, one crystal per block
    options.blocks_per_ring = D
    options.cryst_per_block = 1
    options.linear_multip = R
    options.cryst_per_block_axial = 1
options.diameter = 2 * SC.INNER_RADIUS_MM
options.cr_p = np.pi * 2 * SC.INNER_RADIUS_MM / D
options.cr_pz = PITCH_Z
options.det_per_ring = D
options.det_w_pseudo = D
options.rings = R
options.detectors = D * R
options.FOVa_x = FOV_MM
options.FOVa_y = FOV_MM
options.axial_fov = SC.AXIAL_MM          # full scanner; image cropped after
options.Nx = options.Ny = grid
options.Nz = nz_native
# ---- sinogram definition: span 1 ----
# Span-3 michelogram mashing mishandles oblique-segment z in this port
# (a centered point source smears into a z-axis line; segment tables with
# limited ring difference crash sinogramCoordinates3D). Span 1 keeps every
# ring pair on its own plane — plane ordering and z were verified against
# getCoordinates — and with MLEM (below) reconstructs points cleanly.
options.span = 1
options.ring_difference = R - 1
options.Ndist = max(64, D // 2)
options.Nang = D // 2
options.TotSinos = R ** 2
options.NSinos = options.TotSinos

t0 = time.time()
# ---- bin: snap endpoints to OMEGA's own crystal table ----
cx, cy = detectorCoordinates(options)
cang = np.arctan2(cy, cx)
order = np.argsort(cang)
sorted_ang = cang[order]


def snap_transaxial(p):
    a = np.arctan2(p[:, 1], p[:, 0])
    i = np.searchsorted(sorted_ang, a)
    lo = np.clip(i - 1, 0, D - 1)
    hi = i % D
    d_lo = np.abs(np.mod(a - sorted_ang[lo] + np.pi, 2 * np.pi) - np.pi)
    d_hi = np.abs(np.mod(a - sorted_ang[hi] + np.pi, 2 * np.pi) - np.pi)
    return order[np.where(d_lo < d_hi, lo, hi)].astype(np.uint16)


def snap_ring(p):
    return np.clip(np.round((p[:, 2] + SC.AXIAL_MM / 2 - PITCH_Z / 2)
                            / PITCH_Z), 0, R - 1).astype(np.uint16)


d1, d2 = snap_transaxial(lors[:, :3]), snap_transaxial(lors[:, 3:])
r1, r2 = snap_ring(lors[:, :3]), snap_ring(lors[:, 3:])
Sino = saveSinogram(d1, d2, r1, r2, options.Nang, options.Ndist,
                    options.ring_difference, span=1,
                    rings=R, detPerRing=D,
                    cryst_per_block=int(options.cryst_per_block), nDistSide=1)
print(f"OMEGA-binned: sinogram formed, {int(Sino.sum())}/{len(lors)} counts "
      f"binned ({(Sino > 0).sum()} nonzero bins)", flush=True)

# save a viewable reduction for the dashboard's sinogram panel: the full 3D
# stack is huge (GBs at fine grids); the classic radial x angular view needs
# only the plane-summed sinogram and the direct (ring,ring) planes
_s3 = Sino.reshape(options.Ndist, options.Nang, options.NSinos, order="F")
np.savez_compressed(
    str(Path(cfg["out_npy"]).with_suffix("")) + "_sinogram.npz",
    summed=_s3.sum(axis=2).astype(np.float32),
    direct=_s3[:, :, np.arange(R) * (R + 1)].astype(np.uint16),
    ndist=options.Ndist, nang=options.Nang, rings=R, dets=D,
    radial_bin_mm=float(options.cr_p) / 2.0)

options.SinM = Sino.astype(np.float32)
options.OSEM = True
# At research count levels (10^5-ish events over ~44M sinogram bins) ordered
# subsets produce severe imbalance artifacts (a bright axis column). Fold the
# requested subsets into MLEM iterations instead: same number of updates,
# artifact-free. At clinical count levels normal OSEM would be fine.
options.Niter = int(cfg["iterations"]) * int(cfg["subsets"])
options.subsets = 1
options.subsetType = 8
options.projector_type = 1
options.x0 = np.ones((options.Nx, options.Ny, options.Nz), dtype=np.float32)
options.useCUDA = True
options.verbose = 1

pz, _ = reconstructions_main(options)
core = time.time() - t0
img = np.squeeze(np.asarray(pz))
if img.ndim == 4:
    img = img[..., -1]
# OMEGA reports failures on stdout ("Reconstruction failed!") but still returns
# an all-zero volume with exit code 0 — surface that as a hard error so the
# dashboard shows "failed" instead of a blank comparison
if not np.isfinite(img).all() or img.max() <= 0:
    sys.exit("OMEGA-binned: reconstruction produced an empty/non-finite image "
             "(see OMEGA's own error above)")
# resample the native 79-slice axial grid (3.746 mm over 296 mm) onto the
# comparison cube's z grid (linear interpolation); the cube covers only the
# central 256 mm, which also excludes the low-sensitivity boundary slabs
dz_src = SC.AXIAL_MM / img.shape[2]
zc = (((np.arange(grid) + 0.5) * voxel - FOV_MM / 2) + SC.AXIAL_MM / 2) / dz_src - 0.5
k0 = np.clip(np.floor(zc).astype(int), 0, img.shape[2] - 2)
w = np.clip(zc - k0, 0.0, 1.0).astype(np.float32)
img = np.ascontiguousarray(img[:, :, k0] * (1 - w) + img[:, :, k0 + 1] * w)

np.save(cfg["out_npy"], img)
print(f"OMEGA-binned: max {img.max():.4g} at "
      f"{np.unravel_index(np.argmax(img), img.shape)}")
print(f"CORE_SECONDS={core:.3f}", flush=True)

# provenance manifest
import hashlib
import json
import subprocess as sp


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
manifest = {"engine": "omega-binned", "config": {k: str(v) for k, v in cfg.items()},
            "sinogram": {"span": 1, "Ndist": int(options.Ndist), "Nang": int(options.Nang),
                         "mlem": True, "virtual_dets": D, "virtual_rings": R,
                         "ring_difference": int(options.ring_difference)},
            "core_seconds": round(core, 3),
            "lors_md5": hashlib.md5(open(cfg["lors_npy"], "rb").read()).hexdigest(),
            "versions": {m: _ver(m) for m in ("omegatomo", "numpy")},
            "pet_recon_git": sha, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
Path(cfg["out_npy"]).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
