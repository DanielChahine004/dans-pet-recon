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
nz_full = int(round(SC.AXIAL_MM / voxel))
print(f"OMEGA-binned: {len(lors)} LORs, grid {grid}^3 (recon Nz {nz_full}, "
      f"cropped), span 3, MLEM", flush=True)

options = proj.projectorClass()
options.blocks_per_ring = SC.SECTORS_TRANS
options.cryst_per_block = SC.CRYSTALS_TRANS
options.linear_multip = SC.MODULES_AXIAL
options.cryst_per_block_axial = SC.CRYSTALS_AXIAL
options.diameter = 2 * SC.INNER_RADIUS_MM
options.cr_p = np.pi * 2 * SC.INNER_RADIUS_MM / SC.DETS_PER_RING
options.cr_pz = SC.PITCH_Z_MM
options.det_per_ring = SC.DETS_PER_RING
options.det_w_pseudo = SC.DETS_PER_RING
options.rings = SC.RINGS
options.detectors = SC.DETS_PER_RING * SC.RINGS
options.FOVa_x = FOV_MM
options.FOVa_y = FOV_MM
options.axial_fov = SC.AXIAL_MM          # full scanner; image cropped after
options.Nx = options.Ny = grid
options.Nz = nz_full
# ---- canonical sinogram definition (span 3, gate_PET.py style) ----
options.span = 3
options.ring_difference = SC.RINGS - 1
options.Ndist = 200
options.Nang = SC.DETS_PER_RING // 2
st = np.concatenate((np.array(SC.RINGS * 2 - 1, ndmin=1),
                     np.arange(SC.RINGS * 2 - 1 - (options.span + 1),
                               max(SC.RINGS * 2 - 1 - options.ring_difference * 2,
                                   SC.RINGS - options.ring_difference),
                               -options.span * 2)))
st = np.insert(np.repeat(st[1:], 2), 0, st[0])
options.segment_table = st
options.TotSinos = int(np.sum(st))
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
    lo = np.clip(i - 1, 0, SC.DETS_PER_RING - 1)
    hi = i % SC.DETS_PER_RING
    d_lo = np.abs(np.mod(a - sorted_ang[lo] + np.pi, 2 * np.pi) - np.pi)
    d_hi = np.abs(np.mod(a - sorted_ang[hi] + np.pi, 2 * np.pi) - np.pi)
    return order[np.where(d_lo < d_hi, lo, hi)].astype(np.uint16)


def snap_ring(p):
    return np.clip(np.round((p[:, 2] + SC.AXIAL_MM / 2 - SC.PITCH_Z_MM / 2)
                            / SC.PITCH_Z_MM), 0, SC.RINGS - 1).astype(np.uint16)


d1, d2 = snap_transaxial(lors[:, :3]), snap_transaxial(lors[:, 3:])
r1, r2 = snap_ring(lors[:, :3]), snap_ring(lors[:, 3:])
Sino = saveSinogram(d1, d2, r1, r2, options.Nang, options.Ndist,
                    options.ring_difference, span=options.span,
                    rings=SC.RINGS, detPerRing=SC.DETS_PER_RING,
                    cryst_per_block=SC.CRYSTALS_TRANS, nDistSide=1)
print(f"OMEGA-binned: sinogram formed, {int(Sino.sum())}/{len(lors)} counts "
      f"binned ({(Sino > 0).sum()} nonzero bins)", flush=True)

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
# crop the axial extent to the central FOV_MM cube (also removes the
# low-sensitivity boundary slabs)
z0 = (img.shape[2] - grid) // 2
img = np.ascontiguousarray(img[:, :, z0:z0 + grid])

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
            "sinogram": {"span": 3, "Ndist": 200, "Nang": options.Nang, "mlem": True,
                         "ring_difference": int(options.ring_difference)},
            "core_seconds": round(core, 3),
            "lors_md5": hashlib.md5(open(cfg["lors_npy"], "rb").read()).hexdigest(),
            "versions": {m: _ver(m) for m in ("omegatomo", "numpy")},
            "pet_recon_git": sha, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
Path(cfg["out_npy"]).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
