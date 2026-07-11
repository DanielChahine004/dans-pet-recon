# OMEGA (omegatomo 2.2.0, PyPI) — Python list-mode `reconstructions_main` issues

Found 2026-07-12 while validating OMEGA against petrecon on custom-coordinate
list-mode PET data (continuous LOR endpoints from GATE, cylindrical scanner:
40 rings × 400 detectors, inner radius 235.422 mm). Machine: WSL2, RTX 2080 Ti,
CUDA 12.9 driver, ArrayFire 3.10.0, libs rebuilt with `compileOMEGA()`.
All three cases below use the same data and geometry that reconstruct
**correctly** through the custom-algorithm CuPy path (forwardProject /
backwardProject + hand-written OSEM), so the projector and geometry are fine —
the failures are specific to `reconstructions_main`'s list-mode handling.
Code identical in v2.2.1 tag (checked `recomain.py` lines 255/564).

Reproducer: `python omega_repro_listmode.py <case>` with case = 1, 2, or 3.
The test data is a point source at (96, 96, 96) mm, so the correct peak voxel
on a 128³ / 2 mm grid is ≈ (112, 112, 110). Ground truth checked by
least-squares LOR convergence and by two independent reconstructions
(petrecon and the OMEGA CuPy path), both of which put the peak there.

## Case 1 — coordinate list-mode + `compute_sensitivity_image=True`
`options.x` = 6×N event endpoints, `SinM` = ones. Runs, but the transaxial
information is lost: the reconstruction peaks at x = y = image centre while
the axial position is roughly right (argmax ≈ (64, 63, 118)). Looks like the
`getCoordinates` overwrite of `options.z` (recomain.py:564-570) leaves the
C++ side pairing event transaxial coordinates with the wrong buffers.

## Case 2 — coordinate list-mode + `compute_sensitivity_image=False`
Same setup, sensitivity off: the output volume is identically zero.

## Case 3 — index-based list-mode + `compute_sensitivity_image=True`
`useIndexBasedReconstruction=True`, `trIndex`/`axIndex` (2, N) uint16 from
snapping endpoints to the crystal grid (1 same-crystal event filtered): output
is near-flat noise (finite max ≈ 0.18 from a ones init) with 315 NaN voxels.

## Also worth an issue
- `except ModuleNotFoundError` around `import arrayfire` (checkCUDA.py,
  proj.py OMEGAErrorCheck, init.py) does not catch the RuntimeError the pip
  arrayfire wrapper raises when the native libraries are missing, so the
  friendly fallback never triggers.
- `proj.py` `addProjector` crashes (`'numpy.ndarray' object is not callable`,
  line ~891) when a coordinate list-mode `options.x` has exactly 2 events —
  the (6, 2) array is misrouted to the separate-x/y/z branch.
- compile.py Linux CUDA fallback path `-L/usr/lib/x86_64-linux` is missing the
  `-gnu` suffix.
- The `source/Python/setup.py` in the repo is stale (name/version 2.0.0, no
  package_data) — a git install lacks `cpp/`, `opencl/`, `libs/` and
  `util/compile.py`, so `pip install git+...` yields a broken package.
