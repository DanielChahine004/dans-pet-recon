# Reconstruction engines & scatter correction — integration notes

Working record for the dual-engine (petrecon + OMEGA) + openSSS setup built
2026-07-12. Everything here was empirically verified on this machine (WSL2,
RTX 2080 Ti, CUDA 12.9 driver, miniconda base env).

## The pieces

| File | Role |
|---|---|
| `danvas_compare.py` | Comparison dashboard (port 8001): dataset selector, both engines, timings, slices/diff/profiles, openSSS scatter pipeline |
| `petrecon_runner.py` | petrecon subprocess runner: geometric streamed sensitivity, bp-normalized init, **external per-LOR scatter** via `osem_gpu(scatter=)` |
| `omega_runner.py` | OMEGA subprocess runner: list-mode OSEM with OMEGA's Siddon projector (CuPy custom-algorithm path), shared sensitivity + external scatter |
| `opensss_bridge.py` | openSSS on our scanner: geometry builder, non-TOF scatter estimation, continuous-LOR mapping, manual + (experimental) tail-fit scaling |
| `opensss_runner.py` | Subprocess wrapper for the bridge (so the dashboard's Cancel works) |
| `setup_omega_wsl.sh` | One-time machine setup: CUDA dev headers + ArrayFire 3.10 + ldconfig |
| `omega_upstream_issues.md`, `omega_repro_listmode.py` | OMEGA bugs found + reproducers, ready to file upstream |

## OMEGA (omegatomo 2.2.0)

**Setup.** Prebuilt pip libs need ArrayFire natives + CUDA 13 nvrtc (driver
here caps at 12.9) → rebuild per environment with the supported hook:
`pip install omegatomo arrayfire` then
`python -c "from omegatomo.util.compile import compileOMEGA; compileOMEGA()"`.
Machine prerequisites once via `setup_omega_wsl.sh`. "OpenCL build failed" is
expected on WSL2 (no NVIDIA OpenCL); CUDA + CPU backends are what matter.
Do NOT pip-install from the git repo — its `setup.py` is stale and ships no
sources/libs (pip 2.2.0 wheel + compileOMEGA is the working combination).

**What works: the projector pair** (`forwardProject`/`backwardProject`, CuPy
path), validated voxel-exact against known LORs and off-center point sources.
**What's broken upstream (2.2.0 = 2.2.1): every `reconstructions_main`
list-mode route** — see `omega_upstream_issues.md`. The runner therefore
implements the textbook OSEM update around OMEGA's operators.

**Semantics gotchas (hard-won):**
- `options.x` = (6, N) Fortran array of event endpoints, **mm**; `SinM` = ones.
  Exactly 2 events crashes `addProjector` (upstream bug).
- List-mode supports `subsetType` 0/1/3 only; type 1 = every-nth event; the
  runner's per-event arrays (scatter, lengths) must be partitioned the same way.
- `forwardProject` returns intersection length as a **fraction of the full LOR
  length**; multiply by per-event L (mm) before comparing to counts or adding
  scatter. `backwardProject` is mm-weighted. Do NOT add extra length factors to
  the ratio (breaks the MLEM fixed point).
- Events whose LOR misses the image box give fp = 0 — zero their ratio; the
  fp/bp kernels clip boundaries slightly differently and 1/eps injects garbage.
- An external sensitivity image (petrecon's) is in petrecon-backprojector
  units; unit-match it once via the fixed-point probe (median of
  `bp(m/fp(flat)) / sens`) or the iterate collapses/explodes.

## MLEM/OSEM normalization lessons (both engines)

- **Measured-LOR sensitivity (`H^T 1` over the events) is only valid for
  point-like sources.** For extended objects it shares the activity's support
  and divides the structure out → speckle. Use the full-scanner geometric
  sensitivity (petrecon's streamed pass; shared to OMEGA).
- **Additive scatter only acts when the forward model is at count scale**
  (fp ≈ m = 1 per event at the fixed point). That requires the proper
  sensitivity; with `sens = ones` the fp magnitude is arbitrary and the
  scatter term is silently negligible. The dashboard forces sensitivity on
  whenever scatter is on.
- **Scatter + full-scanner sensitivity needs count-scale normalization.**
  Geometric sensitivity is H^T 1 over ~128M scanner LORs while only ~10^5
  events are measured, so the MLEM equilibrium fp is << 1 count/event —
  *smaller than the scatter term*, making fp + s = m infeasible and
  collapsing the image to ~1e-20 ("black image"). petrecon_runner rescales
  the sensitivity to the measured-event backprojection's magnitude (median
  ratio over the support) whenever scatter is active; omega_runner's
  fixed-point probe achieves the same implicitly.
- Initialization scale cancels after one MLEM update (don't bother tuning it
  without scatter); with scatter, petrecon's bp/sens init is fine.

## openSSS (r-santo/openSSS, cloned at /home/h/openSSS)

- Python-first, published (José Santo et al., EJNMMI Physics 2025). Extra dep: `joblib`.
- **Units are cm** internally (geometry, ImageSize, attenuation 1/cm). Our
  mu-maps are 1/mm → ×10. The shipped `AttenuationTable` water entry
  (0.09600/cm @511 keV) matches `water_attenuation_map.npy` (0.009597/mm) —
  tissue classification works unmodified.
- **Non-TOF = `NrBins=1`** with a TOFRange wide enough to cover any path
  difference (2500 ps used); verified degenerate behavior.
- Scanner mapping: 40 sectors × 1 module × 10 crystals transaxial,
  1 × 4 × 10 axial; `ID_Shidt=False` keeps detector ids identical to our
  `atan2`-based crystal snapping. `SinogramCoordinates` needs no CASToR files.
- Estimation cost ~40–60 s at bridge defaults (maps pooled to ~8 mm, 8 rings ×
  50 detectors coarse grid); output is (radial 401, angular 200, ringpairs 1600).
- **Only the correction term is binned.** Measured events stay continuous;
  `map_to_lors` snaps endpoints to the nearest crystal to *look up* the smooth
  tabulated estimate — same structure as petrecon's own SSS interpolation.
- **Energy parameters are placeholders** (0.108 resolution, 435 keV threshold —
  openSSS demo values). Set from the GATE simulation for quantitative use.
- **Scaling**: `scale_to_fraction` (manual, default 0.25) or `tail_fit_scale`
  (experimental): tails = events whose LOR never crosses activity support
  (line-sampled). At 200k IEC counts the warm background is barely above
  reconstruction noise, making the tail set threshold-sensitive — the fitted
  scatter fraction for `iec_groundtruth_filtered.csv` came out ~0.6%,
  consistent with that file being (nearly) trues-only.

## Validated results (2026-07-12)

- Point source (14,248 LORs, 256³, OSEM 4×3, no sens): petrecon core ~1.8 s
  (after numba cache), OMEGA core ~0.8 s; identical peak voxel; r = 0.9991.
- IEC + openSSS scatter (200k LORs, 128³, shared geometric sens): both engines
  reconstruct all hot spheres, identical peak voxel (35,63,65); smoothed
  cross-engine r ≈ 0.997; scatter correction suppresses background several
  times more than hot spheres on both (direction and rough magnitude agree).
- **petrecon's own SSS vs openSSS disagree** on identical inputs (per-event
  r = 0.23, radial-profile r = 0.52; petrecon's estimate is zero for 48% of
  events — coverage/binning question). Open investigation; openSSS is the
  externally validated reference. Note petrecon's SSS assumes a compact mu-map:
  the whole-FOV water map makes `coarse_factor=8` intractable (use ≥16).

## Data notes

- `water_attenuation_map.npy` = uniform water across the entire 256³ FOV
  (not a phantom body) — anything support-based must use *activity*, not mu.
- The IEC GATE macro includes a **warm background volume source**, so activity
  support ≈ the whole phantom body, not just spheres.
- `iec_groundtruth_filtered.csv` appears nearly trues-only (tail fit: ~0.6%
  scatter); scatter correction is expected to matter more for unfiltered or
  ML-predicted (`iec_prediction_filtered.csv`) events.

## Fragility guards (added 2026-07-12 after the grid-64 hang)

- `estimate_scatter` pools activity and mu-map **each by its own factor** to
  the target coarse voxel (they arrive at different grids); asserts the pooled
  shapes match and stay <= 48³ — openSSS's tissue classification is pure
  Python and a 128³ map wedges it for hours (the original bug: mu pooled by
  the activity's factor).
- `tail_fit_scale` refuses (returns None -> manual fallback) when activity
  support covers > 50% of the FOV — a blurry coarse pre-pass (e.g. grid 64)
  makes the tail set meaningless. Tail fitting wants grid >= 128.

## Anti-fragility layer (added 2026-07-12)

- **`tests/test_engines.py`** — the invariant suite; run it after touching any
  engine (`python -m unittest tests.test_engines -v`, ~5 s warm; the IEC
  scatter-direction test needs `RECON_SLOW_TESTS=1`, ~45 s with caches).
  Covers: off-center point-source position on both engines, cross-engine
  agreement, bridge guards (shapes/size/units/tail refusal), runner aborts
  on scatter-without-sensitivity, scanner.py <-> ReconConfig lockstep,
  manifest presence.
- **`examples/scanner.py`** — single source of truth for the geometry;
  runners and bridge import it (a unit test pins it to petrecon defaults).
- **Runtime guards**: both runners exit nonzero if scatter is requested
  without geometric sensitivity, and warn when mean scatter/event is outside
  (1e-4, 2). The bridge validates mu-map units (1/mm), matching pooled
  shapes, and a <= 48^3 pooled-size ceiling.
- **Provenance**: every runner writes a `manifest.json` next to its output —
  package versions, config, input md5, pet-recon git SHA, timestamps.
- **`requirements-recon.txt`** — pinned stack + per-env install notes.

## OMEGA binned leg (added 2026-07-12) — the conventional baseline

- `omega_sinogram_runner.py`: continuous endpoints snapped to OMEGA's own
  `detectorCoordinates` crystal table (guarantees binning and system matrix
  share one convention), span-3 sinograms via its native `saveSinogram`
  (createSinogram.so — compiled by compileOMEGA), reconstructed with the
  fully-supported turnkey `reconstructions_main` path. This is the
  binned-vs-unbinned comparison the thesis is built on.
- **Conventions that matter**: `axial_fov` must equal the scanner length
  (296 mm) — the recon runs at full axial extent and the output is cropped
  to the central 256 mm cube (also removes low-sensitivity boundary slabs).
  saveSinogram's plane ordering was verified consistent with
  `getCoordinates`' z array (probe: known ring pairs -> planes -> z).
- **Subsets fold into MLEM**: at research count levels, ordered subsets on
  ultra-sparse sinograms produce a bright axis-column artifact; the runner
  runs iterations x subsets MLEM updates instead.
- **Span 3 is broken in this port; use span 1.** Definitive evidence: a
  perfectly centered point source smears into a z-axis line through the
  whole volume (oblique-segment z mishandling — direct planes dominate
  off-center sources, which is why they validated), extended objects (IEC)
  reconstruct with wrong z structure, and limited-ring-difference segment
  tables crash sinogramCoordinates3D. Span 1 (every ring pair its own
  plane; ordering verified against getCoordinates) + MLEM reconstructs
  points AND the IEC correctly (peak matches unbinned within ~1 voxel).
- **Validity domain (span 1 + MLEM)**: point-like sources ~1-2 crystal
  pitches position error (the binning cost); extended objects (IEC 200k)
  reconstruct with the correct structure. Binning drops ~5% (points) to
  ~32% (IEC, scatter-rich) of events at the sinogram acceptance.
- Reconstruct at the native Nz = 2*rings-1 = 79 slices and resample onto
  the comparison grid — a finer recon z-grid starves alternate slices
  (zebra artifact).
- **Debugging lesson**: never use z-centroids to locate reconstructed
  structure in sinogram recons — boundary slabs drag centroids and faked a
  "z-compression" that cost hours. Use local peaks in interior crops.

## Display convention (settled 2026-07-12)

Raw linear windowing, [0, max], everywhere (3D volumes and slice heatmaps).
No smoothing or heuristics touch the data or the display — matching
publication practice (state the window; declare any post-filter; use
unfiltered reconstructions for resolution metrics). The visible speckle in
low-count OSEM reconstructions is a real property of those reconstructions.
Earlier auto-window heuristics (percentile, smoothed-max) are gone: a
display dominated by isolated hot voxels is information, and genuinely
broken runs are caught by the runner guards and the test suite, not hidden
by the renderer. Manual W/L sliders on the 3D panels remain for inspection.

## Tail-fit scaling: field convention (researched 2026-07-12)

- Standard practice (Watson SSS -> STIR -> vendors): tails are **auto-detected
  from the attenuation image** — LORs with ACF < ~1.1 (STIR default) are
  outside the object. Never manual bounding; the mu-map is the object
  definition. Scale = fit of scatter to (prompts − randoms) within the mask,
  classically per plane / per TOF bin.
- openSSS: `GenerateTailsMask` ray-traces detector pairs through the
  attenuation map (`UseAttenuation=True` default; activity-map fallback
  exists — same as our bridge's approach); `BackProjectionScatterRandFit`
  backprojects tail-masked prompts/scatter/randoms and least-squares fits a
  single global scale per TOF bin.
- For us: the all-FOV water mu-map is why attenuation tails don't exist here.
  TODO: build a phantom-shaped mu-map (IEC geometry is in the GATE macro),
  then per-event tails = ACF < 1.1 via petrecon's `compute_acf` — noiseless
  and standard. Include a randoms term in the fit for unfiltered data.

## Future work

- Iterate the openSSS estimate every N OSEM iterations (like petrecon's
  `sss_update_interval`) — currently one estimate from a 2-iteration pre-pass.
- Harden the unbinned tail fit (higher-count data; fit on a coarse count grid
  used only for the scalar).
- Investigate the petrecon-SSS vs openSSS disagreement (coverage, coarse
  detector density, interpolation).
- File the OMEGA issues upstream; revisit `reconstructions_main` when fixed.
- GATE energy window parameters into `opensss_bridge.estimate_scatter`.
