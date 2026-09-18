# Rerunning the endpoint-discretisation study (thesis §5.7, Fig. 5.21)

*Written 2026-09-18 as a parking note. Nothing here has been implemented;
the study in the thesis is unchanged. Read this before redoing it.*

## Why it needs redoing

The §5.7 result ("continuous endpoints beat every discretised grid, 5.06 mm vs
9.20 mm at the finest pitch") is directionally plausible but the experiment has
problems a reviewer will find. In rough order of severity:

1. **The snap filter discards most LORs, unevenly per grid.**
   `snap_lors_to_detectors` ([plotting.py:372](../src/petrecon/helpers/plotting.py#L372))
   rejects any LOR with an endpoint farther than the *transaxial* half-pitch,
   `R·sin(π/N_φ)`, from its nearest crystal centre, but tests the full 3D
   distance ([plotting.py:377](../src/petrecon/helpers/plotting.py#L377)).
   In every Fig. 5.21 grid the axial pitch is 2× the transaxial pitch, so an
   endpoint more than ~half a transaxial pitch from a ring centre in z is
   dropped. About half of endpoints pass, both must pass, so roughly a quarter
   of LORs survive, and the fraction differs by grid. The FWHM-vs-pitch trend
   is therefore confounded with a count-statistics trend. The retained
   fraction was never reported.

2. **Ring centres are misplaced.** Axial centres come from
   `linspace(-L/2, L/2, n_rings)` ([plotting.py:340](../src/petrecon/helpers/plotting.py#L340)),
   which puts crystals on the end faces and gives pitch `L/(N−1)`, not `L/N`.
   Correct form: `(k + ½)·L/N − L/2`.

3. **"Quantisation" and "loss of DOI" are conflated.** Snapping projects every
   endpoint onto the inner surface (r = 235.422 mm), removing all 42.9 mm of
   depth in the same step that quantises θ and z. The thesis claim (§1.3,
   §5.7) is about endpoint quantisation, but the measured penalty includes
   parallax, the bigger effect off-axis. The inner surface is also not where
   a DOI-less scanner places events; it uses an effective interaction depth.

4. **Reconstruction model mismatch inflates the binned FWHM.** Snapped LORs
   are reconstructed with the same thin Siddon line as continuous ones. Real
   binned reconstruction uses a detector-width response (sinogram bins / tube
   of response). Thin lines through a discrete set of crystal pairs on a 1 mm
   grid give streaky non-Gaussian PSFs: the R² rejections in Fig. 5.21, and a
   3 mm pitch "scanner" at 9–12 mm FWHM, which no one will believe (a 3–4 mm
   LYSO scanner measures ~3–4 mm).

5. **Pitch labels describe only one direction.** The quoted mm is inner
   circumference / N_φ. For the figure's grids (500×50 … 100×10) axial pitch is
   exactly double and is never stated, while the bottom panel reports *axial*
   FWHM. The text lists different, isotropic grids (1479×296, 740×148, 493×99,
   370×74 = 1/2/3/4 mm) that were not the ones plotted. Only 100×10 matches.

   | Grid | Transaxial pitch | Axial pitch |
   |---|---|---|
   | 500×50 | 2.96 mm | 5.92 mm |
   | 400×40 | 3.70 mm | 7.40 mm |
   | 300×30 | 4.93 mm | 9.87 mm |
   | 200×20 | 7.40 mm | 14.8 mm |
   | 100×10 | 14.8 mm | 29.6 mm |

6. **Inconsistent algorithm.** §5.7 used unfiltered backprojection, §5.4 used
   OSEM (4 it × 3 subsets); the §5.7 text still says "OSEM". The 5.06 mm
   "Pred continuous" is not comparable with Table 5.3's 3.3–4.7 mm.

7. **Counts are too low.** 50 kBq × 1 s per position. Fits already fail at
   z = 120 mm and the IEC background CV is 47 %. Arms were not compared at
   matched counts.

8. Only GT endpoints were binned (`for kind in ("gt",)` in
   [batch_recon_binned.py](../scripts/batch_recon_binned.py)). Fine, but say so,
   or also bin the predicted endpoints.

## Design for the rerun

Arms, all from the **same event list** at **matched counts**, all OSEM with
identical iterations / subsets / grid / post-filter:

| Arm | θ, z | Depth | Projector | Answers |
|---|---|---|---|---|
| A. Continuous GT | exact | exact | thin line | resolution floor |
| B. Continuous Pred | PRISM | PRISM | thin line | localiser penalty (A→B) |
| C. Binned, DOI kept | quantised | exact (or 2–4 layers) | thin line | pure endpoint-quantisation penalty (A→C): the thesis claim |
| D. Binned, no DOI | quantised | fixed effective depth | thin line | adds parallax (C→D): the DOI benefit |
| E. Binned, conventional | quantised | none | OMEGA sinogram OSEM | what a real binned pipeline gives (sanity anchor for D) |

Rules:

- **No distance rejection.** Snap every endpoint; retained fraction = 100 %.
- Ring centres at `(k + ½)·L/N − L/2`.
- Isotropic, clinically plausible grids: 1 / 2 / 3 / 4 mm (the text's set).
  Keep 14.8 mm only as an illustration, or drop it.
- Effective depth for arm D: mean interaction depth from the GT endpoints
  (`mean(r) − 235.422`), not the inner surface.
- Subsample every arm to the coincidence count of the poorest one. Simulate
  until (120, 120, 120) fits cleanly *before* subsampling.
- Metrics: NEMA-style FWHM/FWTM through the peak on all three axes, **plus** a
  model-free width (second moment or 50 % iso-contour) so non-Gaussian PSFs
  are reported, not rejected. Print `n` and counts per condition.

## What already exists

- `pixi run batch-binned` → [scripts/batch_recon_binned.py](../scripts/batch_recon_binned.py):
  the sweep skeleton (snap phase, then `petrecon run` per grid, resumable).
  Its grids and the snapper it calls carry problems 1, 2, 3, 5.
- [examples/omega_sinogram_runner.py](omega_sinogram_runner.py): arm E. Its
  own snapping is already correct (ring centres at half-pitch offsets, nearest
  crystal by angle, no rejection) and is validated to ~±3 mm on point sources.
  `pixi run compare` drives it interactively with adjustable crystals × rings.
- [scripts/compare_binned_fwhm.py](../scripts/compare_binned_fwhm.py): the
  FWHM bar chart of Fig. 5.21 (has the R² ≥ 0.8 gate).
- `pixi run scanner`: CAD of the scanner + the 25 source positions, for figures.
  [examples/scanner_mesh.py](scanner_mesh.py) draws any crystals × rings grid.

## What needs writing

1. A corrected snapper with `doi="keep" | "layers:N" | "fixed:<mm>"`, no
   rejection, correct ring centres. Reuse the angle/ring logic from
   `omega_sinogram_runner.py`; leave `snap_lors_to_detectors` alone or fix it
   in place and rerun old outputs.
2. A matched-count driver: 25 positions × 4 grids × arms A–E, fixed RNG seed
   for subsampling, writes one tidy CSV (position, arm, grid, counts, FWHM x/y/z,
   FWTM, model-free width, fit R²).
3. Replace the bar chart with FWHM vs pitch (one line per arm, error bars over
   positions), transaxial and axial panels, continuous arms as horizontal bands.
4. More simulated counts per position (OpenGATE side, outside this repo).

Estimated effort once data exists: 1–2 days of code, then GPU time for
~25 × 4 × 5 = 500 short reconstructions.

## Related loose ends in the thesis

- §5.4.1 and §5.1 still contain `[RESULT PENDING]` for the MLEM-vs-OSEM
  control and the subset angular-uniformity test; both are cheap to fill in
  during the same rerun.
- §5.6 radial-threshold retention table is also pending, and uses the same
  point-source data.
