"""openSSS scatter estimation for continuous (unbinned) LOR reconstruction.

Bridges openSSS (github.com/r-santo/openSSS — José Santo et al., EJNMMI
Physics 2025) onto petrecon's scanner model and continuous list-mode data:

  1. builds the 40x400 uniform cylinder in openSSS's format (cm units),
  2. runs the TOF-aware SSS with a single TOF bin (= non-TOF) on downsampled
     activity/attenuation maps,
  3. interpolates the coarse estimate to the full detector-pair space,
  4. maps it onto arbitrary continuous LOR endpoints by snapping each
     endpoint to its nearest crystal (only the smooth *correction term* is
     tabulated — the measured events themselves stay unbinned).

The per-event values plug into OSEM as the additive scatter term:
petrecon `osem_gpu(..., scatter=s)`, or `ratio = m/(fp + s)` in a custom
OMEGA loop. Validated on the IEC phantom (both engines: background
suppressed several times more than the hot spheres, cross-engine
correlation 0.997).

Scaling: SSS produces the scatter *shape*; `scale_to_fraction` applies a
manual global scatter fraction. openSSS's own tail-fitting
(`ScaleScattersToPrompts`) assumes binned prompts — a proper unbinned tail
fit is future work.

Requires: /home/h/Code/openSSS clone, joblib (pip). Iterate: re-estimate from the
current reconstruction every couple of OSEM iterations, like petrecon's SSS.
"""
import sys
from pathlib import Path

import numpy as np

OPENSSS_PATH = "/home/h/Code/openSSS/Python"
sys.path.insert(0, OPENSSS_PATH)

from openSSS.Sinogram import SinogramCoordinates
from openSSS.SingleScatterSimulationTOF import SingleScatterSimulationTOF
from openSSS.Interpolation import InterpolateScatters

# ---- petrecon's default scanner (single source of truth: scanner.py) -------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import (RINGS, DETS_PER_RING as DETS, INNER_RADIUS_MM, AXIAL_MM,
                     PITCH_Z_MM, SECTORS_TRANS, MODULES_TRANS, CRYSTALS_TRANS,
                     SECTORS_AXIAL, MODULES_AXIAL, CRYSTALS_AXIAL)


def build_geometry():
    """Detector positions/normals in openSSS's (rings, dets, xyz) cm format,
    plus the sinogram coordinate LUTs. Angular convention matches snap_to_crystals."""
    r_cm, pitch_cm, axial_cm = INNER_RADIUS_MM / 10, PITCH_Z_MM / 10, AXIAL_MM / 10
    theta = np.arange(DETS) * 2 * np.pi / DETS
    zr = -axial_cm / 2 + pitch_cm / 2 + np.arange(RINGS) * pitch_cm
    geom = np.zeros((RINGS, DETS, 3))
    geom[..., 0] = r_cm * np.cos(theta)[None, :]
    geom[..., 1] = r_cm * np.sin(theta)[None, :]
    geom[..., 2] = zr[:, None]
    normals = np.zeros_like(geom)
    normals[..., 0] = -np.cos(theta)[None, :]
    normals[..., 1] = -np.sin(theta)[None, :]
    det_size = np.array([2 * np.pi * r_cm / DETS, pitch_cm])
    lorc, sino_idx, _ = SinogramCoordinates(
        NrSectorsTrans=SECTORS_TRANS, NrSectorsAxial=SECTORS_AXIAL,
        NrModulesAxial=MODULES_AXIAL, NrModulesTrans=MODULES_TRANS,
        NrCrystalsTrans=CRYSTALS_TRANS, NrCrystalsAxial=CRYSTALS_AXIAL,
        MinSectorDifference=0, ID_Shidt=False)
    return geom, normals, det_size, lorc.astype(int), sino_idx


def _pool(a, f):
    n = a.shape[0] // f
    return a[:n * f, :n * f, :n * f].reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5))


def estimate_scatter(activity, mumap_per_mm, fov_mm=256.0, downsample_to_mm=8.0,
                     energy_resolution=0.108, energy_threshold_kev=435.0,
                     rings_used=8, detectors_used=50, sample_step=(2, 2, 2),
                     save_path="/tmp/opensss_work"):
    """Full-resolution activity + mu-map (1/mm) -> scatter table for every
    detector pair, shape (radial, angular, ring_pairs). ~40 s at defaults."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    geom, normals, det_size, lorc, sino_idx = build_geometry()
    # pool EACH map by its own factor: activity and mu-map may arrive at
    # different grids, and openSSS needs them at the same (coarse) size —
    # also, its tissue-classification loop is pure Python, so a map left at
    # high resolution hangs it for hours
    if float(np.nanmax(mumap_per_mm)) > 0.05:
        raise ValueError(f"mu-map max {np.nanmax(mumap_per_mm):.3g} looks like 1/cm "
                         f"units — this function expects 1/mm (water ~ 0.0096)")
    ds_act = max(1, int(round(downsample_to_mm / (fov_mm / activity.shape[0]))))
    ds_mu = max(1, int(round(downsample_to_mm / (fov_mm / mumap_per_mm.shape[0]))))
    act = _pool(activity.astype(np.float64), ds_act)
    mu = _pool(mumap_per_mm.astype(np.float64) * 10.0, ds_mu)   # 1/mm -> 1/cm
    if act.shape != mu.shape:
        raise ValueError(f"pooled map shapes differ: activity {act.shape} vs "
                         f"mu {mu.shape} — activity/mu grids must both divide "
                         f"down to {downsample_to_mm} mm voxels")
    if max(mu.shape) > 48:
        raise ValueError(f"pooled maps too large ({mu.shape}) — openSSS's "
                         f"tissue classification is pure Python; keep <= 48^3")
    half_cm = fov_mm / 20.0
    image_size = np.array([-half_cm] * 3 + [half_cm] * 3)
    table = np.load(Path(OPENSSS_PATH) / "openSSS/AttenuationTable_AttenuationTable.npy")
    sampled = SingleScatterSimulationTOF(
        act, mu, image_size, geom, lorc, normals, det_size, table,
        EnergyResolution=energy_resolution, EnergyThreshold=energy_threshold_kev,
        NrRingsUsed=rings_used, NrDetectorsUsed=detectors_used,
        SampleStep=np.asarray(sample_step),
        TOFResolution=210.0, TOFRange=2500.0, NrBins=1,   # 1 bin = non-TOF
        SavePath=str(save_path))
    return np.asarray(InterpolateScatters(
        sampled, RINGS, rings_used, DETS, detectors_used,
        lorc, sino_idx, str(save_path), False))


def snap_to_crystals(endpoints_mm):
    """Continuous endpoints (N,3) mm -> (transaxial index, ring index)."""
    tr = np.round(np.arctan2(endpoints_mm[:, 1], endpoints_mm[:, 0])
                  / (2 * np.pi / DETS)).astype(np.int64) % DETS
    ax = np.clip(np.round((endpoints_mm[:, 2] + AXIAL_MM / 2 - PITCH_Z_MM / 2)
                          / PITCH_Z_MM), 0, RINGS - 1).astype(np.int64)
    return tr, ax


def map_to_lors(scatter_full, lors_mm):
    """Look up the tabulated estimate for each continuous LOR (N,6) mm."""
    _, _, _, lorc, sino_idx = build_geometry()
    d1, r1 = snap_to_crystals(lors_mm[:, :3])
    d2, r2 = snap_to_crystals(lors_mm[:, 3:])
    return scatter_full[lorc[d1, d2, 1], lorc[d1, d2, 0],
                        sino_idx[r1, r2].astype(int)].astype(np.float32)


def scale_to_fraction(s_lor, scatter_fraction=0.25):
    """Scale per-event scatter so its total equals `scatter_fraction` of the
    total prompts (each event = 1 count). Manual stand-in for tail fitting."""
    return s_lor * (scatter_fraction * len(s_lor) / max(s_lor.sum(), 1e-12))


def tail_fit_scale(scatter_full, lors_mm, activity, fov_mm=256.0,
                   tail_frac_of_peak=0.005, det_stride=5, n_samples=48):
    """EXPERIMENTAL unbinned tail fit: absolute scale for the SSS estimate.

    An event is a *tail* if its LOR never passes through activity support
    (max sampled activity along the line < tail_frac_of_peak x the activity's
    99.99th percentile). Tails are scatter/randoms only, so

        scale = (# measured tail events) / (SSS prediction summed over all
                                            detector pairs classified as tail)

    Caution: at low count levels (e.g. the 200k-event IEC set) the warm
    background sits barely above reconstruction noise and the tail set is
    threshold-sensitive — inspect the returned tail fraction before trusting
    the scale, and prefer scale_to_fraction with a known scatter fraction
    when statistics are poor. Returns (scale_or_None, tail_event_fraction).
    """
    from scipy.ndimage import gaussian_filter
    act = gaussian_filter(activity.astype(np.float32), 2)
    thr = tail_frac_of_peak * np.percentile(act, 99.99)
    n = activity.shape[0]
    vox = fov_mm / n

    def is_tail(p1, p2):
        t = np.linspace(0.02, 0.98, n_samples)
        pts = p1[..., None, :] + t[:, None] * (p2 - p1)[..., None, :]
        ijk = np.clip(np.round(pts / vox + (n - 1) / 2).astype(int), 0, n - 1)
        return act[ijk[..., 0], ijk[..., 1], ijk[..., 2]].max(axis=-1) < thr

    support_frac = float((act > thr).mean())
    if support_frac > 0.5:
        # a blurry low-count/low-res activity (e.g. a coarse pre-pass) makes
        # "support" swallow the FOV and the tail set meaningless — refuse
        # rather than fit garbage (caller falls back to the manual fraction)
        print(f"tail fit refused: activity support covers {support_frac:.0%} "
              f"of the FOV (need a sharper activity estimate, e.g. grid >= 128)")
        return None, 0.0

    tails = is_tail(lors_mm[:, :3].astype(np.float64), lors_mm[:, 3:].astype(np.float64))
    n_tail = int(tails.sum())
    if n_tail < 50:
        return None, n_tail / len(lors_mm)

    geom, _, _, lorc, sino_idx = build_geometry()
    pos = geom * 10.0                                   # cm -> mm
    ds = np.arange(0, DETS, det_stride)
    d1g, d2g = np.meshgrid(ds, ds, indexing="ij")
    d1g, d2g = d1g.ravel(), d2g.ravel()
    keep = d1g < d2g
    d1g, d2g = d1g[keep], d2g[keep]
    ang, rad = lorc[d1g, d2g, 0], lorc[d1g, d2g, 1]
    pred = 0.0
    for r1 in range(RINGS):
        for r2 in range(RINGS):
            t = is_tail(pos[r1, d1g], pos[r2, d2g])
            if t.any():
                pred += scatter_full[rad[t], ang[t], int(sino_idx[r1, r2])].sum()
    pred *= det_stride ** 2                              # subsampling correction
    return (n_tail / max(pred, 1e-30)), n_tail / len(lors_mm)


if __name__ == "__main__":
    activity = np.load("/home/h/Code/Opengate/ragy_files/IEC/gt/reconstruction_image.npy")
    mumap = np.load("/home/h/Code/pet-recon/water_attenuation_map.npy")
    full = estimate_scatter(activity, mumap)
    print("scatter table:", full.shape, "| nonzero:", f"{(full > 0).mean():.1%}")
    lors = np.loadtxt("/home/h/Code/Opengate/ragy_files/IEC/iec_groundtruth_filtered.csv",
                      delimiter=",", dtype=np.float32)
    s = scale_to_fraction(map_to_lors(full, lors))
    print("per-LOR scatter:", s.shape, f"| mean {s.mean():.3f} counts/event")
