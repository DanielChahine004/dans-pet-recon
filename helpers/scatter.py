"""
Single Scatter Simulation (SSS) for PET Reconstruction
========================================================
Estimates the scatter contribution for each detector pair using
an analytical model based on Klein-Nishina cross sections.

For a single-scatter event (gamma_1: E->A unscattered, gamma_2: E->S->B scattered):

  scatter(A,B) = sum_S [ n_e(S) * KN(theta) * exp(-mu_AS) * Lambda_AS
                         * exp(-mu_SB) * geom * dV ]

Computation strategy:
  1. Coarse scatter point grid (every Nth voxel inside object)
  2. Coarse detector set (subset of scanner ring positions)
  3. GPU precompute: ray-trace from each scatter point to each detector
  4. CPU (Numba JIT): pair-wise accumulation into scatter sinogram
  5. Interpolation: coarse sinogram -> per-LOR scatter estimates
"""

import numpy as np
import math
from numba import cuda, njit
import numba
from tqdm import tqdm
from helpers.siddon import siddon_cuda


# =============================================================================
# Physical constants
# =============================================================================
E_GAMMA = 511.0              # keV, annihilation photon energy
MU_WATER = 0.009597          # mm^-1, water linear attenuation at 511 keV


# =============================================================================
# Scatter point and detector setup
# =============================================================================

def identify_scatter_points(mu_map, grid_size, voxel_size, coarse_factor=8, mu_threshold=0.001):
    """
    Identify scatter point positions from the mu-map on a coarse grid.

    Parameters:
        mu_map: 3D array of mu values (mm^-1)
        grid_size: (nx, ny, nz)
        voxel_size: float, mm
        coarse_factor: int, downsampling factor
        mu_threshold: float, minimum mu to be inside object (mm^-1)

    Returns:
        scatter_points: (K, 3) array of scatter point positions in mm
        scatter_ne: (K,) electron density at each point (proportional to mu)
        delta_V: float, volume element per scatter point (mm^3)
    """
    nx, ny, nz = grid_size
    half_x = nx * voxel_size / 2.0
    half_y = ny * voxel_size / 2.0
    half_z = nz * voxel_size / 2.0

    cx = np.arange(coarse_factor // 2, nx, coarse_factor)
    cy = np.arange(coarse_factor // 2, ny, coarse_factor)
    cz = np.arange(coarse_factor // 2, nz, coarse_factor)

    points = []
    ne_values = []

    for ix in cx:
        for iy in cy:
            for iz in cz:
                if mu_map[ix, iy, iz] > mu_threshold:
                    x = -half_x + (ix + 0.5) * voxel_size
                    y = -half_y + (iy + 0.5) * voxel_size
                    z = -half_z + (iz + 0.5) * voxel_size
                    points.append([x, y, z])
                    ne_values.append(mu_map[ix, iy, iz] / MU_WATER)  # normalized: 1.0 for water

    scatter_points = np.array(points, dtype=np.float32)
    scatter_ne = np.array(ne_values, dtype=np.float32)
    delta_V = (coarse_factor * voxel_size) ** 3

    print(f"SSS: {len(scatter_points)} scatter points "
          f"(coarse grid {len(cx)}x{len(cy)}x{len(cz)}, factor={coarse_factor})")

    return scatter_points, scatter_ne, delta_V


def define_coarse_detectors(n_angular=32, n_axial=15, radius=117.711, axial_length=296.0):
    """
    Define coarse detector positions on the scanner ring for SSS.

    Parameters:
        n_angular: int, angular detector positions per ring
        n_axial: int, axial ring positions
        radius: float, scanner inner radius (mm)
        axial_length: float, scanner axial length (mm)

    Returns:
        detectors: (n_det, 3) array of detector positions in mm
    """
    angles = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    z_positions = np.linspace(-axial_length / 2, axial_length / 2, n_axial)

    n_det = n_angular * n_axial
    detectors = np.zeros((n_det, 3), dtype=np.float32)

    idx = 0
    for z in z_positions:
        for angle in angles:
            detectors[idx, 0] = radius * np.cos(angle)
            detectors[idx, 1] = radius * np.sin(angle)
            detectors[idx, 2] = z
            idx += 1

    print(f"SSS: {n_det} coarse detectors ({n_angular} angular x {n_axial} axial)")
    return detectors


# =============================================================================
# GPU kernel: precompute ray integrals from scatter point to all detectors
# =============================================================================

@cuda.jit
def sss_precompute_kernel(
    scatter_point,        # (3,) single scatter point [x, y, z]
    detectors,            # (n_det, 3) coarse detector positions
    mu_map_flat,          # flattened mu map
    activity_flat,        # flattened activity image
    mu_integrals_out,     # (n_det,) output: mu line integral from S to each detector
    weighted_activity_out,  # (n_det,) output: attenuated activity integral from det toward S
    nx, ny, nz, voxel_size, x0, y0, z0
):
    """
    For a single scatter point S, compute line integrals to all detectors.
    One CUDA thread per detector.

    For each detector D:
      - Traces ray from D toward S (so cumulative attenuation goes from detector inward)
      - mu_integral: total line integral of mu from D to S
      - weighted_activity: sum of activity(v) * exp(-cumulative_mu_from_D_to_v) * dl(v)
    """
    det_idx = cuda.grid(1)
    if det_idx >= detectors.shape[0]:
        return

    voxels = cuda.local.array(1024, numba.int32)
    lens = cuda.local.array(1024, numba.float64)

    sx = scatter_point[0]
    sy = scatter_point[1]
    sz = scatter_point[2]

    dx = detectors[det_idx, 0]
    dy = detectors[det_idx, 1]
    dz = detectors[det_idx, 2]

    # Trace ray from detector D to scatter point S
    count = siddon_cuda(dx, dy, dz, sx, sy, sz,
                        (nx, ny, nz), voxel_size, (x0, y0, z0),
                        voxels, lens)

    mu_cumulative = 0.0
    weighted_act = 0.0

    for i in range(count):
        v_idx = voxels[i]
        dl = lens[i]
        mu_v = mu_map_flat[v_idx]
        act_v = activity_flat[v_idx]

        # Attenuation at midpoint of this segment
        mu_at_mid = mu_cumulative + mu_v * dl * 0.5

        # Activity contribution weighted by cumulative attenuation from detector
        weighted_act += act_v * math.exp(-mu_at_mid) * dl

        mu_cumulative += mu_v * dl

    mu_integrals_out[det_idx] = mu_cumulative
    weighted_activity_out[det_idx] = weighted_act


# =============================================================================
# Pair accumulation (Numba JIT on CPU)
# =============================================================================

@njit
def _find_bin(value, edges):
    """Find bin index for a value given bin edges. Returns -1 if out of range."""
    n = len(edges) - 1
    if value < edges[0] or value >= edges[n]:
        return -1
    # Linear scan (edges are sorted, typically small arrays)
    for i in range(n):
        if value < edges[i + 1]:
            return i
    return -1


@njit
def _accumulate_scatter_for_point(
    sp_x, sp_y, sp_z, ne, delta_V,
    mu_ints, wt_act,
    detectors, scatter_sino,
    s_edges, phi_edges, z_avg_edges, z_diff_edges,
):
    """
    Accumulate scatter contributions from one scatter point into the sinogram.
    Numba JIT-compiled for CPU performance.

    For each detector pair (A, B):
      - Case 1: gamma from emission -> A (unscattered), scattered at S -> B
        scatter_AB = wt_act[A] * ne * KN(theta) * exp(-mu_ints[B]) * geom * dV
      - Case 2: gamma from emission -> B (unscattered), scattered at S -> A
        scatter_BA = wt_act[B] * ne * KN(theta) * exp(-mu_ints[A]) * geom * dV
    """
    n_det = detectors.shape[0]

    for a_idx in range(n_det):
        ax = detectors[a_idx, 0]
        ay = detectors[a_idx, 1]
        az = detectors[a_idx, 2]

        # Vector from S to A
        sa_x = ax - sp_x
        sa_y = ay - sp_y
        sa_z = az - sp_z
        dist_sa = math.sqrt(sa_x * sa_x + sa_y * sa_y + sa_z * sa_z)
        if dist_sa < 1e-6:
            continue

        # Skip if no activity along this direction
        wt_a = wt_act[a_idx]
        mu_a = mu_ints[a_idx]
        if wt_a < 1e-30:
            continue

        for b_idx in range(a_idx + 1, n_det):
            bx = detectors[b_idx, 0]
            by = detectors[b_idx, 1]
            bz = detectors[b_idx, 2]

            # Vector from S to B
            sb_x = bx - sp_x
            sb_y = by - sp_y
            sb_z = bz - sp_z
            dist_sb = math.sqrt(sb_x * sb_x + sb_y * sb_y + sb_z * sb_z)
            if dist_sb < 1e-6:
                continue

            # Scattering angle
            dot = sa_x * sb_x + sa_y * sb_y + sa_z * sb_z
            cos_theta = dot / (dist_sa * dist_sb)
            if cos_theta > 1.0:
                cos_theta = 1.0
            if cos_theta < -1.0:
                cos_theta = -1.0

            # Scattered photon energy: E' = 511 / (2 + cos(∠ASB))
            # cos_theta here is cos(∠ASB) = -cos(θ_compton), so the
            # standard Compton formula E'=511/(2-cos θ) becomes E'=511/(2+cos ∠ASB)
            E_scattered = 511.0 / (2.0 + cos_theta)
            # Energy window check (typical PET: 350-650 keV)
            if E_scattered < 350.0 or E_scattered > 650.0:
                continue

            # Klein-Nishina differential cross section (mm^2/sr)
            ratio = E_scattered / 511.0
            sin2 = 1.0 - cos_theta * cos_theta
            kn = 0.5 * ratio * ratio * (ratio + 1.0 / ratio - sin2)

            # Geometric factor: inverse-square falloff
            geom = 1.0 / (dist_sa * dist_sa * dist_sb * dist_sb)

            wt_b = wt_act[b_idx]
            mu_b = mu_ints[b_idx]

            # Case 1: A receives unscattered, B receives scattered
            scatter_AB = wt_a * ne * kn * math.exp(-mu_b) * geom * delta_V

            # Case 2: B receives unscattered, A receives scattered
            scatter_BA = wt_b * ne * kn * math.exp(-mu_a) * geom * delta_V

            total = scatter_AB + scatter_BA
            if total <= 0:
                continue

            # Compute sinogram coordinates for this detector pair
            dx_ab = bx - ax
            dy_ab = by - ay
            lor_len_xy = math.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)
            if lor_len_xy < 1e-6:
                continue

            s_val = (ax * by - bx * ay) / lor_len_xy

            phi_val = math.atan2(dy_ab, dx_ab)
            if phi_val < 0:
                phi_val += math.pi
                s_val = -s_val
            if phi_val >= math.pi:
                phi_val -= math.pi
                s_val = -s_val

            z_avg = (az + bz) / 2.0
            z_diff = bz - az

            # Bin into sinogram
            s_bin = _find_bin(s_val, s_edges)
            phi_bin = _find_bin(phi_val, phi_edges)
            z_avg_bin = _find_bin(z_avg, z_avg_edges)
            z_diff_bin = _find_bin(z_diff, z_diff_edges)

            if s_bin >= 0 and phi_bin >= 0 and z_avg_bin >= 0 and z_diff_bin >= 0:
                scatter_sino[s_bin, phi_bin, z_avg_bin, z_diff_bin] += total


# =============================================================================
# LOR binning for tail fitting
# =============================================================================

def _bin_lors_to_sinogram(lors, s_edges, phi_edges, z_avg_edges, z_diff_edges):
    """Bin measured LORs into coarse sinogram grid for tail-fitting normalization."""
    x1, y1, z1 = lors[:, 0], lors[:, 1], lors[:, 2]
    x2, y2, z2 = lors[:, 3], lors[:, 4], lors[:, 5]

    dx = x2 - x1
    dy = y2 - y1
    lor_length_xy = np.sqrt(dx**2 + dy**2)
    lor_length_xy = np.maximum(lor_length_xy, 1e-10)

    s = (x1 * y2 - x2 * y1) / lor_length_xy
    phi = np.arctan2(dy, dx)

    flip = phi < 0
    phi[flip] += np.pi
    s[flip] = -s[flip]
    wrap = phi >= np.pi
    phi[wrap] -= np.pi
    s[wrap] = -s[wrap]

    z_avg = (z1 + z2) / 2.0
    z_diff = z2 - z1

    n_s = len(s_edges) - 1
    n_phi = len(phi_edges) - 1
    n_z_avg = len(z_avg_edges) - 1
    n_z_diff = len(z_diff_edges) - 1

    sino = np.zeros((n_s, n_phi, n_z_avg, n_z_diff), dtype=np.float64)

    s_bin = np.digitize(s, s_edges) - 1
    phi_bin = np.digitize(phi, phi_edges) - 1
    z_avg_bin = np.digitize(z_avg, z_avg_edges) - 1
    z_diff_bin = np.digitize(z_diff, z_diff_edges) - 1

    valid = ((s_bin >= 0) & (s_bin < n_s) &
             (phi_bin >= 0) & (phi_bin < n_phi) &
             (z_avg_bin >= 0) & (z_avg_bin < n_z_avg) &
             (z_diff_bin >= 0) & (z_diff_bin < n_z_diff))

    valid_idx = np.where(valid)[0]
    flat_idx = np.ravel_multi_index(
        (s_bin[valid_idx], phi_bin[valid_idx], z_avg_bin[valid_idx], z_diff_bin[valid_idx]),
        (n_s, n_phi, n_z_avg, n_z_diff)
    )
    np.add.at(sino.ravel(), flat_idx, 1.0)

    return sino


# =============================================================================
# Main SSS computation
# =============================================================================

def compute_scatter_sinogram(
    mu_map, activity_image, grid_size, voxel_size,
    scatter_points, scatter_ne, delta_V,
    detectors, n_angular, n_axial,
    radius=117.711, axial_length=296.0,
    measured_lors=None, object_radius=None,
):
    """
    Compute scatter sinogram using Single Scatter Simulation.

    For each scatter point S:
      1. GPU: precompute mu line integrals and attenuated activity to all detectors
      2. CPU (JIT): accumulate scatter for all detector pairs into sinogram

    Parameters:
        mu_map: 3D array of mu values (mm^-1)
        activity_image: 3D array of current activity estimate
        grid_size: (nx, ny, nz)
        voxel_size: float, mm
        scatter_points: (K, 3) scatter point positions
        scatter_ne: (K,) electron density at each scatter point
        delta_V: float, volume element (mm^3)
        detectors: (n_det, 3) coarse detector positions
        n_angular: int, angular detectors per ring
        n_axial: int, axial rings
        radius: float, scanner inner radius (mm)
        axial_length: float, scanner axial length (mm)
        measured_lors: (N, 6) measured LOR endpoints for tail fitting (None = no fitting)
        object_radius: float, transaxial object radius in mm for tail fitting.
                       None = auto-detect from mu-map.

    Returns:
        scatter_sino: 4D array (n_s, n_phi, n_z_avg, n_z_diff)
        sinogram_params: dict with bin edges for interpolation
    """
    nx, ny, nz = grid_size
    x0 = -nx * voxel_size / 2.0
    y0 = -ny * voxel_size / 2.0
    z0 = -nz * voxel_size / 2.0

    # Auto-detect object radius from mu-map for tail fitting
    if object_radius is None and measured_lors is not None:
        half_x = nx * voxel_size / 2.0
        half_y = ny * voxel_size / 2.0
        iy_grid, ix_grid = np.meshgrid(np.arange(ny), np.arange(nx))
        x_coords = -half_x + (ix_grid + 0.5) * voxel_size
        y_coords = -half_y + (iy_grid + 0.5) * voxel_size
        r_coords = np.sqrt(x_coords**2 + y_coords**2)
        # Max transaxial radius of any mu > threshold voxel (across all slices)
        mu_proj = mu_map.max(axis=2)  # project along z
        object_radius = r_coords[mu_proj > 0.001].max() + voxel_size  # +1 voxel margin
        print(f"SSS: Auto-detected object radius = {object_radius:.1f} mm")

    n_det = detectors.shape[0]
    n_scatter = len(scatter_points)

    mu_flat = mu_map.flatten().astype(np.float32)
    act_flat = activity_image.flatten().astype(np.float32)
    d_mu = cuda.to_device(mu_flat)
    d_act = cuda.to_device(act_flat)
    d_detectors = cuda.to_device(detectors)

    # Define scatter sinogram bins
    n_s = 32
    n_phi = n_angular // 2  # phi in [0, pi)
    n_z_avg = n_axial
    n_z_diff = 10

    s_max = radius
    half_axial = axial_length / 2.0

    s_edges = np.linspace(-s_max, s_max, n_s + 1)
    phi_edges = np.linspace(0, np.pi, n_phi + 1)
    z_avg_edges = np.linspace(-half_axial, half_axial, n_z_avg + 1)
    z_diff_edges = np.linspace(-axial_length, axial_length, n_z_diff + 1)

    scatter_sino = np.zeros((n_s, n_phi, n_z_avg, n_z_diff), dtype=np.float64)

    threads = 256
    blocks_det = (n_det + threads - 1) // threads

    print(f"SSS: Computing scatter for {n_scatter} scatter points x {n_det} detectors...")

    for sp_idx in tqdm(range(n_scatter), desc="SSS scatter points"):
        sp = scatter_points[sp_idx]
        ne = float(scatter_ne[sp_idx])

        # GPU: compute line integrals from this scatter point to all detectors
        d_sp = cuda.to_device(sp)
        d_mu_ints = cuda.device_array(n_det, dtype=np.float64)
        d_wt_act = cuda.device_array(n_det, dtype=np.float64)

        sss_precompute_kernel[blocks_det, threads](
            d_sp, d_detectors, d_mu, d_act,
            d_mu_ints, d_wt_act,
            nx, ny, nz, voxel_size, x0, y0, z0
        )

        mu_ints = d_mu_ints.copy_to_host()
        wt_act = d_wt_act.copy_to_host()

        # CPU (JIT): accumulate scatter for all detector pairs
        _accumulate_scatter_for_point(
            sp[0], sp[1], sp[2], ne, delta_V,
            mu_ints, wt_act,
            detectors, scatter_sino,
            s_edges, phi_edges, z_avg_edges, z_diff_edges,
        )

    # Smooth the scatter sinogram (scatter is inherently smooth)
    from scipy.ndimage import gaussian_filter
    for i_z_diff in range(n_z_diff):
        scatter_sino[:, :, :, i_z_diff] = gaussian_filter(
            scatter_sino[:, :, :, i_z_diff], sigma=1.0
        )

    total_scatter_raw = scatter_sino.sum()
    print(f"SSS: Raw scatter sinogram total = {total_scatter_raw:.6e}")

    # Keep a copy before tail fitting
    scatter_sino_raw = scatter_sino.copy()

    # Occupancy sinogram (counts of measured LORs per bin) — needed both for
    # tail fitting and for converting the per-bin scatter total into a per-LOR
    # estimate in scatter_sinogram_to_lor().
    occupancy_sino = None
    if measured_lors is not None:
        occupancy_sino = _bin_lors_to_sinogram(
            measured_lors, s_edges, phi_edges, z_avg_edges, z_diff_edges
        )

    # Tail-fitting normalization: scale scatter to match measured data outside object
    if occupancy_sino is not None and total_scatter_raw > 0:
        measured_sino = occupancy_sino
        s_centers = (s_edges[:-1] + s_edges[1:]) / 2.0
        tail_mask = np.abs(s_centers) > object_radius

        scatter_tail = scatter_sino[tail_mask, :, :, :].sum()
        measured_tail = measured_sino[tail_mask, :, :, :].sum()

        print(f"SSS tail fitting: measured_tail={measured_tail:.0f}, "
              f"scatter_tail_raw={scatter_tail:.6e}")

        if scatter_tail > 1e-30 and measured_tail > 10:
            scale = measured_tail / scatter_tail
            scatter_sino *= scale
            total_scatter = scatter_sino.sum()
            measured_total = measured_sino.sum()
            print(f"SSS: Tail-fit scale factor = {scale:.6e}")
            print(f"SSS: Estimated scatter fraction = {total_scatter / measured_total:.1%}")
        else:
            total_scatter = total_scatter_raw
            print("SSS: Tail fitting failed (insufficient tail counts). Using raw values.")
    else:
        total_scatter = total_scatter_raw

    print(f"SSS: Total scatter counts (final) = {total_scatter:.6e}")
    print(f"SSS: Scatter sinogram shape = {scatter_sino.shape}")

    sinogram_params = {
        's_edges': s_edges,
        'phi_edges': phi_edges,
        'z_avg_edges': z_avg_edges,
        'z_diff_edges': z_diff_edges,
        'scatter_sino_raw': scatter_sino_raw,
        'occupancy_sino': occupancy_sino,
    }

    return scatter_sino, sinogram_params


# =============================================================================
# Sinogram-to-LOR interpolation
# =============================================================================

def scatter_sinogram_to_lor(scatter_sino, sinogram_params, lors):
    """
    Interpolate scatter sinogram to get per-LOR scatter estimates.

    Converts each measured LOR to sinogram coordinates (s, phi, z_avg, z_diff)
    using the same parameterization as plotting.py:create_sinogram(), then
    interpolates from the coarse scatter sinogram.

    The tail-fitted ``scatter_sino`` holds the scatter *count total per coarse
    bin*, but each bin aggregates many detector-pair LORs. The OSEM forward
    model needs a *per-LOR* scatter estimate on the same scale as the trues
    term a_i*(H x)_i. We therefore divide by the measured-LOR occupancy of each
    bin: s_i = scatter_sino[b] / occupancy[b] is the per-event scatter (the
    scatter fraction in bin b), which is bin-size invariant and O(1) per event,
    so SSS_SCALE_FACTOR stays ~1.0. If no occupancy is available we fall back to
    the raw per-bin total (legacy behaviour, requires manual SSS_SCALE_FACTOR).

    Parameters:
        scatter_sino: 4D array (n_s, n_phi, n_z_avg, n_z_diff)
        sinogram_params: dict with bin edges (and optional 'occupancy_sino')
        lors: (N, 6) measured LOR endpoints

    Returns:
        scatter_per_lor: (N,) scatter estimate for each LOR
    """
    from scipy.interpolate import RegularGridInterpolator

    # Convert the per-bin scatter total into a per-LOR density by dividing out
    # the measured-LOR occupancy of each bin (guarded against empty bins).
    occupancy_sino = sinogram_params.get('occupancy_sino')
    if occupancy_sino is not None:
        scatter_field = np.where(occupancy_sino > 0,
                                 scatter_sino / np.maximum(occupancy_sino, 1e-30),
                                 0.0)
    else:
        print("SSS WARNING: no occupancy_sino in sinogram_params — using raw "
              "per-bin scatter (per-LOR scale depends on bin size; tune "
              "SSS_SCALE_FACTOR manually).")
        scatter_field = scatter_sino

    x1, y1, z1 = lors[:, 0], lors[:, 1], lors[:, 2]
    x2, y2, z2 = lors[:, 3], lors[:, 4], lors[:, 5]

    dx = x2 - x1
    dy = y2 - y1
    lor_length_xy = np.sqrt(dx**2 + dy**2)
    lor_length_xy = np.maximum(lor_length_xy, 1e-10)

    # Signed perpendicular distance from origin to LOR
    s = (x1 * y2 - x2 * y1) / lor_length_xy

    # Angle of LOR direction
    phi = np.arctan2(dy, dx)

    # Normalize phi to [0, pi) — LORs are symmetric
    flip = phi < 0
    phi[flip] += np.pi
    s[flip] = -s[flip]
    wrap = phi >= np.pi
    phi[wrap] -= np.pi
    s[wrap] = -s[wrap]

    z_avg = (z1 + z2) / 2.0
    z_diff = z2 - z1

    # Build interpolator on bin centers
    sp = sinogram_params
    s_centers = (sp['s_edges'][:-1] + sp['s_edges'][1:]) / 2
    phi_centers = (sp['phi_edges'][:-1] + sp['phi_edges'][1:]) / 2
    z_avg_centers = (sp['z_avg_edges'][:-1] + sp['z_avg_edges'][1:]) / 2
    z_diff_centers = (sp['z_diff_edges'][:-1] + sp['z_diff_edges'][1:]) / 2

    interpolator = RegularGridInterpolator(
        (s_centers, phi_centers, z_avg_centers, z_diff_centers),
        scatter_field,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    query_points = np.column_stack([s, phi, z_avg, z_diff])
    scatter_per_lor = interpolator(query_points)

    scatter_per_lor = np.maximum(scatter_per_lor, 0.0).astype(np.float32)

    print(f"SSS interpolation: min={scatter_per_lor.min():.6e}, "
          f"max={scatter_per_lor.max():.6e}, mean={scatter_per_lor.mean():.6e}")

    return scatter_per_lor
