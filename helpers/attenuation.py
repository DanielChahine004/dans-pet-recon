"""
Attenuation Correction for PET Reconstruction
===============================================
Computes attenuation correction factors (ACFs) by reusing the existing
forward_project_kernel from siddon.py — passing the mu-map as the image
to get line integrals of mu along each LOR.

ACF_i = exp(-integral of mu along LOR_i)
"""

import numpy as np
from numba import cuda
from tqdm import tqdm
from helpers.siddon import forward_project_kernel, backproject_weighted_kernel


def compute_acf(lors, mu_map, grid_size, voxel_size, batch_size=1_000_000):
    """
    Compute attenuation correction factors for each LOR.

    ACF_i = exp(-integral of mu along LOR_i)

    Parameters:
        lors: np.ndarray, shape (N, 6) - LOR endpoints [x1,y1,z1,x2,y2,z2]
        mu_map: np.ndarray, shape grid_size - 3D mu values in mm^-1
        grid_size: tuple (nx, ny, nz)
        voxel_size: float - voxel size in mm
        batch_size: int - GPU batch size

    Returns:
        acf: np.ndarray, shape (N,) - ACF values in (0, 1] for each LOR
    """
    nx, ny, nz = grid_size
    n_lors = lors.shape[0]

    grid_origin = (
        -nx * voxel_size / 2.0,
        -ny * voxel_size / 2.0,
        -nz * voxel_size / 2.0,
    )
    x0, y0, z0 = grid_origin

    mu_flat = mu_map.flatten().astype(np.float32)
    d_mu = cuda.to_device(mu_flat)

    mu_line_integrals = np.zeros(n_lors, dtype=np.float32)

    threads_per_block = 256

    for start in tqdm(range(0, n_lors, batch_size), desc="Computing ACFs"):
        end = min(start + batch_size, n_lors)
        batch_lors = lors[start:end].astype(np.float32)
        n_batch = end - start

        d_batch_lors = cuda.to_device(batch_lors)
        d_projections = cuda.device_array(n_batch, dtype=np.float32)
        blocks = (n_batch + threads_per_block - 1) // threads_per_block

        forward_project_kernel[blocks, threads_per_block](
            d_batch_lors, d_mu, d_projections,
            nx, ny, nz, voxel_size, x0, y0, z0
        )

        mu_line_integrals[start:end] = d_projections.copy_to_host()

    acf = np.exp(-mu_line_integrals)

    print(f"ACF statistics: min={acf.min():.4f}, max={acf.max():.4f}, mean={acf.mean():.4f}")
    print(f"  LORs with >50% attenuation: {np.sum(acf < 0.5)} / {n_lors}")

    return acf


def compute_attn_sensitivity(scanner_lors, acf_scanner, grid_size, voxel_size, batch_size=1_000_000):
    """
    Compute attenuation-weighted sensitivity image.

    S_j = sum_i acf_i * H_ij

    Replaces geometric sensitivity (backprojection of ones) when AC is enabled.

    Parameters:
        scanner_lors: np.ndarray, shape (M, 6) - all scanner LOR endpoints
        acf_scanner: np.ndarray, shape (M,) - ACF for each scanner LOR
        grid_size: tuple (nx, ny, nz)
        voxel_size: float
        batch_size: int

    Returns:
        sensitivity: np.ndarray, shape grid_size - 3D attenuation-weighted sensitivity
    """
    nx, ny, nz = grid_size
    n_voxels = nx * ny * nz
    n_lors = scanner_lors.shape[0]

    grid_origin = (
        -nx * voxel_size / 2.0,
        -ny * voxel_size / 2.0,
        -nz * voxel_size / 2.0,
    )
    x0, y0, z0 = grid_origin

    sensitivity = np.zeros(n_voxels, dtype=np.float32)
    d_sensitivity = cuda.to_device(sensitivity)

    threads_per_block = 256

    for start in tqdm(range(0, n_lors, batch_size), desc="Attn-weighted sensitivity"):
        end = min(start + batch_size, n_lors)
        batch_lors = scanner_lors[start:end].astype(np.float32)
        batch_acf = acf_scanner[start:end].astype(np.float32)
        n_batch = end - start

        d_batch_lors = cuda.to_device(batch_lors)
        d_batch_acf = cuda.to_device(batch_acf)
        blocks = (n_batch + threads_per_block - 1) // threads_per_block

        backproject_weighted_kernel[blocks, threads_per_block](
            d_batch_lors, d_batch_acf, d_sensitivity,
            nx, ny, nz, voxel_size, x0, y0, z0
        )

    return d_sensitivity.copy_to_host().reshape(grid_size)


def compute_attn_sensitivity_chunked(lors_iterable, mu_map, grid_size, voxel_size,
                                      batch_size=1_000_000, total_lors=None):
    """
    Single-pass attenuation-weighted sensitivity using a streaming LOR generator.

    For each LOR chunk: forward-project mu-map → compute ACF → weighted backproject.
    Never stores the full scanner_lors or acf_scanner arrays in CPU RAM.

    Replaces the two-call pattern:
        acf_scanner = compute_acf(scanner_lors, ...)
        sensitivity  = compute_attn_sensitivity(scanner_lors, acf_scanner, ...)

    Args:
        lors_iterable: iterable of (M_i, 6) float32 LOR chunk arrays (e.g. from
            iter_scanner_lors_chunked())
        mu_map: np.ndarray, shape grid_size — 3D mu values in mm^-1
        grid_size: tuple (nx, ny, nz)
        voxel_size: float — voxel size in mm
        batch_size: GPU sub-batch size within each chunk
        total_lors: optional total LOR count for tqdm progress display

    Returns:
        sensitivity: np.ndarray, shape grid_size — attenuation-weighted sensitivity
    """
    nx, ny, nz = grid_size
    n_voxels = nx * ny * nz
    x0 = -nx * voxel_size / 2.0
    y0 = -ny * voxel_size / 2.0
    z0 = -nz * voxel_size / 2.0

    mu_flat = mu_map.flatten().astype(np.float32)
    d_mu = cuda.to_device(mu_flat)

    sensitivity = np.zeros(n_voxels, dtype=np.float32)
    d_sensitivity = cuda.to_device(sensitivity)
    threads_per_block = 256

    with tqdm(total=total_lors, desc="Attn-weighted sensitivity (streaming)", unit="LOR") as pbar:
        for lors_chunk in lors_iterable:
            for start in range(0, len(lors_chunk), batch_size):
                end = min(start + batch_size, len(lors_chunk))
                batch_lors = lors_chunk[start:end].astype(np.float32)
                n_batch = len(batch_lors)

                d_batch_lors = cuda.to_device(batch_lors)
                d_projections = cuda.device_array(n_batch, dtype=np.float32)
                blocks = (n_batch + threads_per_block - 1) // threads_per_block

                # Forward-project mu-map to get line integrals
                forward_project_kernel[blocks, threads_per_block](
                    d_batch_lors, d_mu, d_projections,
                    nx, ny, nz, voxel_size, x0, y0, z0)

                # ACF = exp(-line_integral), small batch only — never stored in full
                acf_batch = np.exp(-d_projections.copy_to_host()).astype(np.float32)
                d_acf_batch = cuda.to_device(acf_batch)

                # Accumulate weighted backprojection into persistent GPU buffer
                backproject_weighted_kernel[blocks, threads_per_block](
                    d_batch_lors, d_acf_batch, d_sensitivity,
                    nx, ny, nz, voxel_size, x0, y0, z0)

                pbar.update(n_batch)

    return d_sensitivity.copy_to_host().reshape(grid_size)


def generate_mu_map_for_recon(grid_size, voxel_size, phantom_radius=80.0, phantom_height=70.0,
                               mu_water=0.009597, mu_air=1.05e-5):
    """
    Generate mu-map analytically on the reconstruction grid.

    Avoids grid mismatch with simulation mu-map (200^3) vs reconstruction grid (128^3).
    Uses phantom geometry from config.py.

    Parameters:
        grid_size: tuple (nx, ny, nz)
        voxel_size: float - mm
        phantom_radius: float - mm (default: 80mm NEMA IEC)
        phantom_height: float - mm (default: 70mm NEMA IEC)
        mu_water: float - linear attenuation coefficient for water at 511 keV (mm^-1)
        mu_air: float - linear attenuation coefficient for air at 511 keV (mm^-1)

    Returns:
        mu_map: np.ndarray, shape grid_size
    """
    nx, ny, nz = grid_size
    half_x = nx * voxel_size / 2.0
    half_y = ny * voxel_size / 2.0
    half_z = nz * voxel_size / 2.0

    x = np.linspace(-half_x + voxel_size / 2, half_x - voxel_size / 2, nx)
    y = np.linspace(-half_y + voxel_size / 2, half_y - voxel_size / 2, ny)
    z = np.linspace(-half_z + voxel_size / 2, half_z - voxel_size / 2, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    R = np.sqrt(X**2 + Y**2)

    mu_map = np.full(grid_size, mu_air, dtype=np.float32)
    phantom_mask = (R <= phantom_radius) & (np.abs(Z) <= phantom_height / 2)
    mu_map[phantom_mask] = mu_water

    print(f"Mu-map generated: shape={mu_map.shape}")
    print(f"  mu(water)={mu_water:.6f} mm^-1, mu(air)={mu_air:.2e} mm^-1")
    print(f"  Water voxels: {np.sum(phantom_mask):,} / {np.prod(grid_size):,}")

    return mu_map
