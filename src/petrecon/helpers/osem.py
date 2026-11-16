import math

from numba import cuda
import numpy as np
from tqdm import tqdm
from .siddon import forward_project_kernel, backproject_weighted_kernel

THREADS = 256


@cuda.jit
def _ratio_kernel(projections, acf, scatter, has_acf, has_scatter, ratios):
    """ratio_i = a_i / max(a_i * (H x)_i + s_i, 1e-10)  (a_i = 1 without attenuation, s_i = 0 without scatter)."""
    i = cuda.grid(1)
    if i >= projections.shape[0]:
        return
    a = acf[i] if has_acf else 1.0
    predicted = a * projections[i]
    if has_scatter:
        predicted += scatter[i]
    ratios[i] = a / max(predicted, 1e-10)


@cuda.jit
def _zero_kernel(a):
    j = cuda.grid(1)
    if j < a.shape[0]:
        a[j] = 0.0


@cuda.jit
def _update_kernel(image, ratio_backproj, sensitivity, subset_fraction):
    """x_j *= bp_j / (S_j * n_subset / n_lors) where that is > 0, then x_j = max(x_j, 0)."""
    j = cuda.grid(1)
    if j >= image.shape[0]:
        return
    norm = sensitivity[j] * subset_fraction
    if norm > 0:
        image[j] = max(image[j] * (ratio_backproj[j] / norm), 0.0)
    else:
        image[j] = max(image[j], 0.0)


def _direction_order(lors):
    """Order of the LORs by direction (azimuth, polar angle), then radial offset and axial position.

    Neighbouring GPU threads then trace neighbouring rays, which share cache lines in the forward projection and spread
    the atomic adds of the backprojection; the result is the same (up to float32 summation order). Uses CuPy on the
    GPU when available, NumPy otherwise.
    """
    try:
        import cupy as xp
    except Exception:  # pragma: no cover - CuPy is optional
        xp = np
    L = xp.asarray(lors, dtype=xp.float32)
    d = L[:, 3:] - L[:, :3]
    norm = xp.sqrt((d * d).sum(1)) + 1e-12
    phi = xp.mod(xp.arctan2(d[:, 1], d[:, 0]), np.pi)
    phi_bin = xp.floor(phi * (200 / np.pi)).astype(xp.int64)                 # 200 azimuthal bins
    theta_bin = xp.floor((d[:, 2] / norm + 1) * 10).astype(xp.int64)        # 20 polar bins
    s = L[:, 0] * xp.sin(phi) - L[:, 1] * xp.cos(phi)                         # signed radial offset (mm)
    s_bin = xp.clip(xp.floor(s + 512), 0, 1023).astype(xp.int64)             # 1 mm bins
    z_bin = xp.clip(xp.floor(0.5 * (L[:, 2] + L[:, 5]) + 512), 0, 1023).astype(xp.int64)
    key = ((phi_bin * 32 + theta_bin) * 1024 + s_bin) * 1024 + z_bin
    order = xp.argsort(key)
    return order.get() if xp is not np else order


def osem_gpu(lors, grid_size, voxel_size, sensitivity_image, n_iterations=10, n_subsets=8,
             batch_size=500_000, init_image=None, acf=None, scatter=None, sort_lors=True, keep_on_gpu=None):
    """
    OSEM reconstruction with optional attenuation and scatter correction.

    Forward model: predicted_i = a_i * (H * x)_i + s_i
    Update:        x_j *= [sum_i H_ij * a_i / predicted_i] / S_j

    When acf=None and scatter=None, behaviour is identical to uncorrected OSEM.

    The whole loop runs on the GPU: the image, the sensitivity image and (when they fit in device memory) the LORs of
    every subset stay resident; only the final image is copied back. Subsets are the same contiguous blocks of the
    input order as before (np.array_split); within a subset the LORs are processed in direction order (sort_lors),
    which is faster and gives the same result up to float32 summation order.

    Parameters:
        lors: (N, 6) measured LOR endpoints
        grid_size: (nx, ny, nz)
        voxel_size: float, mm
        sensitivity_image: 3D array (attenuation-weighted if AC enabled)
        n_iterations: int
        n_subsets: int
        batch_size: int, LORs per kernel launch (and per host->device copy when the LORs do not fit on the GPU)
        init_image: optional 3D initial estimate
        acf: optional (N,) ACF per measured LOR (exp(-mu_line_integral)). None = no AC.
        scatter: optional (N,) scatter estimate per measured LOR. None = no scatter.
        sort_lors: process the LORs of each subset in direction order (default True)
        keep_on_gpu: keep all LORs on the GPU (True), stream them from the host per batch (False), or decide from the
            free device memory (None, default)
    """
    nx, ny, nz = grid_size
    n_voxels = nx * ny * nz
    n_lors = lors.shape[0]
    x0, y0, z0 = -nx * voxel_size / 2.0, -ny * voxel_size / 2.0, -nz * voxel_size / 2.0

    if init_image is None:
        image = np.ones(n_voxels, dtype=np.float32)
    else:
        image = np.ascontiguousarray(init_image, dtype=np.float32).ravel()
    d_image = cuda.to_device(image)
    d_sens = cuda.to_device(np.ascontiguousarray(sensitivity_image, dtype=np.float32).ravel())
    d_bp = cuda.device_array(n_voxels, dtype=np.float32)
    vox_blocks = (n_voxels + THREADS - 1) // THREADS

    subset_indices = np.array_split(np.arange(n_lors), n_subsets)
    if sort_lors:
        subset_indices = [idx[_direction_order(lors[idx])] if len(idx) > 1 else idx for idx in subset_indices]

    # Keep every subset's LORs (and corrections) on the GPU if they fit in half of the free memory
    per_lor_bytes = 6 * 4 + 4 + 4 + (4 if acf is not None else 0) + (4 if scatter is not None else 0)
    free_bytes = cuda.current_context().get_memory_info()[0]
    resident = (n_lors * per_lor_bytes < 0.5 * free_bytes) if keep_on_gpu is None else bool(keep_on_gpu)
    dummy = cuda.device_array(1, dtype=np.float32)

    def upload(idx):
        d_l = cuda.to_device(np.ascontiguousarray(lors[idx], dtype=np.float32))
        d_a = cuda.to_device(np.ascontiguousarray(acf[idx], dtype=np.float32)) if acf is not None else dummy
        d_s = cuda.to_device(np.ascontiguousarray(scatter[idx], dtype=np.float32)) if scatter is not None else dummy
        return d_l, d_a, d_s

    subset_data = [upload(idx) if (resident and len(idx)) else None for idx in subset_indices]
    max_batch = max((len(idx) for idx in subset_indices), default=0) if resident else min(batch_size, n_lors)
    d_proj = cuda.device_array(max(max_batch, 1), dtype=np.float32)
    d_ratio = cuda.device_array(max(max_batch, 1), dtype=np.float32)

    def process(d_l, d_a, d_s, n):
        """Forward project, ratio and backproject n LORs (device arrays) into d_bp."""
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            m = end - start
            blocks = (m + THREADS - 1) // THREADS
            l = d_l[start:end]
            p, r = d_proj[:m], d_ratio[:m]
            forward_project_kernel[blocks, THREADS](l, d_image, p, nx, ny, nz, voxel_size, x0, y0, z0)
            _ratio_kernel[blocks, THREADS](p, d_a[start:end] if acf is not None else dummy,
                                           d_s[start:end] if scatter is not None else dummy,
                                           acf is not None, scatter is not None, r)
            backproject_weighted_kernel[blocks, THREADS](l, r, d_bp, nx, ny, nz, voxel_size, x0, y0, z0)

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")
        for subset_idx, indices in enumerate(tqdm(subset_indices, desc="  Subsets", leave=False)):
            n_subset = len(indices)
            if n_subset == 0:
                continue
            _zero_kernel[vox_blocks, THREADS](d_bp)
            if resident:
                process(*subset_data[subset_idx], n_subset)
            else:
                for start in range(0, n_subset, batch_size):
                    chunk = indices[start:start + batch_size]
                    process(*upload(chunk), len(chunk))
            # Scale global sensitivity by fraction of LORs in this subset
            _update_kernel[vox_blocks, THREADS](d_image, d_bp, d_sens, np.float32(n_subset / n_lors))

    return d_image.copy_to_host().reshape(grid_size)
