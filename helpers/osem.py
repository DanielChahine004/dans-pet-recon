from numba import cuda
import numpy as np
from tqdm import tqdm
from helpers.siddon import forward_project_kernel, backproject_weighted_kernel, backproject_kernel

def osem_gpu(lors, grid_size, voxel_size, sensitivity_image, n_iterations=10, n_subsets=8,
             batch_size=500_000, init_image=None, acf=None, scatter=None):
    """
    OSEM reconstruction with optional attenuation and scatter correction.

    Forward model: predicted_i = a_i * (H * x)_i + s_i
    Update:        x_j *= [sum_i H_ij * a_i / predicted_i] / S_j

    When acf=None and scatter=None, behaviour is identical to uncorrected OSEM.

    Parameters:
        lors: (N, 6) measured LOR endpoints
        grid_size: (nx, ny, nz)
        voxel_size: float, mm
        sensitivity_image: 3D array (attenuation-weighted if AC enabled)
        n_iterations: int
        n_subsets: int
        batch_size: int
        init_image: optional 3D initial estimate
        acf: optional (N,) ACF per measured LOR (exp(-mu_line_integral)). None = no AC.
        scatter: optional (N,) scatter estimate per measured LOR. None = no scatter.
    """
    nx, ny, nz = grid_size
    n_voxels = nx * ny * nz
    n_lors = lors.shape[0]

    grid_origin = (
        -nx * voxel_size / 2.0,
        -ny * voxel_size / 2.0,
        -nz * voxel_size / 2.0
    )
    x0, y0, z0 = grid_origin

    if init_image is None:
        image = np.ones(n_voxels, dtype=np.float32)
    else:
        image = init_image.flatten().astype(np.float32)
    threads_per_block = 256

    subset_indices = np.array_split(np.arange(n_lors), n_subsets)

    # Flatten sensitivity image for voxel indexing
    sensitivity_flat = sensitivity_image.flatten().astype(np.float32)

    # MAIN OSEM ITERATIONS
    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")

        for subset_idx, indices in enumerate(tqdm(subset_indices, desc="  Subsets", leave=False)):
            subset_lors = lors[indices]
            n_subset = len(indices)

            if n_subset == 0:
                continue

            # Extract correction arrays for this subset
            subset_acf = acf[indices] if acf is not None else None
            subset_scatter = scatter[indices] if scatter is not None else None

            ratio_backproj = np.zeros(n_voxels, dtype=np.float32)
            d_ratio_backproj = cuda.to_device(ratio_backproj)

            # Transfer image ONCE per subset (not per batch)
            d_image = cuda.to_device(image)

            for start in range(0, n_subset, batch_size):
                end = min(start + batch_size, n_subset)
                batch_lors = subset_lors[start:end]
                n_batch = len(batch_lors)

                d_batch_lors = cuda.to_device(batch_lors.astype(np.float32))
                d_projections = cuda.device_array(n_batch, dtype=np.float32)
                blocks = (n_batch + threads_per_block - 1) // threads_per_block

                forward_project_kernel[blocks, threads_per_block](
                    d_batch_lors, d_image, d_projections,
                    nx, ny, nz, voxel_size, x0, y0, z0
                )

                projections = d_projections.copy_to_host()

                # Compute ratio with attenuation and scatter corrections
                if subset_acf is not None:
                    batch_acf = subset_acf[start:end]
                    predicted = batch_acf * projections  # a_i * (H*x)_i
                else:
                    batch_acf = np.ones(n_batch, dtype=np.float32)
                    predicted = projections.copy()

                if subset_scatter is not None:
                    predicted += subset_scatter[start:end]  # + s_i

                predicted = np.maximum(predicted, 1e-10)

                # ratio_i = a_i / predicted_i
                ratios = batch_acf / predicted
                d_ratios = cuda.to_device(ratios.astype(np.float32))

                backproject_weighted_kernel[blocks, threads_per_block](
                    d_batch_lors, d_ratios, d_ratio_backproj,
                    nx, ny, nz, voxel_size, x0, y0, z0
                )

            ratio_backproj = d_ratio_backproj.copy_to_host()

            # Scale global sensitivity by fraction of LORs in this subset
            subset_norm = sensitivity_flat * (n_subset / n_lors)

            valid = subset_norm > 0
            image[valid] *= ratio_backproj[valid] / subset_norm[valid]
            image = np.maximum(image, 0.0)

    return image.reshape(grid_size)
