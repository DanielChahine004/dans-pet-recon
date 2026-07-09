from numba import cuda
import numpy as np
from tqdm import tqdm
from helpers.siddon import backproject_kernel


def backproject_gpu(lors, grid_size, voxel_size, batch_size=100_000):
    """
    Parameters:
    - lors: (N, 6) array of LOR endpoints [x1,y1,z1,x2,y2,z2] in mm
    - grid_size: tuple (nx, ny, nz) - voxel grid dimensions
    - voxel_size: voxel size in mm
    - batch_size: number of LORs per GPU batch (adjust based on GPU memory)
    Returns:
    - image: (nx, ny, nz) backprojected image
    """
    nx, ny, nz = grid_size
    n_voxels = nx * ny * nz
    n_lors = lors.shape[0]
    
    # Center grid at origin
    grid_origin = (
        -nx * voxel_size / 2.0,
        -ny * voxel_size / 2.0,
        -nz * voxel_size / 2.0
    )
    x0, y0, z0 = grid_origin
    
    # Initialize backprojection image
    backproj = np.zeros(n_voxels, dtype=np.float32)
    d_backproj = cuda.to_device(backproj)
    
    threads_per_block = 256
    
    print(f"="*50)
    print(f"STARTING BACKPROJECTION ON GPU...")
    print(f"="*50)
    print(f"Grid size: {grid_size}, Voxel size: {voxel_size} mm")
    print(f"Grid origin: ({x0:.1f}, {y0:.1f}, {z0:.1f}) mm")
    print(f"Total LORs: {n_lors:_}, Batch size: {batch_size:_}")
    print(f"="*50)
    print()
    
    # Process in batches
    n_batches = (n_lors + batch_size - 1) // batch_size
    for start_idx in tqdm(range(0, n_lors, batch_size), total=n_batches, desc="Backprojecting", unit="batch"):
        end_idx = min(start_idx + batch_size, n_lors)
        batch_lors = lors[start_idx:end_idx]
        n_batch = batch_lors.shape[0]

        d_lors = cuda.to_device(batch_lors.astype(np.float32))

        blocks_per_grid = (n_batch + threads_per_block - 1) // threads_per_block
        backproject_kernel[blocks_per_grid, threads_per_block](
            d_lors, d_backproj, nx, ny, nz, voxel_size, x0, y0, z0)
    
    # Copy result back to cpu and reshape
    backproj = d_backproj.copy_to_host()
    image = backproj.reshape(grid_size)
    print(f"BACKPROJECTION - DONE\n")

    return image


def backproject_gpu_streaming(lors_iterable, grid_size, voxel_size,
                               batch_size=1_000_000, total_lors=None):
    """
    Like backproject_gpu() but accepts a generator/iterable of LOR chunk arrays
    instead of a single large array.

    The GPU accumulation buffer is allocated once and kept resident across all
    chunks — LOR chunks stream through without accumulating in CPU RAM.

    Args:
        lors_iterable: iterable of (M_i, 6) float32 arrays (e.g. from
            iter_scanner_lors_chunked())
        grid_size: tuple (nx, ny, nz)
        voxel_size: voxel size in mm
        batch_size: GPU sub-batch size within each chunk
        total_lors: optional total LOR count for tqdm progress display
            (use total_scanner_lors() from sensitivity.py)

    Returns:
        image: (nx, ny, nz) backprojected image
    """
    nx, ny, nz = grid_size
    n_voxels = nx * ny * nz
    x0 = -nx * voxel_size / 2.0
    y0 = -ny * voxel_size / 2.0
    z0 = -nz * voxel_size / 2.0

    backproj = np.zeros(n_voxels, dtype=np.float32)
    d_backproj = cuda.to_device(backproj)
    threads_per_block = 256

    with tqdm(total=total_lors, desc="Backprojecting (streaming)", unit="LOR") as pbar:
        for lors_chunk in lors_iterable:
            for start in range(0, len(lors_chunk), batch_size):
                end = min(start + batch_size, len(lors_chunk))
                batch = lors_chunk[start:end].astype(np.float32)
                n_batch = len(batch)
                d_lors = cuda.to_device(batch)
                blocks = (n_batch + threads_per_block - 1) // threads_per_block
                backproject_kernel[blocks, threads_per_block](
                    d_lors, d_backproj, nx, ny, nz, voxel_size, x0, y0, z0)
                pbar.update(n_batch)

    return d_backproj.copy_to_host().reshape(grid_size)
