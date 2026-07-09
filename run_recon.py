from helpers.sensitivity import generate_scanner_lors, filter_lors_by_fov, iter_scanner_lors_chunked, total_scanner_lors
from helpers.attenuation import compute_acf, compute_attn_sensitivity, compute_attn_sensitivity_chunked
from helpers.diagnostics import plot_sss_diagnostics
from helpers.plotting import plot_interactive, plot_static
from helpers.backprojection import backproject_gpu, backproject_gpu_streaming
from helpers.data_io import load_lors, load_mu_map
from scipy.ndimage import gaussian_filter
from helpers.osem import osem_gpu
from numba import cuda, njit
from recon_config import *
from pathlib import Path
import numpy as np
import pickle
import shutil
import sys
import os

# ---- Configuration validation ----
assert RECON_ALGORITHM in ["Unfiltered_backprojection", "OSEM", "None"], \
    f"Invalid RECON_ALGORITHM: {RECON_ALGORITHM}"

if SCATTER_CORRECTION:
    assert MU_MAP_PATH is not None, \
        "SCATTER_CORRECTION requires MU_MAP_PATH to be set (scatter correction needs attenuation data)"
    assert ATTENUATION_CORRECTION, \
        "SCATTER_CORRECTION requires ATTENUATION_CORRECTION to be enabled"

if ATTENUATION_CORRECTION and MU_MAP_PATH is not None:
    assert Path(MU_MAP_PATH).exists(), \
        f"MU_MAP_PATH does not exist: {MU_MAP_PATH}"

# ---- Clear output directory ----
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
    print(f"Cleared output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_scanner_details():
    print("="*50)
    print("SCANNER GEOMETRY")
    print("="*50)
    print(f"Axial rings:            {N_AXIAL_RINGS}")
    print(f"Detectors per ring:     {N_DETECTORS_PER_RING}")
    print(f"Total detectors:        {N_AXIAL_RINGS * N_DETECTORS_PER_RING}")
    print(f"Inner radius:           {INNER_RADIUS} mm")
    print(f"Outer radius:           {OUTER_RADIUS} mm")
    print("="*50)


def print_recon_details():
    print("RECONSTRUCTION PROPERTIES")
    print("="*50)
    print(f"Voxel size:             {VOXEL_SIZE_IN_MM} mm")
    print(f"Reconstruction grid:    {IMAGE_GRID_SIZE}")
    print(f"Image space size:       ({IMAGE_GRID_SIZE[0]*VOXEL_SIZE_IN_MM}, {IMAGE_GRID_SIZE[1]*VOXEL_SIZE_IN_MM}, {IMAGE_GRID_SIZE[2]*VOXEL_SIZE_IN_MM}) mm")
    print(f"Algorithm:              {RECON_ALGORITHM}")
    print(f"Sensitivity correction: {SENSITIVITY_CORRECTION}")
    print(f"Smooth Sensitivity: {SMOOTH_SENSITIVITY}")
    print(f"Attenuation correction: {ATTENUATION_CORRECTION}")
    print(f"Scatter correction:     {SCATTER_CORRECTION}")
    if RECON_ALGORITHM == "OSEM":
        print(f"OSEM iterations:        {OSEM_ITERATIONS}")
        print(f"OSEM subsets:           {OSEM_SUBSETS}")
    print("="*50 + "\n")


def print_processing_details():    
    print("\n")
    print("="*50)
    print("PROCESSING INFO")
    print("="*50)
    print(f"CUDA available: {cuda.is_available()}")
    if cuda.is_available():
        print(f"GPU: {cuda.get_current_device().name.decode()}")


def save_config():
    config = {
        "OUTPUT_DIR": OUTPUT_DIR,
        "PATH_TO_LORS": PATH_TO_LORS,
        "RECON_ALGORITHM": RECON_ALGORITHM,
        "SENSITIVITY_CORRECTION": SENSITIVITY_CORRECTION,
        "SMOOTH_SENSITIVITY": SMOOTH_SENSITIVITY,
        "GAUSSIAN_SIGMA": GAUSSIAN_SIGMA,
        "GAUSSIAN_RADIUS": GAUSSIAN_RADIUS,
        "IMAGE_GRID_SIZE": IMAGE_GRID_SIZE,
        "VOXEL_SIZE_IN_MM": VOXEL_SIZE_IN_MM,
        "N_AXIAL_RINGS": N_AXIAL_RINGS,
        "N_DETECTORS_PER_RING": N_DETECTORS_PER_RING,
        "AXIAL_LENGTH": AXIAL_LENGTH,
        "INNER_RADIUS": INNER_RADIUS,
        "OUTER_RADIUS": OUTER_RADIUS,
        "OSEM_ITERATIONS": OSEM_ITERATIONS if RECON_ALGORITHM == "OSEM" else None,
        "OSEM_SUBSETS": OSEM_SUBSETS if RECON_ALGORITHM == "OSEM" else None,
        "FOV_RADIUS": FOV_RADIUS,
        "FOV_AXIAL_HALF_LENGTH": FOV_AXIAL_HALF_LENGTH,
        "ATTENUATION_CORRECTION": ATTENUATION_CORRECTION,
        "MU_MAP_PATH": MU_MAP_PATH,
        "SCATTER_CORRECTION": SCATTER_CORRECTION,
        "SSS_COARSE_FACTOR": SSS_COARSE_FACTOR,
        "SSS_N_ANGULAR_DETECTORS": SSS_N_ANGULAR_DETECTORS,
        "SSS_N_AXIAL_DETECTORS": SSS_N_AXIAL_DETECTORS,
        "SSS_START_ITERATION": SSS_START_ITERATION,
        "SSS_UPDATE_INTERVAL": SSS_UPDATE_INTERVAL,
        "SSS_SCALE_FACTOR": SSS_SCALE_FACTOR,
        "FILTER_OOB_LORS": FILTER_OOB_LORS,
    }
        
    with open(os.path.join(OUTPUT_DIR, "recon_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"Configuration saved to {OUTPUT_DIR}/recon_config.pkl")


print_processing_details()
lor_data = load_lors(PATH_TO_LORS)
print(f"LORs loaded shape: ({lor_data.shape[0]:_}, {lor_data.shape[1]:_})")

if FILTER_OOB_LORS:
    axial_half = AXIAL_LENGTH / 2.0
    r1 = np.sqrt(lor_data[:, 0]**2 + lor_data[:, 1]**2)
    r2 = np.sqrt(lor_data[:, 3]**2 + lor_data[:, 4]**2)
    valid = (
        (r1 >= INNER_RADIUS) & (r1 <= OUTER_RADIUS) &
        (r2 >= INNER_RADIUS) & (r2 <= OUTER_RADIUS) &
        (np.abs(lor_data[:, 2]) <= axial_half) &
        (np.abs(lor_data[:, 5]) <= axial_half)
    )
    n_removed = int((~valid).sum())
    lor_data = lor_data[valid]
    print(f"OOB LOR filter: removed {n_removed:_} LORs ({100*n_removed/(n_removed+lor_data.shape[0]):.2f}%), "
          f"{lor_data.shape[0]:_} remaining")

print("="*50 + '\n')
print_scanner_details()
print_recon_details()

recon_image = None
backprojection_image = None
sensitivity_image = None
mu_map = None
acf_measured = None
scatter_estimate = None

# With buffer for partial voxel overlap
# lor_data = filter_lors_by_fov(
#     lor_data,
#     fov_radius=FOV_RADIUS,
#     fov_axial_half_length=FOV_AXIAL_HALF_LENGTH,
#     buffer=VOXEL_SIZE_IN_MM * 3  # optional margin
# )

# ---- Attenuation correction setup ----
if ATTENUATION_CORRECTION:
    print("="*50)
    print("ATTENUATION CORRECTION")
    print("="*50)
    mu_map = load_mu_map(MU_MAP_PATH, IMAGE_GRID_SIZE)
    acf_measured = compute_acf(lor_data, mu_map, IMAGE_GRID_SIZE, VOXEL_SIZE_IN_MM)

# ---- Sensitivity image ----
if SENSITIVITY_CORRECTION == True:
    n_scanner_lors = total_scanner_lors(N_AXIAL_RINGS, N_DETECTORS_PER_RING)

    if ATTENUATION_CORRECTION:
        # Single-pass attenuation-weighted sensitivity — streams LOR chunks, never
        # materialises the full scanner_lors or acf_scanner arrays in CPU RAM.
        print("Computing attenuation-weighted sensitivity (streaming)...")
        sensitivity_image = compute_attn_sensitivity_chunked(
            iter_scanner_lors_chunked(N_AXIAL_RINGS, N_DETECTORS_PER_RING,
                                      INNER_RADIUS, AXIAL_LENGTH,
                                      chunk_size=SENSITIVITY_CHUNK_SIZE,
                                      balance_directions=True),
            mu_map, IMAGE_GRID_SIZE, VOXEL_SIZE_IN_MM,
            batch_size=SENSITIVITY_BATCH_SIZE,
            total_lors=n_scanner_lors,
        )
    else:
        # Geometric sensitivity (backprojection of ones) — streamed to avoid OOM
        sensitivity_image = backproject_gpu_streaming(
            iter_scanner_lors_chunked(N_AXIAL_RINGS, N_DETECTORS_PER_RING,
                                      INNER_RADIUS, AXIAL_LENGTH,
                                      chunk_size=SENSITIVITY_CHUNK_SIZE,
                                      balance_directions=True),
            grid_size=IMAGE_GRID_SIZE,
            voxel_size=VOXEL_SIZE_IN_MM,
            batch_size=SENSITIVITY_BATCH_SIZE,
            total_lors=n_scanner_lors,
        )

    if SMOOTH_SENSITIVITY:
        sensitivity_image = gaussian_filter(sensitivity_image, sigma=GAUSSIAN_SIGMA, radius=GAUSSIAN_RADIUS)
else:
    sensitivity_image = np.ones(IMAGE_GRID_SIZE)

# ---- Scatter correction setup ----
if SCATTER_CORRECTION and RECON_ALGORITHM == "OSEM":
    from helpers.scatter import (
        identify_scatter_points, define_coarse_detectors,
        compute_scatter_sinogram, scatter_sinogram_to_lor,
    )

    if not ATTENUATION_CORRECTION:
        print("WARNING: Scatter correction requires attenuation correction. Enabling AC.")
        mu_map = load_mu_map(MU_MAP_PATH, IMAGE_GRID_SIZE)
        acf_measured = compute_acf(lor_data, mu_map, IMAGE_GRID_SIZE, VOXEL_SIZE_IN_MM)

    scatter_points, scatter_ne, delta_V = identify_scatter_points(
        mu_map, IMAGE_GRID_SIZE, VOXEL_SIZE_IN_MM, coarse_factor=SSS_COARSE_FACTOR,
    )
    coarse_detectors = define_coarse_detectors(
        n_angular=SSS_N_ANGULAR_DETECTORS, n_axial=SSS_N_AXIAL_DETECTORS,
        radius=INNER_RADIUS, axial_length=AXIAL_LENGTH,
    )

# ---- Reconstruction ----
# Backprojection is the first step for both algorithms
if RECON_ALGORITHM in ["Unfiltered_backprojection", "OSEM"]:
    backprojection_image = backproject_gpu(
        lors=lor_data,
        grid_size=IMAGE_GRID_SIZE,
        voxel_size=VOXEL_SIZE_IN_MM,
        batch_size=1_000_000,
    )
    # Initialize recon_image with sensitivity-normalized backprojection
    recon_image = backprojection_image.copy()
    if SENSITIVITY_CORRECTION:
        nonzero_mask = sensitivity_image > 0
        recon_image[nonzero_mask] /= sensitivity_image[nonzero_mask]
        recon_image[~nonzero_mask] = 0

if RECON_ALGORITHM == "OSEM":

    if SCATTER_CORRECTION:
        # Iterative OSEM with SSS updates
        for osem_iter in range(OSEM_ITERATIONS):
            # Update scatter estimate at configured intervals
            if osem_iter >= SSS_START_ITERATION and (osem_iter - SSS_START_ITERATION) % SSS_UPDATE_INTERVAL == 0:
                print(f"\n--- Updating SSS scatter estimate (iteration {osem_iter + 1}) ---")
                scatter_sino, sino_params = compute_scatter_sinogram(
                    mu_map=mu_map,
                    activity_image=recon_image,
                    grid_size=IMAGE_GRID_SIZE,
                    voxel_size=VOXEL_SIZE_IN_MM,
                    scatter_points=scatter_points,
                    scatter_ne=scatter_ne,
                    delta_V=delta_V,
                    detectors=coarse_detectors,
                    n_angular=SSS_N_ANGULAR_DETECTORS,
                    n_axial=SSS_N_AXIAL_DETECTORS,
                    radius=INNER_RADIUS,
                    axial_length=AXIAL_LENGTH,
                    measured_lors=lor_data,
                )
                scatter_estimate = scatter_sinogram_to_lor(scatter_sino, sino_params, lor_data)
                scatter_estimate *= SSS_SCALE_FACTOR

            # Run one OSEM iteration
            recon_image = osem_gpu(
                lors=lor_data,
                grid_size=IMAGE_GRID_SIZE,
                voxel_size=VOXEL_SIZE_IN_MM,
                sensitivity_image=sensitivity_image,
                init_image=recon_image,
                n_iterations=1,
                n_subsets=OSEM_SUBSETS,
                batch_size=1_000_000,
                acf=acf_measured,
                scatter=scatter_estimate,
            )
    else:
        # OSEM with attenuation correction only (or no corrections)
        recon_image = osem_gpu(
            lors=lor_data,
            grid_size=IMAGE_GRID_SIZE,
            voxel_size=VOXEL_SIZE_IN_MM,
            sensitivity_image=sensitivity_image,
            init_image=recon_image,
            n_iterations=OSEM_ITERATIONS,
            n_subsets=OSEM_SUBSETS,
            batch_size=1_000_000,
            acf=acf_measured,
        )

elif RECON_ALGORITHM == "None":
    recon_image = np.zeros(IMAGE_GRID_SIZE)

np.save(os.path.join(OUTPUT_DIR, "reconstruction_image.npy"), recon_image)
np.save(os.path.join(OUTPUT_DIR, "sensitivity_image.npy"), sensitivity_image)
if backprojection_image is not None:
    np.save(os.path.join(OUTPUT_DIR, "backprojection_image.npy"), backprojection_image)
if SCATTER_CORRECTION and scatter_estimate is not None:
    try:
        np.save(os.path.join(OUTPUT_DIR, "scatter_sino_raw.npy"), sino_params['scatter_sino_raw'])
        np.save(os.path.join(OUTPUT_DIR, "scatter_sino_fitted.npy"), scatter_sino)
        np.save(os.path.join(OUTPUT_DIR, "scatter_estimate_per_lor.npy"), scatter_estimate)
        np.savez(os.path.join(OUTPUT_DIR, "sino_params.npz"),
                 s_edges=sino_params['s_edges'], phi_edges=sino_params['phi_edges'],
                 z_avg_edges=sino_params['z_avg_edges'], z_diff_edges=sino_params['z_diff_edges'])
    except NameError:
        pass
save_config()

# ---- Diagnostic outputs ----

if SCATTER_CORRECTION and scatter_estimate is not None:
    # scatter_sino and sino_params persist from the last SSS update
    try:
        # Auto-detect object radius from mu-map (same logic as in scatter.py)
        nx, ny, nz = IMAGE_GRID_SIZE
        half = nx * VOXEL_SIZE_IN_MM / 2.0
        iy_g, ix_g = np.meshgrid(np.arange(ny), np.arange(nx))
        r_coords = np.sqrt((-half + (ix_g + 0.5) * VOXEL_SIZE_IN_MM)**2 +
                           (-half + (iy_g + 0.5) * VOXEL_SIZE_IN_MM)**2)
        mu_proj = mu_map.max(axis=2)
        obj_r = r_coords[mu_proj > 0.001].max() + VOXEL_SIZE_IN_MM if mu_map.max() > 0.001 else FOV_RADIUS

        plot_sss_diagnostics(
            scatter_sino=scatter_sino,
            sinogram_params=sino_params,
            scatter_per_lor=scatter_estimate,
            measured_lors=lor_data,
            object_radius=obj_r,
            output_dir=OUTPUT_DIR,
        )
    except NameError:
        print("SSS diagnostics skipped (no scatter sinogram computed).")