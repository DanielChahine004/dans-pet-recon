"""Core reconstruction pipeline (formerly run_recon.py)."""
import os
import pickle
import shutil

import numpy as np
from numba import cuda
from scipy.ndimage import gaussian_filter

from petrecon.config import ReconConfig
from petrecon.helpers.attenuation import compute_acf
from petrecon.helpers.backprojection import backproject_gpu
from petrecon.helpers.data_io import load_lors, load_mu_map
from petrecon.helpers.diagnostics import plot_sss_diagnostics
from petrecon.helpers.osem import osem_gpu
from petrecon.helpers.sensitivity import sensitivity_gpu


def print_scanner_details(cfg: ReconConfig):
    print("=" * 50)
    print("SCANNER GEOMETRY")
    print("=" * 50)
    print(f"Axial rings:            {cfg.n_axial_rings}")
    print(f"Detectors per ring:     {cfg.n_detectors_per_ring}")
    print(f"Total detectors:        {cfg.n_axial_rings * cfg.n_detectors_per_ring}")
    print(f"Inner radius:           {cfg.inner_radius} mm")
    print(f"Outer radius:           {cfg.outer_radius} mm")
    print("=" * 50)


def print_recon_details(cfg: ReconConfig):
    print("RECONSTRUCTION PROPERTIES")
    print("=" * 50)
    gx, gy, gz = cfg.image_grid_size
    print(f"Voxel size:             {cfg.voxel_size_in_mm} mm")
    print(f"Reconstruction grid:    {cfg.image_grid_size}")
    print(f"Image space size:       ({gx * cfg.voxel_size_in_mm}, {gy * cfg.voxel_size_in_mm}, {gz * cfg.voxel_size_in_mm}) mm")
    print(f"Algorithm:              {cfg.recon_algorithm}")
    print(f"Sensitivity correction: {cfg.sensitivity_correction}")
    print(f"Sensitivity mode:       {cfg.sensitivity_mode}"
          + (f" ({cfg.sensitivity_samples} samples)" if cfg.sensitivity_mode == "continuous" else ""))
    blur_sigma, blur_radius = cfg.sensitivity_blur_voxels()
    print(f"Smooth Sensitivity: {cfg.smooth_sensitivity}"
          + (f" (sigma {blur_sigma:g} vox = {blur_sigma * cfg.voxel_size_in_mm:g} mm, radius {blur_radius} vox)"
             if cfg.smooth_sensitivity else ""))
    print(f"Attenuation correction: {cfg.attenuation_correction}")
    print(f"Scatter correction:     {cfg.scatter_correction}")
    if cfg.recon_algorithm == "OSEM":
        print(f"OSEM iterations:        {cfg.osem_iterations}")
        print(f"OSEM subsets:           {cfg.osem_subsets}")
    print("=" * 50 + "\n")


def print_processing_details():
    print("\n")
    print("=" * 50)
    print("PROCESSING INFO")
    print("=" * 50)
    print(f"CUDA available: {cuda.is_available()}")
    if cuda.is_available():
        print(f"GPU: {cuda.get_current_device().name.decode()}")


def save_config(cfg: ReconConfig):
    with open(os.path.join(cfg.output_dir, "recon_config.pkl"), "wb") as f:
        pickle.dump(cfg.to_legacy_dict(), f)
    print(f"Configuration saved to {cfg.output_dir}/recon_config.pkl")


def run_reconstruction(cfg: ReconConfig):
    """Run a full reconstruction described by cfg; writes outputs to cfg.output_dir."""
    cfg.validate()

    # ---- Clear output directory ----
    if os.path.exists(cfg.output_dir):
        shutil.rmtree(cfg.output_dir)
        print(f"Cleared output directory: {cfg.output_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print_processing_details()
    lor_data = load_lors(cfg.path_to_lors)
    print(f"LORs loaded shape: ({lor_data.shape[0]:_}, {lor_data.shape[1]:_})")

    if cfg.filter_oob_lors:
        axial_half = cfg.axial_length / 2.0
        r1 = np.sqrt(lor_data[:, 0]**2 + lor_data[:, 1]**2)
        r2 = np.sqrt(lor_data[:, 3]**2 + lor_data[:, 4]**2)
        valid = (
            (r1 >= cfg.inner_radius) & (r1 <= cfg.outer_radius) &
            (r2 >= cfg.inner_radius) & (r2 <= cfg.outer_radius) &
            (np.abs(lor_data[:, 2]) <= axial_half) &
            (np.abs(lor_data[:, 5]) <= axial_half)
        )
        n_removed = int((~valid).sum())
        lor_data = lor_data[valid]
        print(f"OOB LOR filter: removed {n_removed:_} LORs ({100*n_removed/(n_removed+lor_data.shape[0]):.2f}%), "
              f"{lor_data.shape[0]:_} remaining")

    print("=" * 50 + "\n")
    print_scanner_details(cfg)
    print_recon_details(cfg)

    recon_image = None
    backprojection_image = None
    mu_map = None
    acf_measured = None
    scatter_estimate = None
    scatter_sino = None
    sino_params = None

    # ---- Attenuation correction setup ----
    if cfg.attenuation_correction:
        print("=" * 50)
        print("ATTENUATION CORRECTION")
        print("=" * 50)
        mu_map = load_mu_map(cfg.mu_map_path, cfg.image_grid_size)
        acf_measured = compute_acf(lor_data, mu_map, cfg.image_grid_size, cfg.voxel_size_in_mm)

    # ---- Sensitivity image ----
    if cfg.sensitivity_correction:
        # Backprojection of every detector pair, generated and traced on the GPU. With attenuation
        # correction every pair is weighted by its ACF (attenuation-weighted sensitivity).
        if cfg.attenuation_correction:
            print("Computing attenuation-weighted sensitivity (GPU)...")
        sensitivity_image = sensitivity_gpu(
            cfg.n_axial_rings, cfg.n_detectors_per_ring, cfg.inner_radius, cfg.axial_length,
            cfg.image_grid_size, cfg.voxel_size_in_mm,
            mode=cfg.sensitivity_mode, n_samples=cfg.sensitivity_samples,
            mu_map=mu_map if cfg.attenuation_correction else None,
        )

        if cfg.smooth_sensitivity:
            blur_sigma, blur_radius = cfg.sensitivity_blur_voxels()
            sensitivity_image = gaussian_filter(sensitivity_image, sigma=blur_sigma, radius=blur_radius)
    else:
        sensitivity_image = np.ones(cfg.image_grid_size)

    # ---- Scatter correction setup ----
    if cfg.scatter_correction and cfg.recon_algorithm == "OSEM":
        from petrecon.helpers.scatter import (
            identify_scatter_points, define_coarse_detectors,
            compute_scatter_sinogram, scatter_sinogram_to_lor,
        )

        if not cfg.attenuation_correction:
            print("WARNING: Scatter correction requires attenuation correction. Enabling AC.")
            mu_map = load_mu_map(cfg.mu_map_path, cfg.image_grid_size)
            acf_measured = compute_acf(lor_data, mu_map, cfg.image_grid_size, cfg.voxel_size_in_mm)

        scatter_points, scatter_ne, delta_V = identify_scatter_points(
            mu_map, cfg.image_grid_size, cfg.voxel_size_in_mm, coarse_factor=cfg.sss_coarse_factor,
        )
        coarse_detectors = define_coarse_detectors(
            n_angular=cfg.sss_n_angular_detectors, n_axial=cfg.sss_n_axial_detectors,
            radius=cfg.inner_radius, axial_length=cfg.axial_length,
        )

    # ---- Reconstruction ----
    # Backprojection is the first step for both algorithms
    if cfg.recon_algorithm in ["Unfiltered_backprojection", "OSEM"]:
        backprojection_image = backproject_gpu(
            lors=lor_data,
            grid_size=cfg.image_grid_size,
            voxel_size=cfg.voxel_size_in_mm,
            batch_size=1_000_000,
        )
        # Initialize recon_image with sensitivity-normalized backprojection
        recon_image = backprojection_image.copy()
        if cfg.sensitivity_correction:
            nonzero_mask = sensitivity_image > 0
            recon_image[nonzero_mask] /= sensitivity_image[nonzero_mask]
            recon_image[~nonzero_mask] = 0

    if cfg.recon_algorithm == "OSEM":

        if cfg.scatter_correction:
            # Iterative OSEM with SSS updates
            for osem_iter in range(cfg.osem_iterations):
                # Update scatter estimate at configured intervals
                if osem_iter >= cfg.sss_start_iteration and (osem_iter - cfg.sss_start_iteration) % cfg.sss_update_interval == 0:
                    print(f"\n--- Updating SSS scatter estimate (iteration {osem_iter + 1}) ---")
                    scatter_sino, sino_params = compute_scatter_sinogram(
                        mu_map=mu_map,
                        activity_image=recon_image,
                        grid_size=cfg.image_grid_size,
                        voxel_size=cfg.voxel_size_in_mm,
                        scatter_points=scatter_points,
                        scatter_ne=scatter_ne,
                        delta_V=delta_V,
                        detectors=coarse_detectors,
                        n_angular=cfg.sss_n_angular_detectors,
                        n_axial=cfg.sss_n_axial_detectors,
                        radius=cfg.inner_radius,
                        axial_length=cfg.axial_length,
                        measured_lors=lor_data,
                    )
                    scatter_estimate = scatter_sinogram_to_lor(scatter_sino, sino_params, lor_data)
                    scatter_estimate *= cfg.sss_scale_factor

                # Run one OSEM iteration
                recon_image = osem_gpu(
                    lors=lor_data,
                    grid_size=cfg.image_grid_size,
                    voxel_size=cfg.voxel_size_in_mm,
                    sensitivity_image=sensitivity_image,
                    init_image=recon_image,
                    n_iterations=1,
                    n_subsets=cfg.osem_subsets,
                    batch_size=1_000_000,
                    acf=acf_measured,
                    scatter=scatter_estimate,
                )
        else:
            # OSEM with attenuation correction only (or no corrections)
            recon_image = osem_gpu(
                lors=lor_data,
                grid_size=cfg.image_grid_size,
                voxel_size=cfg.voxel_size_in_mm,
                sensitivity_image=sensitivity_image,
                init_image=recon_image,
                n_iterations=cfg.osem_iterations,
                n_subsets=cfg.osem_subsets,
                batch_size=1_000_000,
                acf=acf_measured,
            )

    elif cfg.recon_algorithm == "None":
        recon_image = np.zeros(cfg.image_grid_size)

    np.save(os.path.join(cfg.output_dir, "reconstruction_image.npy"), recon_image)
    np.save(os.path.join(cfg.output_dir, "sensitivity_image.npy"), sensitivity_image)
    if backprojection_image is not None:
        np.save(os.path.join(cfg.output_dir, "backprojection_image.npy"), backprojection_image)
    if cfg.scatter_correction and scatter_estimate is not None and sino_params is not None:
        np.save(os.path.join(cfg.output_dir, "scatter_sino_raw.npy"), sino_params['scatter_sino_raw'])
        np.save(os.path.join(cfg.output_dir, "scatter_sino_fitted.npy"), scatter_sino)
        np.save(os.path.join(cfg.output_dir, "scatter_estimate_per_lor.npy"), scatter_estimate)
        np.savez(os.path.join(cfg.output_dir, "sino_params.npz"),
                 s_edges=sino_params['s_edges'], phi_edges=sino_params['phi_edges'],
                 z_avg_edges=sino_params['z_avg_edges'], z_diff_edges=sino_params['z_diff_edges'])
    save_config(cfg)

    # ---- Diagnostic outputs ----
    if cfg.scatter_correction and scatter_estimate is not None and scatter_sino is not None:
        # scatter_sino and sino_params persist from the last SSS update
        # Auto-detect object radius from mu-map (same logic as in scatter.py)
        nx, ny, nz = cfg.image_grid_size
        half = nx * cfg.voxel_size_in_mm / 2.0
        iy_g, ix_g = np.meshgrid(np.arange(ny), np.arange(nx))
        r_coords = np.sqrt((-half + (ix_g + 0.5) * cfg.voxel_size_in_mm)**2 +
                           (-half + (iy_g + 0.5) * cfg.voxel_size_in_mm)**2)
        mu_proj = mu_map.max(axis=2)
        obj_r = r_coords[mu_proj > 0.001].max() + cfg.voxel_size_in_mm if mu_map.max() > 0.001 else cfg.fov_radius

        plot_sss_diagnostics(
            scatter_sino=scatter_sino,
            sinogram_params=sino_params,
            scatter_per_lor=scatter_estimate,
            measured_lors=lor_data,
            object_radius=obj_r,
            output_dir=cfg.output_dir,
        )
    elif cfg.scatter_correction:
        print("SSS diagnostics skipped (no scatter sinogram computed).")

    return recon_image
