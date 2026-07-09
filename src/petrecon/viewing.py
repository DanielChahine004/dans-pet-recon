"""Visualization of PET reconstruction outputs (formerly recon_viewing.py)."""
import os
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter

from petrecon.helpers.plotting import (
    plot_interactive, plot_static, create_sinogram,
    plot_scanner_geometry, plot_line_profile_with_gaussian_fit,
)
from petrecon.helpers.data_io import load_lors


def view_reconstruction(output_dir, smooth=True, smooth_sigma=1.0, lors_path=None):
    """Display all outputs of a reconstruction run.

    Args:
        output_dir: Directory containing reconstruction_image.npy, recon_config.pkl, etc.
        smooth: Apply gaussian smoothing to the recon image for display.
        smooth_sigma: Sigma of the display smoothing.
        lors_path: Override the LOR file path recorded in the config pickle
            (useful when data has moved since the reconstruction ran).
    """
    print("Loading reconstruction outputs...")

    # Load config
    with open(os.path.join(output_dir, "recon_config.pkl"), "rb") as f:
        config = pickle.load(f)

    # Load main images
    recon_image = np.load(os.path.join(output_dir, "reconstruction_image.npy"))
    sensitivity_image = np.load(os.path.join(output_dir, "sensitivity_image.npy"))

    # Load backprojection if available
    backprojection_path = os.path.join(output_dir, "backprojection_image.npy")
    backprojection_image = np.load(backprojection_path) if os.path.exists(backprojection_path) else None

    # Load scatter outputs if available
    scatter_sino_raw = None
    scatter_sino_fitted = None
    scatter_estimate_per_lor = None
    if config.get("SCATTER_CORRECTION"):
        scatter_raw_path = os.path.join(output_dir, "scatter_sino_raw.npy")
        scatter_fitted_path = os.path.join(output_dir, "scatter_sino_fitted.npy")
        scatter_per_lor_path = os.path.join(output_dir, "scatter_estimate_per_lor.npy")

        if os.path.exists(scatter_raw_path):
            scatter_sino_raw = np.load(scatter_raw_path)
        if os.path.exists(scatter_fitted_path):
            scatter_sino_fitted = np.load(scatter_fitted_path)
        if os.path.exists(scatter_per_lor_path):
            scatter_estimate_per_lor = np.load(scatter_per_lor_path)

    # Load mu-map if attenuation correction was used
    mu_map = None
    if config.get("ATTENUATION_CORRECTION") and config.get("MU_MAP_PATH"):
        mu_map_path = config["MU_MAP_PATH"]
        if os.path.exists(mu_map_path):
            mu_map = np.load(mu_map_path)

    # Load LOR data
    lor_data = load_lors(lors_path or config["PATH_TO_LORS"])

    print(f"Loaded {len(lor_data):,} LORs")
    print(f"Reconstruction shape: {recon_image.shape}")

    # ---- Display configuration ----
    print("\n" + "=" * 60)
    print("RECONSTRUCTION CONFIGURATION")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # ---- Scanner geometry ----
    print("Plotting scanner geometry...")
    plot_scanner_geometry(
        config["N_AXIAL_RINGS"],
        config["N_DETECTORS_PER_RING"],
        config["INNER_RADIUS"],
        config["AXIAL_LENGTH"],
        config["IMAGE_GRID_SIZE"],
        config["VOXEL_SIZE_IN_MM"],
        fov_radius=config.get("FOV_RADIUS"),
        fov_axial_half_length=config.get("FOV_AXIAL_HALF_LENGTH"),
        lors=lor_data,
        max_lors_to_plot=1000,
    ).show()

    # ---- Sinogram (with scatter correction comparison if available) ----
    print("Creating sinograms...")
    n_radial_bins = 150
    n_angle_bins = 150
    n_axial_bins = 100

    # Prompt sinogram (includes scatter)
    fig_prompts, sino_prompts = create_sinogram(
        lor_data, n_radial_bins=n_radial_bins, n_angle_bins=n_angle_bins, n_axial_bins=n_axial_bins,
        title="Sinogram - Prompts (includes scatter)"
    )
    fig_prompts.show()

    # If scatter estimates are available, show scatter and corrected sinograms
    if scatter_estimate_per_lor is not None:
        print("Creating scatter and scatter-corrected sinograms...")

        # Scatter sinogram (weighted by scatter estimate per LOR)
        fig_scatter, sino_scatter = create_sinogram(
            lor_data, n_radial_bins=n_radial_bins, n_angle_bins=n_angle_bins, n_axial_bins=n_axial_bins,
            weights=scatter_estimate_per_lor,
            title="Sinogram - Scatter Estimate"
        )
        fig_scatter.show()

        # Scatter-corrected sinogram (prompts - scatter, clipped to >= 0)
        sino_corrected = np.maximum(sino_prompts - sino_scatter, 0)

        import plotly.express as px
        sino_corrected_for_plot = np.transpose(sino_corrected, (2, 1, 0))
        fig_corrected = px.imshow(
            sino_corrected_for_plot,
            animation_frame=0,
            color_continuous_scale='Hot',
            zmin=0,
            zmax=sino_corrected.max(),
            labels={'animation_frame': 'Axial Slice'},
        )
        fig_corrected.update_layout(title="Sinogram - Scatter Corrected (prompts - scatter)")
        fig_corrected.show()

    # ---- Backprojection image ----
    voxel_size = config["VOXEL_SIZE_IN_MM"]

    if backprojection_image is not None:
        print("Displaying backprojection image...")
        plot_interactive(backprojection_image, axis=2, vmin=0, title="Backprojection",
                         voxel_size=voxel_size).show()

    # ---- Sensitivity image ----
    print("Displaying sensitivity image...")
    sens_display = sensitivity_image.copy()
    if config.get("SMOOTH_SENSITIVITY"):
        sens_display = gaussian_filter(sens_display, sigma=config["GAUSSIAN_SIGMA"], radius=config["GAUSSIAN_RADIUS"])
    plot_interactive(sens_display, axis=2, vmin=0, title="Sensitivity", voxel_size=voxel_size).show()

    # ---- Mu-map (attenuation) ----
    if mu_map is not None:
        print("Displaying mu-map...")
        plot_interactive(mu_map, axis=2, vmin=0, title="Mu-map (Attenuation)", voxel_size=voxel_size).show()

    # ---- Reconstruction image ----
    print("Displaying reconstruction image...")
    recon_display = recon_image.copy()
    if smooth:
        recon_display = gaussian_filter(recon_display, sigma=smooth_sigma)

    plot_interactive(recon_display, axis=2, vmin=0, title="Reconstruction", voxel_size=voxel_size).show()

    # find max voxel in entire image and use its coordinates for line profile
    max_idx = np.unravel_index(np.argmax(recon_display), recon_display.shape)
    strongest_x, strongest_y, strongest_z = max_idx[0], max_idx[1], max_idx[2]

    plot_static(recon_display,
                title=f"Reconstruction (hottest voxel slice, z={(strongest_z - config['IMAGE_GRID_SIZE'][2]//2) * voxel_size} mm)",
                axis=2, depth=strongest_z, vmin=0, vmax=np.max(recon_display), voxel_size=voxel_size).show()

    # ---- Line profile analysis ----
    print("Analyzing line profile...")
    result = plot_line_profile_with_gaussian_fit(
        recon_display,
        axis='z',
        coord1=strongest_x,
        coord2=strongest_y,
        voxel_size=config["VOXEL_SIZE_IN_MM"]
    )
    if result.get('fwhm_mm'):
        print(f"FWHM: {result['fwhm_mm']:.2f} mm")
    if result.get('fwtm_mm'):
        print(f"FWTM: {result['fwtm_mm']:.2f} mm")

    # ---- SSS scatter sinograms (coarse resolution, if available) ----
    if scatter_sino_raw is not None or scatter_sino_fitted is not None:
        print("Displaying SSS scatter sinograms (coarse resolution)...")
        if scatter_sino_raw is not None:
            print(f"  Raw scatter sinogram shape: {scatter_sino_raw.shape}")
            # Sinograms are 4D (radial, angle, z_avg, z_diff) - sum over last axis for 3D visualization
            scatter_raw_3d = scatter_sino_raw.sum(axis=-1)
            plot_interactive(scatter_raw_3d, axis=2, vmin=0, title="SSS Scatter Raw (summed over z_diff)").show()

        if scatter_sino_fitted is not None:
            print(f"  Fitted scatter sinogram shape: {scatter_sino_fitted.shape}")
            scatter_fitted_3d = scatter_sino_fitted.sum(axis=-1)
            plot_interactive(scatter_fitted_3d, axis=2, vmin=0, title="SSS Scatter Fitted (summed over z_diff)").show()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Reconstruction algorithm: {config['RECON_ALGORITHM']}")
    print(f"Image size: {config['IMAGE_GRID_SIZE']}")
    print(f"Voxel size: {config['VOXEL_SIZE_IN_MM']} mm")
    print(f"Sensitivity correction: {config['SENSITIVITY_CORRECTION']}")
    print(f"Attenuation correction: {config['ATTENUATION_CORRECTION']}")
    print(f"Scatter correction: {config['SCATTER_CORRECTION']}")
    if config['RECON_ALGORITHM'] == 'OSEM':
        print(f"OSEM iterations: {config['OSEM_ITERATIONS']}")
        print(f"OSEM subsets: {config['OSEM_SUBSETS']}")
    print(f"Reconstruction range: [{recon_image.min():.4f}, {recon_image.max():.4f}]")
    print("=" * 60)

    print("Tip: run `napari /path/to/reconstruction_image.npy` for a dedicated interactive viewer.")
