"""Batch generate PDF plots for each point source reconstruction.

Iterates through Point_Sources_Processed subfolders, loads gt and pred
reconstruction outputs, and saves static slice + line profile PDFs.
Outputs raw and blurred variants into a structured output directory.
"""
import numpy as np
import pickle
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PDF saving
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(os.path.expanduser("~/Code/Opengate/ragy_files/Point_Sources_Processed"))
OUTPUT_DIR = Path(os.path.expanduser("~/Code/Opengate/ragy_files/Point_Sources_Processed/ubp_pdf_outputs"))
# BASE_DIR = Path(os.path.expanduser("~/Code/Opengate/ragy_files/Point_Sources_r235_to_257mm"))
# OUTPUT_DIR = Path(os.path.expanduser("~/Code/Opengate/ragy_files/Point_Sources_r235_to_257mm/pdf_outputs"))
# BASE_DIR = Path(os.path.expanduser("~/Code/Opengate/ragy_files/Point_Sources_polar_max50deg"))
# OUTPUT_DIR = Path(os.path.expanduser("~/Code/Opengate/ragy_files/Point_Sources_polar_max50deg/pdf_outputs"))
BLUR_SIGMA = 1
BLUR_RADIUS = 1
USE_BACKPROJECTION = True  # True → load backprojection_image.npy, False → reconstruction_image.npy


def count_lors(lor_path: str) -> tuple[int, int] | None:
    """Count total and valid (post-filter) LORs. Returns (total, valid) or None."""
    from petrecon.helpers.data_io import load_lors
    lor_path = Path(lor_path)
    if not lor_path.exists():
        return None
    lors = load_lors(str(lor_path))
    # load_lors prints counts; we need the raw count too
    if lor_path.suffix == '.csv':
        with open(lor_path) as f:
            total = sum(1 for _ in f) - 1
    else:
        data = np.load(lor_path)
        try:
            total = data.shape[0]
        except AttributeError:
            total = data['xyz1'].shape[0]
    return total, lors.shape[0]


def load_recon(output_dir: Path):
    """Load a reconstruction image and config. Returns (recon_image, config) or (None, None)."""
    config_path = output_dir / "recon_config.pkl"
    image_filename = "backprojection_image.npy" if USE_BACKPROJECTION else "reconstruction_image.npy"
    recon_path = output_dir / image_filename

    if not config_path.exists() or not recon_path.exists():
        return None, None

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    recon_image = np.load(recon_path)
    return recon_image, config


def _get_slice_and_extent(image, axis_idx, depth, voxel_size):
    """Extract a 2D slice and its extent in mm for imshow."""
    nx, ny, nz = image.shape
    half_x = nx * voxel_size / 2
    half_y = ny * voxel_size / 2
    half_z = nz * voxel_size / 2
    if axis_idx == 0:  # YZ plane (fixed x)
        slc = image[depth, :, :].T
        extent = [-half_y, half_y, -half_z, half_z]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
    elif axis_idx == 1:  # XZ plane (fixed y)
        slc = image[:, depth, :].T
        extent = [-half_x, half_x, -half_z, half_z]
        xlabel, ylabel = "X (mm)", "Z (mm)"
    else:  # XY plane (fixed z)
        slc = image[:, :, depth].T
        extent = [-half_x, half_x, -half_y, half_y]
        xlabel, ylabel = "X (mm)", "Y (mm)"
    return slc, extent, xlabel, ylabel


def _fit_line_profile(image, axis_name, coord1, coord2, voxel_size):
    """Run Gaussian fit on a line profile, return result dict (without creating a figure)."""
    from scipy.optimize import curve_fit

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_int = axis_map[axis_name]

    if axis_int == 0:
        line = image[:, coord1, coord2]
    elif axis_int == 1:
        line = image[coord1, :, coord2]
    else:
        line = image[coord1, coord2, :]

    n = len(line)
    x_data = np.arange(n)
    x_mm = (x_data - (n - 1) / 2) * voxel_size

    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset

    offset_guess = np.min(line)
    amplitude_guess = np.max(line) - offset_guess
    mean_guess = np.argmax(line)
    sigma_guess = n / 10
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    result = {'line': line, 'x_mm': x_mm, 'axis_label': axis_name.upper()}
    try:
        popt, _ = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
        amplitude, mean, sigma, offset = popt
        fwhm_mm = 2.355 * abs(sigma) * voxel_size
        fwtm_mm = 4.292 * abs(sigma) * voxel_size
        mean_mm = (mean - (n - 1) / 2) * voxel_size

        x_fit = np.linspace(0, n - 1, 500)
        x_fit_mm = (x_fit - (n - 1) / 2) * voxel_size
        y_fit = gaussian(x_fit, *popt)

        result.update({
            'fwhm_mm': fwhm_mm, 'fwtm_mm': fwtm_mm,
            'amplitude': amplitude, 'mean_mm': mean_mm, 'sigma': sigma, 'offset': offset,
            'x_fit_mm': x_fit_mm, 'y_fit': y_fit,
            'half_max': offset + amplitude / 2,
            'tenth_max': offset + amplitude / 10,
        })
    except RuntimeError:
        print(f"    Warning: Gaussian fit failed for {axis_name}-axis")
    return result


def save_sinogram(lor_path: str, label: str, save_dir: Path,
                  n_radial_bins=150, n_angle_bins=150, n_axial_bins=100):
    """Compute sinogram from LOR data and save axially-summed 2D sinogram as PDF."""
    from petrecon.helpers.data_io import load_lors

    lor_path = Path(lor_path)
    if not lor_path.exists():
        print(f"    Sinogram skipped: LOR file not found ({lor_path})")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    lors = load_lors(str(lor_path))

    x1, y1, z1 = lors[:, 0], lors[:, 1], lors[:, 2]
    x2, y2, z2 = lors[:, 3], lors[:, 4], lors[:, 5]

    dx = x2 - x1
    dy = y2 - y1
    lor_length_xy = np.sqrt(dx**2 + dy**2)

    # Signed perpendicular distance from origin to LOR
    s = (x1 * y2 - x2 * y1) / np.maximum(lor_length_xy, 1e-10)

    # Angle of LOR direction, normalized to [0, pi)
    phi = np.arctan2(dy, dx)
    flip_mask = phi < 0
    phi[flip_mask] += np.pi
    s[flip_mask] = -s[flip_mask]
    wrap_mask = phi >= np.pi
    phi[wrap_mask] -= np.pi
    s[wrap_mask] = -s[wrap_mask]

    z = (z1 + z2) / 2

    max_radius = np.max(np.abs(s)) * 1.01
    min_z, max_z = np.min(z), np.max(z)
    z_pad = (max_z - min_z) * 0.01 if max_z > min_z else 1.0

    radial_edges = np.linspace(-max_radius, max_radius, n_radial_bins + 1)
    angle_edges = np.linspace(0, np.pi, n_angle_bins + 1)
    axial_edges = np.linspace(min_z - z_pad, max_z + z_pad, n_axial_bins + 1)

    sinogram_3d, _ = np.histogramdd(
        np.column_stack([s, phi, z]),
        bins=[radial_edges, angle_edges, axial_edges],
    )

    # Sum over axial dimension → 2D sinogram (radial, angle)
    sino_2d = sinogram_3d.sum(axis=2).T  # (angle, radial) for imshow

    radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
    angle_centers = np.degrees((angle_edges[:-1] + angle_edges[1:]) / 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sino_2d, cmap='hot', origin='lower', aspect='auto',
              extent=[radial_centers[0], radial_centers[-1],
                      angle_centers[0], angle_centers[-1]],
              vmin=0, vmax=sino_2d.max())
    ax.set_xlabel("Radial distance s (mm)")
    ax.set_ylabel("Angle φ (degrees)")
    ax.set_title(f"{label} — Sinogram (summed over axial)")
    fig.tight_layout()

    pdf_path = save_dir / "sinogram.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"    Saved {pdf_path.relative_to(OUTPUT_DIR)}")


def _get_planes_and_overlays(grid_size, voxel_size, strongest_x, strongest_y, strongest_z):
    """Return plane definitions and overlay info for the 3 orthogonal slices."""
    sx_mm = (strongest_x - grid_size[0] // 2) * voxel_size
    sy_mm = (strongest_y - grid_size[1] // 2) * voxel_size
    sz_mm = (strongest_z - grid_size[2] // 2) * voxel_size

    planes = [
        {"name": "XY", "axis_idx": 2, "depth": strongest_z,
         "coord_str": f"z={sz_mm:.1f} mm"},
        {"name": "YZ", "axis_idx": 0, "depth": strongest_x,
         "coord_str": f"x={sx_mm:.1f} mm"},
        {"name": "XZ", "axis_idx": 1, "depth": strongest_y,
         "coord_str": f"y={sy_mm:.1f} mm"},
    ]

    profiles = [
        {"axis_name": "x", "coord1": strongest_y, "coord2": strongest_z},
        {"axis_name": "y", "coord1": strongest_x, "coord2": strongest_z},
        {"axis_name": "z", "coord1": strongest_x, "coord2": strongest_y},
    ]

    overlay_info = [
        [("x", sy_mm, "horizontal"), ("y", sx_mm, "vertical")],
        [("y", sz_mm, "horizontal"), ("z", sy_mm, "vertical")],
        [("x", sz_mm, "horizontal"), ("z", sx_mm, "vertical")],
    ]

    return planes, profiles, overlay_info


def _draw_slice(ax, fig, image, plane, overlay_info, voxel_size, vmax, profile_colors, title_prefix=""):
    """Draw a slice image with line overlays on the given axes."""
    slc, extent, xlabel, ylabel = _get_slice_and_extent(
        image, plane["axis_idx"], plane["depth"], voxel_size
    )
    im = ax.imshow(slc, cmap='hot', vmin=0, vmax=vmax, origin='lower', extent=extent)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    prefix = f"{title_prefix} " if title_prefix else ""
    ax.set_title(f"{prefix}{plane['name']} ({plane['coord_str']})")

    for axis_name, line_pos_mm, orientation in overlay_info:
        color = profile_colors[axis_name]
        if orientation == "horizontal":
            ax.axhline(y=line_pos_mm, color=color, linestyle='--',
                       alpha=0.5, linewidth=1.5, label=f"{axis_name.upper()}-profile")
        else:
            ax.axvline(x=line_pos_mm, color=color, linestyle='--',
                       alpha=0.5, linewidth=1.5, label=f"{axis_name.upper()}-profile")
    ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.0, -0.08),
              ncol=2, frameon=False)


def _normalise(values):
    """Normalise an array to [0, 1] using its min and max."""
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin == 0:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _draw_line_profile(ax, res, color):
    """Draw a single line profile with Gaussian fit on the given axes."""
    line_norm = _normalise(res['line'])
    ax.plot(res['x_mm'], line_norm, color=color, linewidth=2, label='Data')

    if 'x_fit_mm' in res:
        fit_norm = _normalise(res['y_fit'])
        ax.plot(res['x_fit_mm'], fit_norm, 'r--', linewidth=1.5,
                label=f"Fit (FWHM={res['fwhm_mm']:.2f} mm)")
        # Half-max line at 0.5 in normalised space
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        left_hm = res['mean_mm'] - res['fwhm_mm'] / 2
        right_hm = res['mean_mm'] + res['fwhm_mm'] / 2
        ax.axvline(x=left_hm, color='green', linestyle=':', alpha=0.5)
        ax.axvline(x=right_hm, color='green', linestyle=':', alpha=0.5)

    ax.set_xlabel(f"{res['axis_label']} position (mm)")
    ax.set_ylabel("Normalised intensity")
    ax.set_title(f"{res['axis_label']}-axis line profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def save_pdfs(recon_display, config, label: str, save_dir: Path,
              strongest_x: int, strongest_y: int, strongest_z: int):
    """Save a single 2x3 PDF for a single source (GT only)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    voxel_size = config["VOXEL_SIZE_IN_MM"]
    grid_size = config["IMAGE_GRID_SIZE"]
    vmax = np.max(recon_display)

    planes, profiles, overlay_info = _get_planes_and_overlays(
        grid_size, voxel_size, strongest_x, strongest_y, strongest_z
    )
    profile_colors = {"x": "cyan", "y": "magenta", "z": "lime"}

    # Compute line profile fits
    profile_results = {}
    for p in profiles:
        res = _fit_line_profile(recon_display, p["axis_name"], p["coord1"], p["coord2"], voxel_size)
        profile_results[p["axis_name"]] = res

    # Create 2x3 figure: slices on top, profiles on bottom
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, plane in enumerate(planes):
        _draw_slice(axes[0, col], fig, recon_display, plane, overlay_info[col],
                    voxel_size, vmax, profile_colors)

    for col, axis_name in enumerate(["x", "y", "z"]):
        _draw_line_profile(axes[1, col], profile_results[axis_name], profile_colors[axis_name])

    fig.suptitle(label, fontsize=14)
    fig.tight_layout()
    pdf_path = save_dir / "slices.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"    Saved {pdf_path.relative_to(OUTPUT_DIR)}")

    metrics = {}
    for axis_name in ["x", "y", "z"]:
        res = profile_results[axis_name]
        metrics[axis_name] = {"gt_fwhm": res.get('fwhm_mm'), "gt_fwtm": res.get('fwtm_mm')}
    res_z = profile_results["z"]
    if res_z.get('fwhm_mm'):
        print(f"      Z-axis FWHM: {res_z['fwhm_mm']:.2f} mm")
    return metrics


def save_comparison_pdfs(gt_image, pred_image, config, label: str, save_dir: Path,
                         strongest_x: int, strongest_y: int, strongest_z: int):
    """Save comparison PDFs: 2x3 slice comparison + 1x3 overlaid line profiles."""
    save_dir.mkdir(parents=True, exist_ok=True)
    voxel_size = config["VOXEL_SIZE_IN_MM"]
    grid_size = config["IMAGE_GRID_SIZE"]
    gt_vmax = np.max(gt_image)
    pred_vmax = np.max(pred_image)

    planes, profiles, overlay_info = _get_planes_and_overlays(
        grid_size, voxel_size, strongest_x, strongest_y, strongest_z
    )
    profile_colors = {"x": "cyan", "y": "magenta", "z": "lime"}

    # Compute line profile fits for both
    gt_profiles = {}
    pred_profiles = {}
    for p in profiles:
        gt_profiles[p["axis_name"]] = _fit_line_profile(
            gt_image, p["axis_name"], p["coord1"], p["coord2"], voxel_size)
        pred_profiles[p["axis_name"]] = _fit_line_profile(
            pred_image, p["axis_name"], p["coord1"], p["coord2"], voxel_size)

    # --- Figure 1: Slice comparison (2x3) ---
    # Top row: GT slices, Bottom row: Pred slices (independent color scales)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, plane in enumerate(planes):
        _draw_slice(axes[0, col], fig, gt_image, plane, overlay_info[col],
                    voxel_size, gt_vmax, profile_colors, title_prefix="GT")
        _draw_slice(axes[1, col], fig, pred_image, plane, overlay_info[col],
                    voxel_size, pred_vmax, profile_colors, title_prefix="Pred")

    fig.suptitle(f"{label} — Slice comparison", fontsize=14)
    fig.tight_layout()
    pdf_path = save_dir / "slices.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"    Saved {pdf_path.relative_to(OUTPUT_DIR)}")

    # --- Figure 2: Overlaid line profiles (3x1, stacked vertically) ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 15))

    for col, axis_name in enumerate(["x", "y", "z"]):
        ax = axes[col]
        gt_res = gt_profiles[axis_name]
        pred_res = pred_profiles[axis_name]

        gt_line_norm = _normalise(gt_res['line'])
        pred_line_norm = _normalise(pred_res['line'])
        ax.plot(gt_res['x_mm'], gt_line_norm, color='tab:blue', linewidth=2, label='GT data')
        ax.plot(pred_res['x_mm'], pred_line_norm, color='tab:orange', linewidth=2, label='Pred data')

        if 'x_fit_mm' in gt_res:
            gt_fit_norm = _normalise(gt_res['y_fit'])
            ax.plot(gt_res['x_fit_mm'], gt_fit_norm, 'b--', linewidth=1.5,
                    label=f"GT fit (FWHM={gt_res['fwhm_mm']:.2f} mm)")
        if 'x_fit_mm' in pred_res:
            pred_fit_norm = _normalise(pred_res['y_fit'])
            ax.plot(pred_res['x_fit_mm'], pred_fit_norm, '--', color='tab:orange',
                    linewidth=1.5, alpha=0.8,
                    label=f"Pred fit (FWHM={pred_res['fwhm_mm']:.2f} mm)")

        ax.set_xlabel(f"{gt_res['axis_label']} position (mm)")
        ax.set_ylabel("Normalised intensity")
        ax.set_title(f"{gt_res['axis_label']}-axis line profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Line profile comparison", fontsize=14)
    fig.tight_layout()
    pdf_path = save_dir / "profiles.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"    Saved {pdf_path.relative_to(OUTPUT_DIR)}")

    # Collect and print FWHM comparison (only z-axis metrics printed)
    metrics = {}
    for axis_name in ["x", "y", "z"]:
        gt_fwhm = gt_profiles[axis_name].get('fwhm_mm')
        gt_fwtm = gt_profiles[axis_name].get('fwtm_mm')
        pred_fwhm = pred_profiles[axis_name].get('fwhm_mm')
        pred_fwtm = pred_profiles[axis_name].get('fwtm_mm')
        metrics[axis_name] = {
            "gt_fwhm": gt_fwhm, "gt_fwtm": gt_fwtm,
            "pred_fwhm": pred_fwhm, "pred_fwtm": pred_fwtm,
        }
    gt_str = f"{metrics['z']['gt_fwhm']:.2f}" if metrics['z']['gt_fwhm'] else "N/A"
    pred_str = f"{metrics['z']['pred_fwhm']:.2f}" if metrics['z']['pred_fwhm'] else "N/A"
    print(f"      Z-axis FWHM: GT={gt_str} mm, Pred={pred_str} mm")
    return metrics


def save_fwhm_trend_plots(positions: list[dict], metrics_key: str = "raw"):
    """Generate FWHM vs radial distance and FWHM vs axial position plots.

    Args:
        positions: list of position dicts with parsed coordinates.
        metrics_key: which metrics variant to plot ("raw" or "blurred").
    """
    any_pred = any(p["has_pred"] for p in positions)
    suffix = f"_{metrics_key}" if metrics_key != "raw" else ""
    title_suffix = f" ({metrics_key})" if metrics_key != "raw" else ""

    # Parse coordinates from position names (idempotent)
    for pos in positions:
        if "x_mm" not in pos:
            parts = pos["name"].split("_")
            pos["x_mm"] = int(parts[0])
            pos["y_mm"] = int(parts[1])
            pos["z_mm"] = int(parts[2])
            pos["radial_mm"] = np.sqrt(pos["x_mm"]**2 + pos["y_mm"]**2)

    # --- Figure 1: Z-axis FWHM vs radial distance, grouped by z ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Group positions by z
    z_values = sorted(set(p["z_mm"] for p in positions))
    z_colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(z_values), 1)))

    for z_idx, z_val in enumerate(z_values):
        pts = sorted([p for p in positions if p["z_mm"] == z_val],
                     key=lambda p: p["radial_mm"])
        radials = [p["radial_mm"] for p in pts]
        gt_fwhms = [p["metrics"].get(metrics_key, {}).get("z", {}).get("gt_fwhm") for p in pts]

        valid = [(r, f) for r, f in zip(radials, gt_fwhms) if f is not None]
        if valid:
            r_vals, f_vals = zip(*valid)
            ax.plot(r_vals, f_vals, 'o-', color=z_colors[z_idx],
                    label=f"GT z={z_val} mm", markersize=6)

        if any_pred:
            pred_fwhms = [p["metrics"].get(metrics_key, {}).get("z", {}).get("pred_fwhm") for p in pts]
            valid_p = [(r, f) for r, f in zip(radials, pred_fwhms) if f is not None]
            if valid_p:
                r_vals, f_vals = zip(*valid_p)
                ax.plot(r_vals, f_vals, 's--', color=z_colors[z_idx],
                        alpha=0.7, markersize=5, label=f"Pred z={z_val} mm")

    ax.set_xlabel("Radial distance (mm)")
    ax.set_ylabel("FWHM (mm)")
    ax.set_title("Z-axis FWHM")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"FWHM vs radial distance{title_suffix}", fontsize=14)
    fig.tight_layout()
    pdf_path = OUTPUT_DIR / f"fwhm_vs_radial{suffix}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"\nSaved {pdf_path.relative_to(OUTPUT_DIR)}")

    # --- Figure 2: Z-axis FWHM vs axial position (z), grouped by radial distance ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Group by (x, y) → radial distance
    radial_groups = {}
    for p in positions:
        key = (p["x_mm"], p["y_mm"])
        radial_groups.setdefault(key, []).append(p)
    radial_keys = sorted(radial_groups.keys(), key=lambda k: np.sqrt(k[0]**2 + k[1]**2))
    r_colors = plt.cm.plasma(np.linspace(0, 0.9, max(len(radial_keys), 1)))

    for r_idx, key in enumerate(radial_keys):
        pts = sorted(radial_groups[key], key=lambda p: p["z_mm"])
        r_dist = np.sqrt(key[0]**2 + key[1]**2)
        z_vals = [p["z_mm"] for p in pts]
        gt_fwhms = [p["metrics"].get(metrics_key, {}).get("z", {}).get("gt_fwhm") for p in pts]

        valid = [(z, f) for z, f in zip(z_vals, gt_fwhms) if f is not None]
        if valid:
            z_v, f_v = zip(*valid)
            ax.plot(z_v, f_v, 'o-', color=r_colors[r_idx],
                    label=f"GT r={r_dist:.0f} mm ({key[0]},{key[1]})", markersize=6)

        if any_pred:
            pred_fwhms = [p["metrics"].get(metrics_key, {}).get("z", {}).get("pred_fwhm") for p in pts]
            valid_p = [(z, f) for z, f in zip(z_vals, pred_fwhms) if f is not None]
            if valid_p:
                z_v, f_v = zip(*valid_p)
                ax.plot(z_v, f_v, 's--', color=r_colors[r_idx],
                        alpha=0.7, markersize=5,
                        label=f"Pred r={r_dist:.0f} mm ({key[0]},{key[1]})")

    ax.set_xlabel("Axial position z (mm)")
    ax.set_ylabel("FWHM (mm)")
    ax.set_title("Z-axis FWHM")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"FWHM vs axial position{title_suffix}", fontsize=14)
    fig.tight_layout()
    pdf_path = OUTPUT_DIR / f"fwhm_vs_axial{suffix}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"Saved {pdf_path.relative_to(OUTPUT_DIR)}")


def save_intensity_trend_plots(positions: list[dict], metrics_key: str = "raw",
                               unnormalised: bool = False):
    """Generate peak intensity vs radial distance and vs axial position plots.

    Args:
        positions: list of position dicts with parsed coordinates and peak intensities.
        metrics_key: which variant to plot ("raw" or "blurred").
        unnormalised: if True, use raw peak values (not divided by LOR count).
    """
    any_pred = any(p["has_pred"] for p in positions)
    suffix = f"_{metrics_key}" if metrics_key != "raw" else ""
    title_suffix = f" ({metrics_key})" if metrics_key != "raw" else ""
    if unnormalised:
        gt_key = f"gt_peak_unnorm_{metrics_key}"
        pred_key = f"pred_peak_unnorm_{metrics_key}"
        suffix += "_unnorm"
        title_suffix += " unnormalised"
        ylabel = "Peak intensity (raw)"
    else:
        gt_key = f"gt_peak_{metrics_key}"
        pred_key = f"pred_peak_{metrics_key}"
        ylabel = "Peak intensity / valid LORs"

    # Ensure coordinates are parsed
    for pos in positions:
        if "x_mm" not in pos:
            parts = pos["name"].split("_")
            pos["x_mm"] = int(parts[0])
            pos["y_mm"] = int(parts[1])
            pos["z_mm"] = int(parts[2])
            pos["radial_mm"] = np.sqrt(pos["x_mm"]**2 + pos["y_mm"]**2)

    # --- Figure 1: Peak intensity vs radial distance, grouped by z ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    z_values = sorted(set(p["z_mm"] for p in positions))
    z_colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(z_values), 1)))

    for z_idx, z_val in enumerate(z_values):
        pts = sorted([p for p in positions if p["z_mm"] == z_val],
                     key=lambda p: p["radial_mm"])
        radials = [p["radial_mm"] for p in pts]
        gt_peaks = [p.get(gt_key) for p in pts]

        valid = [(r, v) for r, v in zip(radials, gt_peaks) if v is not None]
        if valid:
            r_vals, v_vals = zip(*valid)
            ax.plot(r_vals, v_vals, 'o-', color=z_colors[z_idx],
                    label=f"GT z={z_val} mm", markersize=6)

        if any_pred:
            pred_peaks = [p.get(pred_key) for p in pts]
            valid_p = [(r, v) for r, v in zip(radials, pred_peaks) if v is not None]
            if valid_p:
                r_vals, v_vals = zip(*valid_p)
                ax.plot(r_vals, v_vals, 's--', color=z_colors[z_idx],
                        alpha=0.7, markersize=5, label=f"Pred z={z_val} mm")

    ax.set_xlabel("Radial distance (mm)")
    ax.set_ylabel(ylabel)
    ax.set_title("Peak voxel intensity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Peak intensity vs radial distance{title_suffix}", fontsize=14)
    fig.tight_layout()
    pdf_path = OUTPUT_DIR / f"intensity_vs_radial{suffix}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"\nSaved {pdf_path.relative_to(OUTPUT_DIR)}")

    # --- Figure 2: Peak intensity vs axial position (z), grouped by radial distance ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    radial_groups = {}
    for p in positions:
        key = (p["x_mm"], p["y_mm"])
        radial_groups.setdefault(key, []).append(p)
    radial_keys = sorted(radial_groups.keys(), key=lambda k: np.sqrt(k[0]**2 + k[1]**2))
    r_colors = plt.cm.plasma(np.linspace(0, 0.9, max(len(radial_keys), 1)))

    for r_idx, key in enumerate(radial_keys):
        pts = sorted(radial_groups[key], key=lambda p: p["z_mm"])
        r_dist = np.sqrt(key[0]**2 + key[1]**2)
        z_vals = [p["z_mm"] for p in pts]
        gt_peaks = [p.get(gt_key) for p in pts]

        valid = [(z, v) for z, v in zip(z_vals, gt_peaks) if v is not None]
        if valid:
            z_v, v_v = zip(*valid)
            ax.plot(z_v, v_v, 'o-', color=r_colors[r_idx],
                    label=f"GT r={r_dist:.0f} mm ({key[0]},{key[1]})", markersize=6)

        if any_pred:
            pred_peaks = [p.get(pred_key) for p in pts]
            valid_p = [(z, v) for z, v in zip(z_vals, pred_peaks) if v is not None]
            if valid_p:
                z_v, v_v = zip(*valid_p)
                ax.plot(z_v, v_v, 's--', color=r_colors[r_idx],
                        alpha=0.7, markersize=5,
                        label=f"Pred r={r_dist:.0f} mm ({key[0]},{key[1]})")

    ax.set_xlabel("Axial position z (mm)")
    ax.set_ylabel(ylabel)
    ax.set_title("Peak voxel intensity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Peak intensity vs axial position{title_suffix}", fontsize=14)
    fig.tight_layout()
    pdf_path = OUTPUT_DIR / f"intensity_vs_axial{suffix}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"Saved {pdf_path.relative_to(OUTPUT_DIR)}")


def save_lor_count_plots(positions: list[dict]):
    """Generate valid LOR count vs radial distance and vs axial position plots."""
    any_pred = any(p["has_pred"] for p in positions)

    # Ensure coordinates are parsed
    for pos in positions:
        if "x_mm" not in pos:
            parts = pos["name"].split("_")
            pos["x_mm"] = int(parts[0])
            pos["y_mm"] = int(parts[1])
            pos["z_mm"] = int(parts[2])
            pos["radial_mm"] = np.sqrt(pos["x_mm"]**2 + pos["y_mm"]**2)

    # --- Figure 1: Valid LOR count vs radial distance, grouped by z ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    z_values = sorted(set(p["z_mm"] for p in positions))
    z_colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(z_values), 1)))

    for z_idx, z_val in enumerate(z_values):
        pts = sorted([p for p in positions if p["z_mm"] == z_val],
                     key=lambda p: p["radial_mm"])
        radials = [p["radial_mm"] for p in pts]
        gt_counts = [p["gt_lor_counts"][1] if p["gt_lor_counts"] else None for p in pts]

        valid = [(r, c) for r, c in zip(radials, gt_counts) if c is not None]
        if valid:
            r_vals, c_vals = zip(*valid)
            ax.plot(r_vals, c_vals, 'o-', color=z_colors[z_idx],
                    label=f"GT z={z_val} mm", markersize=6)

        if any_pred:
            pred_counts = [p["pred_lor_counts"][1] if p["pred_lor_counts"] else None for p in pts]
            valid_p = [(r, c) for r, c in zip(radials, pred_counts) if c is not None]
            if valid_p:
                r_vals, c_vals = zip(*valid_p)
                ax.plot(r_vals, c_vals, 's--', color=z_colors[z_idx],
                        alpha=0.7, markersize=5, label=f"Pred z={z_val} mm")

    ax.set_xlabel("Radial distance (mm)")
    ax.set_ylabel("Valid LOR count")
    ax.set_title("Valid LORs (post-filter)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Valid LOR count vs radial distance", fontsize=14)
    fig.tight_layout()
    pdf_path = OUTPUT_DIR / "lor_count_vs_radial.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"\nSaved {pdf_path.relative_to(OUTPUT_DIR)}")

    # --- Figure 2: Valid LOR count vs axial position, grouped by radial distance ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    radial_groups = {}
    for p in positions:
        key = (p["x_mm"], p["y_mm"])
        radial_groups.setdefault(key, []).append(p)
    radial_keys = sorted(radial_groups.keys(), key=lambda k: np.sqrt(k[0]**2 + k[1]**2))
    r_colors = plt.cm.plasma(np.linspace(0, 0.9, max(len(radial_keys), 1)))

    for r_idx, key in enumerate(radial_keys):
        pts = sorted(radial_groups[key], key=lambda p: p["z_mm"])
        r_dist = np.sqrt(key[0]**2 + key[1]**2)
        z_vals = [p["z_mm"] for p in pts]
        gt_counts = [p["gt_lor_counts"][1] if p["gt_lor_counts"] else None for p in pts]

        valid = [(z, c) for z, c in zip(z_vals, gt_counts) if c is not None]
        if valid:
            z_v, c_v = zip(*valid)
            ax.plot(z_v, c_v, 'o-', color=r_colors[r_idx],
                    label=f"GT r={r_dist:.0f} mm ({key[0]},{key[1]})", markersize=6)

        if any_pred:
            pred_counts = [p["pred_lor_counts"][1] if p["pred_lor_counts"] else None for p in pts]
            valid_p = [(z, c) for z, c in zip(z_vals, pred_counts) if c is not None]
            if valid_p:
                z_v, c_v = zip(*valid_p)
                ax.plot(z_v, c_v, 's--', color=r_colors[r_idx],
                        alpha=0.7, markersize=5,
                        label=f"Pred r={r_dist:.0f} mm ({key[0]},{key[1]})")

    ax.set_xlabel("Axial position z (mm)")
    ax.set_ylabel("Valid LOR count")
    ax.set_title("Valid LORs (post-filter)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Valid LOR count vs axial position", fontsize=14)
    fig.tight_layout()
    pdf_path = OUTPUT_DIR / "lor_count_vs_axial.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close("all")
    print(f"Saved {pdf_path.relative_to(OUTPUT_DIR)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_positions = []

    # Iterate through all position folders (skip outside_fov), sorted numerically
    def _numeric_sort_key(p):
        try:
            return tuple(int(x) for x in p.name.split("_"))
        except ValueError:
            return (float('inf'),)

    for pos_dir in sorted(BASE_DIR.iterdir(), key=_numeric_sort_key):
        if not pos_dir.is_dir() or pos_dir.name in ("outside_fov", "pdf_outputs"):
            continue

        pos_name = pos_dir.name  # e.g. "0_0_0"
        print(f"\nProcessing {pos_name}...")

        # Load GT first to get the hottest voxel coordinates
        gt_raw, gt_config = load_recon(pos_dir / "gt")
        if gt_raw is None:
            print(f"  Skipping {pos_name}: missing GT reconstruction")
            continue

        # Find hottest voxel from raw GT image
        max_idx = np.unravel_index(np.argmax(gt_raw), gt_raw.shape)
        strongest_x, strongest_y, strongest_z = max_idx[0], max_idx[1], max_idx[2]
        print(f"  GT hottest voxel: ({strongest_x}, {strongest_y}, {strongest_z})")

        # Folder name is the source position in mm (e.g. "0_0_0" = center)
        coords_mm = pos_name.replace("_", ", ")

        gt_blurred = gaussian_filter(gt_raw.copy(), sigma=BLUR_SIGMA, radius=BLUR_RADIUS)

        # Load pred
        pred_raw, pred_config = load_recon(pos_dir / "pred")
        pred_blurred = None
        if pred_raw is not None:
            pred_blurred = gaussian_filter(pred_raw.copy(), sigma=BLUR_SIGMA, radius=BLUR_RADIUS)

        # Count LORs
        gt_lor_counts = count_lors(gt_config["PATH_TO_LORS"])
        pred_lor_counts = count_lors(pred_config["PATH_TO_LORS"]) if pred_config else None

        # Save sinograms (one per source, independent of blur)
        print(f"  gt sinogram:")
        save_sinogram(gt_config["PATH_TO_LORS"], f"{pos_name} gt",
                      OUTPUT_DIR / pos_name / "gt")
        if pred_config is not None:
            print(f"  pred sinogram:")
            save_sinogram(pred_config["PATH_TO_LORS"], f"{pos_name} pred",
                          OUTPUT_DIR / pos_name / "pred")

        # Generate comparison or GT-only PDFs for raw and blurred variants
        pos_metrics = {}
        for blur_label, gt_img, pred_img in [("raw", gt_raw, pred_raw),
                                              ("blurred", gt_blurred, pred_blurred)]:
            out_dir = OUTPUT_DIR / pos_name / blur_label
            if pred_img is not None:
                label = f"{pos_name} ({blur_label})"
                print(f"  comparison/{blur_label}:")
                metrics = save_comparison_pdfs(gt_img, pred_img, gt_config, label, out_dir,
                                               strongest_x, strongest_y, strongest_z)
            else:
                label = f"{pos_name} GT ({blur_label})"
                print(f"  gt-only/{blur_label}:")
                metrics = save_pdfs(gt_img, gt_config, label, out_dir,
                                    strongest_x, strongest_y, strongest_z)
            pos_metrics[blur_label] = metrics

        if pred_raw is None:
            print(f"  Skipping {pos_name} pred: missing reconstruction")

        processed_positions.append({
            "name": pos_name,
            "coords_mm": coords_mm,
            "has_pred": pred_raw is not None,
            "metrics": pos_metrics,
            "gt_lor_counts": gt_lor_counts,
            "pred_lor_counts": pred_lor_counts,
            "gt_peak_unnorm_raw": float(np.max(gt_raw)),
            "gt_peak_unnorm_blurred": float(np.max(gt_blurred)),
            "pred_peak_unnorm_raw": float(np.max(pred_raw)) if pred_raw is not None else None,
            "pred_peak_unnorm_blurred": float(np.max(pred_blurred)) if pred_blurred is not None else None,
        })

    # Generate FWHM trend plots for raw and blurred, then LaTeX
    save_fwhm_trend_plots(processed_positions, metrics_key="raw")
    save_fwhm_trend_plots(processed_positions, metrics_key="blurred")
    save_intensity_trend_plots(processed_positions, metrics_key="raw", unnormalised=True)
    save_intensity_trend_plots(processed_positions, metrics_key="blurred", unnormalised=True)
    save_lor_count_plots(processed_positions)
    generate_latex(processed_positions)


def generate_latex(positions: list[dict]):
    """Generate a .tex file that can be \\input{} into the main document.

    Args:
        positions: list of dicts with keys 'name', 'coords_mm', 'has_pred', 'metrics'.
    """
    # Path prefix for \includegraphics — adjust to match your LaTeX project structure
    FIG_PREFIX = "figures/Chapter 5"

    lines = []
    lines.append(r"\section{Point source reconstruction results}")
    lines.append(r"\label{sec:point-source-results}")
    lines.append("")
    lines.append(
        r"Point source reconstructions were performed at several positions "
        r"within the scanner field of view to characterise spatial resolution "
        r"as a function of position. For each source location, reconstructions "
        r"were performed using both the ground truth (GT) LOR endpoints from "
        r"simulation and the machine learning model predictions. The resulting "
        r"images are compared side-by-side, with line profiles and Gaussian "
        r"fits providing quantitative FWHM measurements along each axis."
    )
    lines.append("")

    for pos in positions:
        name = pos["name"]
        coords = pos["coords_mm"]
        has_pred = pos["has_pred"]
        label_safe = name.replace("_", "-")

        lines.append(f"\\subsection{{Source at ({coords}) mm}}")
        lines.append(f"\\label{{sec:ps-{label_safe}}}")
        lines.append("")

        # === Sinograms ===
        if has_pred:
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"    \centering")
            lines.append(f"    \\subfigure[Ground truth]{{\\includegraphics[width=0.48\\textwidth]{{{FIG_PREFIX}/{name}/gt/sinogram.pdf}}}}")
            lines.append(r"    \hfill")
            lines.append(f"    \\subfigure[Prediction]{{\\includegraphics[width=0.48\\textwidth]{{{FIG_PREFIX}/{name}/pred/sinogram.pdf}}}}")
            lines.append(f"    \\caption{{Sinograms (axially summed) for source at ({coords})\\,mm.}}")
            lines.append(f"    \\label{{fig:ps-{label_safe}-sino}}")
            lines.append(r"\end{figure}")
        else:
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"    \centering")
            lines.append(f"    \\includegraphics[width=0.6\\textwidth]{{{FIG_PREFIX}/{name}/gt/sinogram.pdf}}")
            lines.append(f"    \\caption{{Sinogram (axially summed) for source at ({coords})\\,mm (GT only).}}")
            lines.append(f"    \\label{{fig:ps-{label_safe}-sino}}")
            lines.append(r"\end{figure}")
        lines.append("")

        # === Slice comparison (2x3: GT top, Pred bottom) ===
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"    \centering")
        lines.append(f"    \\includegraphics[width=0.95\\textwidth]{{{FIG_PREFIX}/{name}/blurred/slices.pdf}}")
        if has_pred:
            lines.append(f"    \\caption{{Reconstruction slices at ({coords})\\,mm (blurred). Top: GT, Bottom: Prediction.}}")
        else:
            lines.append(f"    \\caption{{Reconstruction slices at ({coords})\\,mm (blurred, GT only).}}")
        lines.append(f"    \\label{{fig:ps-{label_safe}-slices}}")
        lines.append(r"\end{figure}")
        lines.append("")

        # === Line profile comparison (1x3, overlaid) ===
        if has_pred:
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"    \centering")
            lines.append(f"    \\includegraphics[width=0.7\\textwidth]{{{FIG_PREFIX}/{name}/blurred/profiles.pdf}}")
            lines.append(f"    \\caption{{Line profile comparison at ({coords})\\,mm. GT (blue) vs Prediction (orange).}}")
            lines.append(f"    \\label{{fig:ps-{label_safe}-profiles}}")
            lines.append(r"\end{figure}")
            lines.append("")

        lines.append(r"\clearpage")
        lines.append("")

    # === LOR count table ===
    any_pred = any(p["has_pred"] for p in positions)

    lines.append(r"\section{Summary of spatial resolution measurements}")
    lines.append(r"\label{sec:ps-summary}")
    lines.append("")

    if any_pred:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Number of LORs used for each point source reconstruction.}")
        lines.append(r"\label{tab:ps-lor-counts}")
        lines.append(r"\begin{tabular}{l r r r r}")
        lines.append(r"\hline")
        lines.append(r"Position (mm) & \multicolumn{2}{c}{GT} & \multicolumn{2}{c}{Pred} \\")
        lines.append(r" & Total & Valid & Total & Valid \\")
        lines.append(r"\hline")
        for pos in positions:
            gt_c = pos.get("gt_lor_counts")
            pred_c = pos.get("pred_lor_counts")
            gt_total = f"{gt_c[0]:,}" if gt_c else "---"
            gt_valid = f"{gt_c[1]:,}" if gt_c else "---"
            pred_total = f"{pred_c[0]:,}" if pred_c else "---"
            pred_valid = f"{pred_c[1]:,}" if pred_c else "---"
            lines.append(f"({pos['coords_mm']}) & {gt_total} & {gt_valid} & {pred_total} & {pred_valid} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    else:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Number of LORs used for each point source reconstruction (GT only).}")
        lines.append(r"\label{tab:ps-lor-counts}")
        lines.append(r"\begin{tabular}{l r r}")
        lines.append(r"\hline")
        lines.append(r"Position (mm) & Total & Valid \\")
        lines.append(r"\hline")
        for pos in positions:
            gt_c = pos.get("gt_lor_counts")
            gt_total = f"{gt_c[0]:,}" if gt_c else "---"
            gt_valid = f"{gt_c[1]:,}" if gt_c else "---"
            lines.append(f"({pos['coords_mm']}) & {gt_total} & {gt_valid} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    lines.append("")

    # === FWHM trend plots ===
    lines.append("")

    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\includegraphics[width=0.95\\textwidth]{{{FIG_PREFIX}/fwhm_vs_radial.pdf}}")
    lines.append(r"    \caption{FWHM as a function of radial distance from the scanner axis, grouped by axial position (raw).}")
    lines.append(r"    \label{fig:fwhm-vs-radial}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\includegraphics[width=0.95\\textwidth]{{{FIG_PREFIX}/fwhm_vs_axial.pdf}}")
    lines.append(r"    \caption{FWHM as a function of axial position, grouped by radial distance (raw).}")
    lines.append(r"    \label{fig:fwhm-vs-axial}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\includegraphics[width=0.95\\textwidth]{{{FIG_PREFIX}/fwhm_vs_radial_blurred.pdf}}")
    lines.append(r"    \caption{FWHM as a function of radial distance from the scanner axis, grouped by axial position (blurred).}")
    lines.append(r"    \label{fig:fwhm-vs-radial-blurred}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\includegraphics[width=0.95\\textwidth]{{{FIG_PREFIX}/fwhm_vs_axial_blurred.pdf}}")
    lines.append(r"    \caption{FWHM as a function of axial position, grouped by radial distance (blurred).}")
    lines.append(r"    \label{fig:fwhm-vs-axial-blurred}")
    lines.append(r"\end{figure}")
    lines.append("")

    # === Summary table ===
    if any_pred:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Z-axis FWHM and FWTM measurements for all point source positions (raw).}")
        lines.append(r"\label{tab:ps-fwhm-summary}")
        lines.append(r"\begin{tabular}{l c c c c}")
        lines.append(r"\hline")
        lines.append(r" & \multicolumn{2}{c}{\textbf{FWHM (mm)}} & \multicolumn{2}{c}{\textbf{FWTM (mm)}} \\")
        lines.append(r"Position (mm) & GT & Pred & GT & Pred \\")
        lines.append(r"\hline")
        for pos in positions:
            am = pos["metrics"].get("raw", {}).get("z", {})
            gt_fwhm = f"{am['gt_fwhm']:.2f}" if am.get('gt_fwhm') else "---"
            pred_fwhm = f"{am['pred_fwhm']:.2f}" if am.get('pred_fwhm') else "---"
            gt_fwtm = f"{am['gt_fwtm']:.2f}" if am.get('gt_fwtm') else "---"
            pred_fwtm = f"{am['pred_fwtm']:.2f}" if am.get('pred_fwtm') else "---"
            lines.append(f"({pos['coords_mm']}) & {gt_fwhm} & {pred_fwhm} & {gt_fwtm} & {pred_fwtm} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    else:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Z-axis FWHM and FWTM measurements for all point source positions (raw, GT only).}")
        lines.append(r"\label{tab:ps-fwhm-summary}")
        lines.append(r"\begin{tabular}{l c c}")
        lines.append(r"\hline")
        lines.append(r"Position (mm) & \textbf{FWHM (mm)} & \textbf{FWTM (mm)} \\")
        lines.append(r"\hline")
        for pos in positions:
            am = pos["metrics"].get("raw", {}).get("z", {})
            gt_fwhm = f"{am['gt_fwhm']:.2f}" if am.get('gt_fwhm') else "---"
            gt_fwtm = f"{am['gt_fwtm']:.2f}" if am.get('gt_fwtm') else "---"
            lines.append(f"({pos['coords_mm']}) & {gt_fwhm} & {gt_fwtm} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    # === Blurred summary table ===
    lines.append("")
    if any_pred:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Z-axis FWHM and FWTM measurements for all point source positions (blurred, $\sigma=" + str(BLUR_SIGMA) + r"$, radius=" + str(BLUR_RADIUS) + r").}")
        lines.append(r"\label{tab:ps-fwhm-summary-blurred}")
        lines.append(r"\begin{tabular}{l c c c c}")
        lines.append(r"\hline")
        lines.append(r" & \multicolumn{2}{c}{\textbf{FWHM (mm)}} & \multicolumn{2}{c}{\textbf{FWTM (mm)}} \\")
        lines.append(r"Position (mm) & GT & Pred & GT & Pred \\")
        lines.append(r"\hline")
        for pos in positions:
            am = pos["metrics"].get("blurred", {}).get("z", {})
            gt_fwhm = f"{am['gt_fwhm']:.2f}" if am.get('gt_fwhm') else "---"
            pred_fwhm = f"{am['pred_fwhm']:.2f}" if am.get('pred_fwhm') else "---"
            gt_fwtm = f"{am['gt_fwtm']:.2f}" if am.get('gt_fwtm') else "---"
            pred_fwtm = f"{am['pred_fwtm']:.2f}" if am.get('pred_fwtm') else "---"
            lines.append(f"({pos['coords_mm']}) & {gt_fwhm} & {pred_fwhm} & {gt_fwtm} & {pred_fwtm} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    else:
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Z-axis FWHM and FWTM measurements for all point source positions (blurred, $\sigma=" + str(BLUR_SIGMA) + r"$, radius=" + str(BLUR_RADIUS) + r", GT only).}")
        lines.append(r"\label{tab:ps-fwhm-summary-blurred}")
        lines.append(r"\begin{tabular}{l c c}")
        lines.append(r"\hline")
        lines.append(r"Position (mm) & \textbf{FWHM (mm)} & \textbf{FWTM (mm)} \\")
        lines.append(r"\hline")
        for pos in positions:
            am = pos["metrics"].get("blurred", {}).get("z", {})
            gt_fwhm = f"{am['gt_fwhm']:.2f}" if am.get('gt_fwhm') else "---"
            gt_fwtm = f"{am['gt_fwtm']:.2f}" if am.get('gt_fwtm') else "---"
            lines.append(f"({pos['coords_mm']}) & {gt_fwhm} & {gt_fwtm} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

    lines.append("")

    tex_path = OUTPUT_DIR / "point_sources.tex"
    tex_path.write_text("\n".join(lines))
    print(f"\nGenerated LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
