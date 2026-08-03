#%%
"""Comparison viewer: show reconstruction outputs from multiple runs side-by-side."""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from petrecon.helpers.data_io import load_lors

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIRS = [
    "/home/h/Code/Opengate/ragy_files/IEC/gt",
    "/home/h/Code/Opengate/ragy_files/IEC/pred",
]
LABELS = None  # Set to a list of strings to label each run, or None to use folder names

SMOOTH_RECON_FOR_DISPLAY = True
DISPLAY_SMOOTH_SIGMA = 5 / 2.355
# DISPLAY_SMOOTH_RADIUS = 5 #mm

# Which slice to show: "hottest" picks the slice with the max voxel per image,
# "center" picks the center slice, or an int for a fixed slice index.
SLICE_MODE = "center" # "hottest" , "center" or int slice index

# ==============================================================================
# LOAD ALL RUNS
# ==============================================================================
if LABELS is None:
    LABELS = [os.path.basename(os.path.normpath(d)) for d in OUTPUT_DIRS]

n_runs = len(OUTPUT_DIRS)

runs = []
for i, output_dir in enumerate(OUTPUT_DIRS):
    print(f"Loading [{LABELS[i]}] from {output_dir}...")
    run = {}

    with open(os.path.join(output_dir, "recon_config.pkl"), "rb") as f:
        run["config"] = pickle.load(f)

    run["recon"] = np.load(os.path.join(output_dir, "reconstruction_image.npy"))
    run["sensitivity"] = np.load(os.path.join(output_dir, "sensitivity_image.npy"))

    bp_path = os.path.join(output_dir, "backprojection_image.npy")
    run["backprojection"] = np.load(bp_path) if os.path.exists(bp_path) else None

    mu_map = None
    cfg = run["config"]
    if cfg.get("ATTENUATION_CORRECTION") and cfg.get("MU_MAP_PATH"):
        if os.path.exists(cfg["MU_MAP_PATH"]):
            mu_map = np.load(cfg["MU_MAP_PATH"])
    run["mu_map"] = mu_map

    # Prepare display images
    recon_display = run["recon"].copy()
    if SMOOTH_RECON_FOR_DISPLAY:
        recon_display = gaussian_filter(recon_display, sigma=DISPLAY_SMOOTH_SIGMA)
        # recon_display = gaussian_filter(recon_display, sigma=DISPLAY_SMOOTH_SIGMA, radius=DISPLAY_SMOOTH_RADIUS)
    run["recon_display"] = recon_display

    sens_display = run["sensitivity"].copy()
    if cfg.get("SMOOTH_SENSITIVITY"):
        sens_display = gaussian_filter(sens_display, sigma=cfg["GAUSSIAN_SIGMA"], radius=cfg["GAUSSIAN_RADIUS"])
    run["sens_display"] = sens_display

    runs.append(run)

print(f"\nLoaded {n_runs} runs for comparison.")

# ==============================================================================
# HELPER: side-by-side plot for a given image type
# ==============================================================================
def get_slice_and_extent(image, axis, slice_mode, voxel_size):
    """Return 2D slice, extent for imshow, axis labels, and slice index."""
    nx, ny, nz = image.shape

    if slice_mode == "hottest":
        max_idx = np.unravel_index(np.argmax(image), image.shape)
        depth = max_idx[axis]
    elif slice_mode == "center":
        depth = image.shape[axis] // 2
    else:
        depth = int(slice_mode)

    half_x = nx * voxel_size / 2
    half_y = ny * voxel_size / 2
    half_z = nz * voxel_size / 2

    if axis == 0:
        slc = image[depth, :, :].T
        extent = [-half_y, half_y, -half_z, half_z]
        xlabel, ylabel = "Y (mm)", "Z (mm)"
    elif axis == 1:
        slc = image[:, depth, :].T
        extent = [-half_x, half_x, -half_z, half_z]
        xlabel, ylabel = "X (mm)", "Z (mm)"
    else:
        slc = image[:, :, depth].T
        extent = [-half_x, half_x, -half_y, half_y]
        xlabel, ylabel = "X (mm)", "Y (mm)"

    return slc, extent, xlabel, ylabel, depth


def compare_images(images, labels, suptitle, axis=2, slice_mode="hottest",
                   voxel_sizes=None, shared_vmax=True):
    """Plot a row of images side-by-side for comparison."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    # Compute shared color range if requested
    if shared_vmax:
        global_vmax = max(img.max() for img in images)
    else:
        global_vmax = None

    for i, (img, label) in enumerate(zip(images, labels)):
        vs = voxel_sizes[i] if voxel_sizes else 1.0
        slc, extent, xlabel, ylabel, depth = get_slice_and_extent(
            img, axis, slice_mode, vs
        )
        vmax = global_vmax if shared_vmax else img.max()
        im = axes[i].imshow(slc, cmap='hot', vmin=0, vmax=vmax,
                            origin='lower', extent=extent)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        depth_mm = (depth - img.shape[axis] // 2) * vs
        axes[i].set_title(f"{label}\n(slice {depth}, {depth_mm:.1f} mm)")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_line_profiles(images, labels, axis_profile='z', voxel_sizes=None):
    """Overlay line profiles from each run on the same plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + offset

    for i, (img, label) in enumerate(zip(images, labels)):
        vs = voxel_sizes[i] if voxel_sizes else 1.0
        max_idx = np.unravel_index(np.argmax(img), img.shape)

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax_int = axis_map[axis_profile] if isinstance(axis_profile, str) else axis_profile

        if ax_int == 0:
            line = img[:, max_idx[1], max_idx[2]]
        elif ax_int == 1:
            line = img[max_idx[0], :, max_idx[2]]
        else:
            line = img[max_idx[0], max_idx[1], :]

        n = len(line)
        x_mm = (np.arange(n) - (n - 1) / 2) * vs
        ax.plot(x_mm, line, linewidth=2, label=f"{label} (data)")

        # Fit gaussian
        x_data = np.arange(n)
        offset_guess = np.min(line)
        p0 = [np.max(line) - offset_guess, np.argmax(line), n / 10, offset_guess]
        try:
            popt, _ = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
            fwhm_mm = 2.355 * abs(popt[2]) * vs
            x_fit = np.linspace(0, n - 1, 500)
            x_fit_mm = (x_fit - (n - 1) / 2) * vs
            ax.plot(x_fit_mm, gaussian(x_fit, *popt), '--', linewidth=1.5,
                    label=f"{label} (FWHM={fwhm_mm:.2f} mm)")
        except RuntimeError:
            pass

    ax.set_xlabel(f"{axis_profile.upper()} position (mm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Line Profile Comparison (along {axis_profile.upper()}, through hottest voxel)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# DISPLAY CONFIGURATION COMPARISON
# ==============================================================================
print("\n" + "=" * 80)
print("CONFIGURATION COMPARISON")
print("=" * 80)
all_keys = set()
for r in runs:
    all_keys.update(r["config"].keys())
for key in sorted(all_keys):
    values = [str(r["config"].get(key, "N/A")) for r in runs]
    if len(set(values)) == 1:
        print(f"  {key}: {values[0]}")
    else:
        print(f"  {key}: " + " | ".join(f"[{LABELS[i]}] {v}" for i, v in enumerate(values)))
print("=" * 80 + "\n")

# ==============================================================================
# COMPARE: RECONSTRUCTION
# ==============================================================================
voxel_sizes = [r["config"]["VOXEL_SIZE_IN_MM"] for r in runs]

print("Comparing reconstruction images...")
compare_images(
    [r["recon_display"] for r in runs], LABELS,
    "Reconstruction Comparison (XY slice)", axis=2, slice_mode=SLICE_MODE,
    voxel_sizes=voxel_sizes,
)

compare_images(
    [r["recon_display"] for r in runs], LABELS,
    "Reconstruction Comparison (XZ slice)", axis=1, slice_mode=SLICE_MODE,
    voxel_sizes=voxel_sizes,
)

compare_images(
    [r["recon_display"] for r in runs], LABELS,
    "Reconstruction Comparison (YZ slice)", axis=0, slice_mode=SLICE_MODE,
    voxel_sizes=voxel_sizes,
)

# ==============================================================================
# COMPARE: SENSITIVITY
# ==============================================================================
print("Comparing sensitivity images...")
compare_images(
    [r["sens_display"] for r in runs], LABELS,
    "Sensitivity Comparison (XY slice)", axis=2, slice_mode="center",
    voxel_sizes=voxel_sizes,
)

# ==============================================================================
# COMPARE: BACKPROJECTION (if all runs have it)
# ==============================================================================
if all(r["backprojection"] is not None for r in runs):
    print("Comparing backprojection images...")
    compare_images(
        [r["backprojection"] for r in runs], LABELS,
        "Backprojection Comparison (XY slice)", axis=2, slice_mode=SLICE_MODE,
        voxel_sizes=voxel_sizes,
    )

# ==============================================================================
# COMPARE: MU-MAP (if all runs have it)
# ==============================================================================
if all(r["mu_map"] is not None for r in runs):
    print("Comparing mu-maps...")
    compare_images(
        [r["mu_map"] for r in runs], LABELS,
        "Mu-map Comparison (XY slice)", axis=2, slice_mode="center",
        voxel_sizes=voxel_sizes,
    )

# ==============================================================================
# COMPARE: LINE PROFILES
# ==============================================================================
print("Comparing line profiles...")
compare_line_profiles(
    [r["recon_display"] for r in runs], LABELS,
    axis_profile='z', voxel_sizes=voxel_sizes,
)

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for i, (r, label) in enumerate(zip(runs, LABELS)):
    cfg = r["config"]
    print(f"\n  [{label}]")
    print(f"    Algorithm: {cfg['RECON_ALGORITHM']}")
    print(f"    Grid: {cfg['IMAGE_GRID_SIZE']}, Voxel: {cfg['VOXEL_SIZE_IN_MM']} mm")
    print(f"    Sens: {cfg['SENSITIVITY_CORRECTION']}, Atten: {cfg['ATTENUATION_CORRECTION']}, Scatter: {cfg['SCATTER_CORRECTION']}")
    if cfg['RECON_ALGORITHM'] == 'OSEM':
        print(f"    OSEM: {cfg['OSEM_ITERATIONS']} iters x {cfg['OSEM_SUBSETS']} subsets")
    print(f"    Recon range: [{r['recon'].min():.4f}, {r['recon'].max():.4f}]")
print("=" * 80)
