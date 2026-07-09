#%%
"""Comparison viewer: show reconstruction outputs from multiple runs side-by-side."""
import numpy as np
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from helpers.data_io import load_lors

def make_sphere_mask(shape, centre, radius):
    """
    Create a boolean spherical mask.
    
    Parameters
    ----------
    shape  : tuple (nx, ny, nz) - image dimensions in voxels
    centre : tuple (cx, cy, cz) - sphere centre in voxel coordinates
    radius : float              - sphere radius in voxels (= mm for 1mm voxels)
    
    Returns
    -------
    mask : boolean ndarray of shape `shape`
    """
    cx, cy, cz = centre
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2


def sample_sphere(img, centre, radius):
    """Return mean voxel value inside a spherical ROI."""
    mask = make_sphere_mask(img.shape, centre, radius)
    return img[mask].mean(), mask

# ==============================================================================
# IEC CONFIGURATION
# ==============================================================================

IEC_SPHERES = [
    {"diameter_mm": 10, "angle_deg":   0, "cold": False},
    {"diameter_mm": 13, "angle_deg":  60, "cold": False},
    {"diameter_mm": 17, "angle_deg": 120, "cold": False},
    {"diameter_mm": 22, "angle_deg": 180, "cold": False},
    {"diameter_mm": 28, "angle_deg": 240, "cold": True},   # water-filled, no activity
    {"diameter_mm": 37, "angle_deg": 300, "cold": True},   # water-filled, no activity
]

RADIAL_OFFSET_MM = 57.2
PHANTOM_CENTRE = (128, 128, 128)

def get_sphere_centres(phantom_centre=(128, 128, 128), radial_offset=57.2):
    cx, cy, cz = phantom_centre
    centres = []
    for s in IEC_SPHERES:
        angle = np.deg2rad(s["angle_deg"])
        x = cx + radial_offset * np.cos(angle)
        y = cy + radial_offset * np.sin(angle)
        centres.append((x, y, cz))
    return centres

def get_background_centres(phantom_centre=(128, 128, 128), bg_radius=75.0, angle_offset_deg=15.0):
    cx, cy, cz = phantom_centre
    centres = []
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.deg2rad(angle_offset_deg)
        x = cx + bg_radius * np.cos(angle)
        y = cy + bg_radius * np.sin(angle)
        centres.append((x, y, cz))
    return centres

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIRS = [
    "/home/h/Opengate/ragy_files/IEC/gt",
    "/home/h/Opengate/ragy_files/IEC/pred",
]
LABELS = None  # Set to a list of strings to label each run, or None to use folder names

SMOOTH_RECON_FOR_DISPLAY = True
# DISPLAY_SMOOTH_SIGMA = 5 / 2.355
DISPLAY_SMOOTH_SIGMA = 3.4

BG_SPHERE_DIAMETER_MM = 15    # diameter of each of the 12 background ROI spheres
BG_RADIAL_OFFSET_MM = 85.0   # radial distance of background sphere centres from phantom centre

# Phantom geometry (from iec_body_phantom.mac)
PHANTOM_SEMI_MAJOR_MM = 147.5  # X semi-axis of elliptical body phantom (setLong)
PHANTOM_SEMI_MINOR_MM = 113.5  # Y semi-axis of elliptical body phantom (setShort)
LUNG_INSERT_RADIUS_MM = 25.0   # cold lung cylinder radius; set to None to hide

# CRC calculation
TRUE_HOT_RATIO = 4.0  # hot sphere : background activity concentration ratio (from .mac)

# Which image to use for sphere/background metrics: "recon" or "backprojection"
ANALYSIS_IMAGE = "backprojection"

# Set to a path to save figures as PDFs (e.g. for Overleaf), or None to skip saving
FIGURES_DIR = "/home/h/Opengate/ragy_files/IEC/figures"

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
    run["recon_display"] = recon_display

    sens_display = run["sensitivity"].copy()
    if cfg.get("SMOOTH_SENSITIVITY"):
        sens_display = gaussian_filter(sens_display, sigma=cfg["GAUSSIAN_SIGMA"], radius=cfg["GAUSSIAN_RADIUS"])
    run["sens_display"] = sens_display

    # Smoothed backprojection for display/analysis
    if run["backprojection"] is not None:
        bp_display = run["backprojection"].copy()
        if SMOOTH_RECON_FOR_DISPLAY:
            bp_display = gaussian_filter(bp_display, sigma=DISPLAY_SMOOTH_SIGMA)
        run["bp_display"] = bp_display
    else:
        run["bp_display"] = None

    # Select which image to use for sphere/background metrics
    if ANALYSIS_IMAGE == "backprojection":
        if run["bp_display"] is None:
            raise RuntimeError(
                f"ANALYSIS_IMAGE='backprojection' but no backprojection found in {output_dir}"
            )
        run["analysis_img"] = run["bp_display"]
    else:
        run["analysis_img"] = run["recon_display"]

    runs.append(run)

print(f"\nLoaded {n_runs} runs for comparison.")

if FIGURES_DIR is not None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name):
    """Save current figure to FIGURES_DIR as a PDF, if configured."""
    if FIGURES_DIR is not None:
        path = os.path.join(FIGURES_DIR, f"{name}.pdf")
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")

# ==============================================================================
# IEC SPHERE MASKS
# ==============================================================================
print("Computing IEC sphere masks...")

for run in runs:
    vs = run["config"]["VOXEL_SIZE_IN_MM"]
    img_shape = run["recon"].shape

    # Convert radial offset from mm to voxels; phantom centre stays in voxels
    radial_offset_vox = RADIAL_OFFSET_MM / vs
    centres_vox = get_sphere_centres(PHANTOM_CENTRE, radial_offset_vox)

    masks = []
    for j, s in enumerate(IEC_SPHERES):
        radius_vox = (s["diameter_mm"] / 2.0) / vs
        mask = make_sphere_mask(img_shape, centres_vox[j], radius_vox)
        masks.append(mask)

    run["sphere_centres_vox"] = centres_vox
    run["sphere_masks"] = masks

    # Background spheres
    bg_radial_offset_vox = BG_RADIAL_OFFSET_MM / vs
    bg_centres_vox = get_background_centres(PHANTOM_CENTRE, bg_radial_offset_vox)
    bg_radius_vox = (BG_SPHERE_DIAMETER_MM / 2.0) / vs
    bg_masks = [make_sphere_mask(img_shape, c, bg_radius_vox) for c in bg_centres_vox]

    run["bg_centres_vox"] = bg_centres_vox
    run["bg_masks"] = bg_masks

    combined_mask = np.zeros(img_shape, dtype=bool)
    for m in masks:
        combined_mask |= m
    bg_combined_mask = np.zeros(img_shape, dtype=bool)
    for m in bg_masks:
        bg_combined_mask |= m
    run["combined_mask"] = combined_mask
    run["bg_combined_mask"] = bg_combined_mask

    # Smoothed analysis image with voxels outside all sphere ROIs zeroed
    run["recon_masked"] = run["analysis_img"] * combined_mask

print("Sphere masks computed.")

# ==============================================================================
# CRC AND BACKGROUND METRICS
# ==============================================================================
print("Computing CRC and background metrics...")

for run in runs:
    img = run["analysis_img"]  # recon_display or bp_display depending on ANALYSIS_IMAGE

    # --- Background ROIs ---
    bg_roi_means = np.array([img[m].mean() for m in run["bg_masks"]])
    Q_bg = bg_roi_means.mean()
    bg_sd = bg_roi_means.std(ddof=1)          # between-ROI SD
    bg_cov_pct = bg_sd / Q_bg * 100.0         # coefficient of variation (%)

    run["bg_roi_means"] = bg_roi_means
    run["Q_bg"] = Q_bg
    run["bg_sd"] = bg_sd
    run["bg_cov_pct"] = bg_cov_pct

    # --- Per-sphere metrics ---
    sphere_metrics = []
    for s, mask in zip(IEC_SPHERES, run["sphere_masks"]):
        Q_sphere = img[mask].mean()
        sphere_bg_ratio = Q_sphere / Q_bg

        if s["cold"]:
            # Cold sphere: ideal ratio = 0, CRC = 1 - measured/background
            crc = 1.0 - sphere_bg_ratio
        else:
            # Hot sphere: CRC = (measured_ratio - 1) / (true_ratio - 1)
            crc = (sphere_bg_ratio - 1.0) / (TRUE_HOT_RATIO - 1.0)

        sphere_metrics.append({
            "diameter_mm": s["diameter_mm"],
            "cold": s["cold"],
            "Q_sphere": Q_sphere,
            "sphere_bg_ratio": sphere_bg_ratio,
            "CRC": crc,
        })
    run["sphere_metrics"] = sphere_metrics

print("Metrics computed.")

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
                   voxel_sizes=None, shared_vmax=True, save_name=None):
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
    savefig(save_name)
    plt.show()


def compare_line_profiles(images, labels, axis_profile='z', voxel_sizes=None, save_name=None):
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
    savefig(save_name)
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
    voxel_sizes=voxel_sizes, save_name="fig_recon_xy",
)

compare_images(
    [r["recon_display"] for r in runs], LABELS,
    "Reconstruction Comparison (XZ slice)", axis=1, slice_mode=SLICE_MODE,
    voxel_sizes=voxel_sizes, save_name="fig_recon_xz",
)

compare_images(
    [r["recon_display"] for r in runs], LABELS,
    "Reconstruction Comparison (YZ slice)", axis=0, slice_mode=SLICE_MODE,
    voxel_sizes=voxel_sizes, save_name="fig_recon_yz",
)

# ==============================================================================
# COMPARE: SENSITIVITY
# ==============================================================================
print("Comparing sensitivity images...")
compare_images(
    [r["sens_display"] for r in runs], LABELS,
    "Sensitivity Comparison (XY slice)", axis=2, slice_mode="center",
    voxel_sizes=voxel_sizes, save_name="fig_sensitivity_xy",
)

# ==============================================================================
# COMPARE: BACKPROJECTION (if all runs have it)
# ==============================================================================
if all(r["bp_display"] is not None for r in runs):
    print("Comparing backprojection images...")
    compare_images(
        [r["bp_display"] for r in runs], LABELS,
        "Backprojection Comparison (XY slice)", axis=2, slice_mode=SLICE_MODE,
        voxel_sizes=voxel_sizes, save_name="fig_backprojection_xy",
    )

# ==============================================================================
# COMPARE: MU-MAP (if all runs have it)
# ==============================================================================
if all(r["mu_map"] is not None for r in runs):
    print("Comparing mu-maps...")
    compare_images(
        [r["mu_map"] for r in runs], LABELS,
        "Mu-map Comparison (XY slice)", axis=2, slice_mode="center",
        voxel_sizes=voxel_sizes, save_name="fig_mumap_xy",
    )

# ==============================================================================
# COMPARE: LINE PROFILES
# ==============================================================================
print("Comparing line profiles...")
compare_line_profiles(
    [r["recon_display"] for r in runs], LABELS,
    axis_profile='z', voxel_sizes=voxel_sizes, save_name="fig_line_profiles",
)

# ==============================================================================
# PRINT: RAW SPHERE AND BACKGROUND METRICS
# ==============================================================================
col_w = 12
header = f"{'Sphere':>10}  {'Type':>6}  " + "  ".join(f"{l:>{col_w}}" for l in LABELS)
sep = "=" * len(header)

print("\n" + sep)
print(f"IEC SPHERE METRICS  (ANALYSIS_IMAGE='{ANALYSIS_IMAGE}', smoothed)")
print(sep)

# Background validity check
print("\n--- Background ROI metrics ---")
for label, run in zip(LABELS, runs):
    measured_ratio = np.mean([m["sphere_bg_ratio"] for m, s in
                               zip(run["sphere_metrics"], IEC_SPHERES) if not s["cold"]])
    bg_reliable = measured_ratio < (TRUE_HOT_RATIO * 3)
    flag = "" if bg_reliable else "  *** BACKGROUND STARVED — CRC UNRELIABLE ***"
    print(f"  [{label}]{flag}")
    print(f"    Q_bg (mean of 12 ROIs)   : {run['Q_bg']:.6f}")
    print(f"    Between-ROI SD           : {run['bg_sd']:.6f}")
    print(f"    Coefficient of variation : {run['bg_cov_pct']:.2f} %")
    print(f"    Mean hot Q_sphere/Q_bg   : {measured_ratio:.1f}  (expected ≈ {TRUE_HOT_RATIO:.1f})")

# Raw sphere means
print("\n--- Raw sphere means (Q_sphere) ---")
print(header)
for j, s in enumerate(IEC_SPHERES):
    tag = "cold" if s["cold"] else "hot"
    row = f"{s['diameter_mm']:>8} mm  {tag:>6}  "
    row += "  ".join(f"{run['sphere_metrics'][j]['Q_sphere']:>{col_w}.6f}" for run in runs)
    print(row)

# CRC table — printed with caveat if background is unreliable
any_starved = any(
    np.mean([m["sphere_bg_ratio"] for m, s in zip(run["sphere_metrics"], IEC_SPHERES)
             if not s["cold"]]) >= TRUE_HOT_RATIO * 3
    for run in runs
)
caveat = "  [CAVEAT: background starved — values not physically meaningful]" if any_starved else ""
print(f"\n--- Hot sphere CRC  (formula: (Q_s/Q_bg − 1)/(r−1), r={TRUE_HOT_RATIO:.0f}){caveat} ---")
print(header)
for j, s in enumerate(IEC_SPHERES):
    if s["cold"]:
        continue
    row = f"{s['diameter_mm']:>8} mm  {'hot':>6}  "
    row += "  ".join(f"{run['sphere_metrics'][j]['CRC']:>{col_w}.4f}" for run in runs)
    print(row)

print(f"\n--- Cold sphere residual  (1 − Q_s/Q_bg; ideal = 1.000){caveat} ---")
print(header)
for j, s in enumerate(IEC_SPHERES):
    if not s["cold"]:
        continue
    row = f"{s['diameter_mm']:>8} mm  {'cold':>6}  "
    row += "  ".join(f"{run['sphere_metrics'][j]['CRC']:>{col_w}.4f}" for run in runs)
    print(row)

# Relative recovery: run[1]/run[0] etc. (pred vs gt)
if n_runs >= 2:
    ref_run = runs[0]
    print(f"\n--- Relative sphere mean  [{LABELS[1]}] / [{LABELS[0]}] ---")
    print(header)
    for j, s in enumerate(IEC_SPHERES):
        tag = "cold" if s["cold"] else "hot"
        q_ref = ref_run["sphere_metrics"][j]["Q_sphere"]
        row = f"{s['diameter_mm']:>8} mm  {tag:>6}  "
        row += "  ".join(
            f"{run['sphere_metrics'][j]['Q_sphere'] / q_ref:>{col_w}.4f}"
            if ri > 0 else f"{'1.0000 (ref)':>{col_w}}"
            for ri, run in enumerate(runs)
        )
        print(row)

# Intra-run sphere-to-sphere ratios normalised to 22mm hot sphere
ref_sphere_idx = next(j for j, s in enumerate(IEC_SPHERES)
                      if s["diameter_mm"] == 22 and not s["cold"])
print(f"\n--- Intra-run sphere ratio  (normalised to 22 mm hot sphere) ---")
print(header)
for j, s in enumerate(IEC_SPHERES):
    tag = "cold" if s["cold"] else "hot"
    row = f"{s['diameter_mm']:>8} mm  {tag:>6}  "
    row += "  ".join(
        f"{run['sphere_metrics'][j]['Q_sphere'] / run['sphere_metrics'][ref_sphere_idx]['Q_sphere']:>{col_w}.4f}"
        for run in runs
    )
    print(row)

print(sep + "\n")

# ==============================================================================
# PLOT: RAW SPHERE MEANS (bar chart)
# ==============================================================================
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
hot_idxs  = [j for j, s in enumerate(IEC_SPHERES) if not s["cold"]]
cold_idxs = [j for j, s in enumerate(IEC_SPHERES) if s["cold"]]
hot_diams  = [IEC_SPHERES[j]["diameter_mm"] for j in hot_idxs]
cold_diams = [IEC_SPHERES[j]["diameter_mm"] for j in cold_idxs]

fig, axes_q = plt.subplots(1, 2, figsize=(13, 5))
bar_w = 0.8 / n_runs

for ri, (run, label) in enumerate(zip(runs, LABELS)):
    c = colours[ri % len(colours)]
    offset = (ri - (n_runs - 1) / 2) * bar_w

    hot_vals  = [run["sphere_metrics"][j]["Q_sphere"] for j in hot_idxs]
    cold_vals = [run["sphere_metrics"][j]["Q_sphere"] for j in cold_idxs]

    axes_q[0].bar(np.arange(len(hot_idxs)) + offset, hot_vals,
                  width=bar_w, color=c, alpha=0.8, label=label)
    axes_q[1].bar(np.arange(len(cold_idxs)) + offset, cold_vals,
                  width=bar_w, color=c, alpha=0.8, label=label)

for ax, diams, title in [
    (axes_q[0], hot_diams,  "Hot sphere mean voxel value"),
    (axes_q[1], cold_diams, "Cold sphere mean voxel value"),
]:
    ax.set_xticks(np.arange(len(diams)))
    ax.set_xticklabels([f"{d} mm" for d in diams])
    ax.set_ylabel("Mean voxel value (smoothed recon)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Sphere Mean Voxel Values", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("fig_sphere_means")
plt.show()

# ==============================================================================
# PLOT: CRC vs SPHERE DIAMETER  (fig:crc)
# ==============================================================================
fig_crc, axes_crc = plt.subplots(1, 2, figsize=(13, 5))
hot_diams_crc  = [s["diameter_mm"] for s in IEC_SPHERES if not s["cold"]]
cold_diams_crc = [s["diameter_mm"] for s in IEC_SPHERES if s["cold"]]

for ri, (run, label) in enumerate(zip(runs, LABELS)):
    c = colours[ri % len(colours)]
    hot_crcs  = [m["CRC"] for m, s in zip(run["sphere_metrics"], IEC_SPHERES) if not s["cold"]]
    cold_crcs = [m["CRC"] for m, s in zip(run["sphere_metrics"], IEC_SPHERES) if s["cold"]]
    axes_crc[0].plot(hot_diams_crc,  hot_crcs,  "o-", color=c, linewidth=2, markersize=7, label=label)
    axes_crc[1].plot(cold_diams_crc, cold_crcs, "s--", color=c, linewidth=2, markersize=7, label=label)

axes_crc[0].axhline(1.0, color="grey", linewidth=1, linestyle="--", label="ideal = 1")
axes_crc[0].set_xlabel("Sphere diameter (mm)")
axes_crc[0].set_ylabel("CRC")
axes_crc[0].set_title(f"Hot sphere CRC  (true ratio {TRUE_HOT_RATIO:.0f}:1)")
axes_crc[0].legend()
axes_crc[0].grid(True, alpha=0.3)
axes_crc[0].set_ylim(bottom=0)

axes_crc[1].axhline(1.0, color="grey", linewidth=1, linestyle="--", label="ideal = 1")
axes_crc[1].set_xlabel("Sphere diameter (mm)")
axes_crc[1].set_ylabel(r"CRC  $(1 - \bar{Q}_\mathrm{sphere}/\bar{Q}_\mathrm{bg})$")
axes_crc[1].set_title("Cold sphere residual contrast")
axes_crc[1].legend()
axes_crc[1].grid(True, alpha=0.3)
axes_crc[1].set_ylim(bottom=0)

fig_crc.suptitle("Contrast Recovery Coefficient", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("fig_crc")
plt.show()

# ==============================================================================
# PLOT: RELATIVE RECOVERY  pred / gt  per sphere  (fig:recovery)
# ==============================================================================
if n_runs >= 2:
    fig_rec, ax_rel = plt.subplots(figsize=(9, 4))
    all_diams = [s["diameter_mm"] for s in IEC_SPHERES]
    all_tags  = ["cold" if s["cold"] else "hot" for s in IEC_SPHERES]
    x = np.arange(len(IEC_SPHERES))

    for ri in range(1, n_runs):
        ratios = [
            runs[ri]["sphere_metrics"][j]["Q_sphere"] /
            runs[0]["sphere_metrics"][j]["Q_sphere"]
            for j in range(len(IEC_SPHERES))
        ]
        ax_rel.plot(x, ratios, "o-", linewidth=2, markersize=7,
                    label=f"{LABELS[ri]} / {LABELS[0]}")

    ax_rel.axhline(1.0, color="grey", linewidth=1, linestyle="--", label="ideal = 1")
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels([f"{d} mm\n({t})" for d, t in zip(all_diams, all_tags)], fontsize=8)
    ax_rel.set_ylabel("$Q_\\mathrm{sphere}$ ratio")
    ax_rel.set_title("Relative sphere mean recovery  (pred / gt per sphere)")
    ax_rel.legend()
    ax_rel.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("fig_recovery")
    plt.show()

# ==============================================================================
# PLOT: INTRA-RUN SPHERE-TO-SPHERE RATIOS  (normalised to 22 mm hot)
# ==============================================================================
fig, ax_ratio = plt.subplots(figsize=(9, 4))
x = np.arange(len(IEC_SPHERES))

for ri, (run, label) in enumerate(zip(runs, LABELS)):
    q_ref = run["sphere_metrics"][ref_sphere_idx]["Q_sphere"]
    ratios = [run["sphere_metrics"][j]["Q_sphere"] / q_ref for j in range(len(IEC_SPHERES))]
    ax_ratio.plot(x, ratios, "o-", linewidth=2, markersize=7,
                  color=colours[ri % len(colours)], label=label)

ax_ratio.axhline(1.0, color="grey", linewidth=1, linestyle="--", label="22 mm reference")
ax_ratio.set_xticks(x)
ax_ratio.set_xticklabels([f"{s['diameter_mm']} mm\n({'cold' if s['cold'] else 'hot'})"
                           for s in IEC_SPHERES], fontsize=8)
ax_ratio.set_ylabel("$Q_\\mathrm{sphere}$ / $Q_\\mathrm{22mm\\ hot}$")
ax_ratio.set_title("Intra-run sphere-to-sphere ratios  (normalised to 22 mm hot sphere)")
ax_ratio.legend()
ax_ratio.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig_sphere_ratios")
plt.show()

# ==============================================================================
# QC: IEC SPHERE MASK VISUALISATION
# ==============================================================================
print("QC: visualising IEC sphere masks...")

fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5))
if n_runs == 1:
    axes = [axes]

for i, (run, label) in enumerate(zip(runs, LABELS)):
    vs = run["config"]["VOXEL_SIZE_IN_MM"]
    img = run["recon_display"]
    nx, ny, nz = img.shape

    z_centre = nz // 2
    slc = img[:, :, z_centre].T

    half_x = nx * vs / 2
    half_y = ny * vs / 2
    extent = [-half_x, half_x, -half_y, half_y]

    axes[i].imshow(slc, cmap="hot", origin="lower", extent=extent,
                   vmin=0, vmax=img.max())

    for j, (s, centre) in enumerate(zip(IEC_SPHERES, run["sphere_centres_vox"])):
        cx_mm = (centre[0] - nx / 2) * vs
        cy_mm = (centre[1] - ny / 2) * vs
        r_mm = s["diameter_mm"] / 2.0
        colour = "deepskyblue" if not s["cold"] else "lime"
        circle = plt.Circle((cx_mm, cy_mm), r_mm, fill=False,
                             edgecolor=colour, linewidth=1.5)
        axes[i].add_patch(circle)
        tag = f"{s['diameter_mm']} mm" + (" (cold)" if s["cold"] else "")
        axes[i].text(cx_mm, cy_mm + r_mm + 1.5, tag,
                     color=colour, fontsize=7, ha="center", va="bottom")

    bg_r_mm = BG_SPHERE_DIAMETER_MM / 2.0
    for centre in run["bg_centres_vox"]:
        cx_mm = (centre[0] - nx / 2) * vs
        cy_mm = (centre[1] - ny / 2) * vs
        circle = plt.Circle((cx_mm, cy_mm), bg_r_mm, fill=False,
                             edgecolor="yellow", linewidth=1.0, linestyle="--")
        axes[i].add_patch(circle)

    # Phantom outer boundary (ellipse from .mac: setLong x setShort)
    ellipse = matplotlib.patches.Ellipse(
        (0, 0),
        width=2 * PHANTOM_SEMI_MAJOR_MM,
        height=2 * PHANTOM_SEMI_MINOR_MM,
        fill=False, edgecolor="white", linewidth=1.5, linestyle="-",
    )
    axes[i].add_patch(ellipse)

    # Lung insert
    if LUNG_INSERT_RADIUS_MM is not None:
        lung_circle = plt.Circle((0, 0), LUNG_INSERT_RADIUS_MM, fill=False,
                                 edgecolor="lightgrey", linewidth=1.0, linestyle=":")
        axes[i].add_patch(lung_circle)

    axes[i].set_xlabel("X (mm)")
    axes[i].set_ylabel("Y (mm)")
    axes[i].set_title(f"{label}\nIEC sphere masks — centre XY slice\n"
                      f"blue=hot  green=cold  yellow=bg  white=phantom  grey=lung")

fig.suptitle("IEC Sphere Mask QC", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("fig_mask_qc")
plt.show()

# ==============================================================================
# COMPARE: ORIGINAL (SMOOTHED) vs SPHERE-MASKED
# ==============================================================================
print("Comparing smoothed vs sphere-masked reconstructions...")

for run, label in zip(runs, LABELS):
    vs = run["config"]["VOXEL_SIZE_IN_MM"]
    compare_images(
        [run["analysis_img"], run["recon_masked"]],
        [f"{label} — {ANALYSIS_IMAGE} (smoothed)", f"{label} — masked"],
        f"Smoothed vs Sphere-Masked: {label}",
        axis=2,
        slice_mode=SLICE_MODE,
        voxel_sizes=[vs, vs],
        shared_vmax=True,
        save_name=f"fig_masked_{label}",
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

# %%
