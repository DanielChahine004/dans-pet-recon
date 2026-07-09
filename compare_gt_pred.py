#%%
"""GT vs Predicted comparison: recon images, line profiles, and LOR endpoint error analysis."""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from helpers.data_io import load_lors

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Set to a single coord string OR a list of coord strings.
# e.g. COORDS = "0_0_120"
COORDS = [
    "0_0_0", "0_0_32", "0_0_64", "0_0_64", "0_0_120",
    "32_32_0", "32_32_32", "32_32_64", "32_32_64", "32_32_120",
    "64_64_0", "64_64_32", "64_64_64", "64_64_64", "64_64_120",
    "96_96_0", "96_96_32", "96_96_64", "96_96_64", "96_96_120",
    "120_120_0", "120_120_32", "120_120_64", "120_120_64", "120_120_120",
          ]
# COORDS = "0_0_120"

BASE_DIR = "/home/h/Opengate/ragy_files/Point_Sources_Processed"

# Normalise to list
COORD_LIST = [COORDS] if isinstance(COORDS, str) else list(COORDS)

SMOOTH_RECON_FOR_DISPLAY = True
DISPLAY_SMOOTH_SIGMA = 1
DISPLAY_SMOOTH_RADIUS = 1

# Which slice to show: "hottest", "center", or an int index.
SLICE_MODE = "hottest"

# Scanner geometry (for reference lines on endpoint distribution plots)
AXIAL_LENGTH = 296        # mm  (full axial FOV)
INNER_RADIUS = 117.711 * 2  # mm
OUTER_RADIUS = 139.148 * 2  # mm

# Number of radial / axial / angular bins for error histograms
N_RADIAL_BINS  = 10
N_AXIAL_BINS   = 10
N_ANGULAR_BINS = 36   # bullseye angular slices (360 / N_ANGULAR_BINS degrees each)

# ==============================================================================
# LOAD RECON OUTPUTS
# ==============================================================================
def load_run(output_dir, smooth=True, sigma=1, radius=1):
    run = {}
    with open(os.path.join(output_dir, "recon_config.pkl"), "rb") as f:
        run["config"] = pickle.load(f)

    run["recon"]       = np.load(os.path.join(output_dir, "reconstruction_image.npy"))
    run["sensitivity"] = np.load(os.path.join(output_dir, "sensitivity_image.npy"))

    recon_display = run["recon"].copy()
    if smooth:
        recon_display = gaussian_filter(recon_display, sigma=sigma, radius=radius)
    run["recon_display"] = recon_display
    return run


# ==============================================================================
# LOAD ALL COORDS
# ==============================================================================
coord_data = {}  # coord -> {"gt": ..., "pred": ..., "gt_lors": ..., "pred_lors": ...}

for coord in COORD_LIST:
    root = f"{BASE_DIR}/{coord}"
    gt_dir       = f"{root}/gt"
    gt_lor_path  = f"{root}/gt_processed_{coord}.csv"
    pred_dir     = f"{root}/pred"
    pred_lor_path = f"{root}/prediction_processed_{coord}.csv"

    print(f"\n[{coord}] Loading GT ...")
    gt   = load_run(gt_dir,   SMOOTH_RECON_FOR_DISPLAY, DISPLAY_SMOOTH_SIGMA, DISPLAY_SMOOTH_RADIUS)
    print(f"[{coord}] Loading Pred ...")
    pred = load_run(pred_dir, SMOOTH_RECON_FOR_DISPLAY, DISPLAY_SMOOTH_SIGMA, DISPLAY_SMOOTH_RADIUS)

    print(f"[{coord}] Loading LORs ...")
    gt_lors   = load_lors(gt_lor_path,   default_path=os.path.join(gt_dir,   "lors.npz"))
    pred_lors = load_lors(pred_lor_path, default_path=os.path.join(pred_dir, "lors.npz"))

    n_pairs = min(len(gt_lors), len(pred_lors))
    if len(gt_lors) != len(pred_lors):
        print(f"  WARNING: GT has {len(gt_lors)} LORs, Pred has {len(pred_lors)}. "
              f"Using first {n_pairs} pairs.")
    gt_lors   = gt_lors[:n_pairs]
    pred_lors = pred_lors[:n_pairs]

    coord_data[coord] = {
        "gt": gt, "pred": pred,
        "gt_lors": gt_lors, "pred_lors": pred_lors,
    }

# Aggregate LOR errors across all coords
all_gt_lors   = np.concatenate([d["gt_lors"]   for d in coord_data.values()], axis=0)
all_pred_lors = np.concatenate([d["pred_lors"] for d in coord_data.values()], axis=0)

err_p1   = np.linalg.norm(all_pred_lors[:, :3] - all_gt_lors[:, :3], axis=1)
err_p2   = np.linalg.norm(all_pred_lors[:, 3:] - all_gt_lors[:, 3:], axis=1)
err_mean = (err_p1 + err_p2) / 2.0

gt_mid_x  = (all_gt_lors[:, 0] + all_gt_lors[:, 3]) / 2.0
gt_mid_y  = (all_gt_lors[:, 1] + all_gt_lors[:, 4]) / 2.0
gt_mid_z  = (all_gt_lors[:, 2] + all_gt_lors[:, 5]) / 2.0
gt_radial = np.sqrt(gt_mid_x**2 + gt_mid_y**2)
gt_axial  = gt_mid_z

# Endpoint radial distances (both GT and Pred, both endpoints)
gt_r1   = np.sqrt(all_gt_lors[:, 0]**2   + all_gt_lors[:, 1]**2)
gt_r2   = np.sqrt(all_gt_lors[:, 3]**2   + all_gt_lors[:, 4]**2)
pred_r1 = np.sqrt(all_pred_lors[:, 0]**2 + all_pred_lors[:, 1]**2)
pred_r2 = np.sqrt(all_pred_lors[:, 3]**2 + all_pred_lors[:, 4]**2)

gt_z1   = all_gt_lors[:, 2]
gt_z2   = all_gt_lors[:, 5]
pred_z1 = all_pred_lors[:, 2]
pred_z2 = all_pred_lors[:, 5]

# LORs where either predicted endpoint falls outside the physical scanner bounds
oob_radial = ((pred_r1 < INNER_RADIUS) | (pred_r1 > OUTER_RADIUS) |
              (pred_r2 < INNER_RADIUS) | (pred_r2 > OUTER_RADIUS))
oob_axial  = ((np.abs(pred_z1) > AXIAL_LENGTH / 2) |
              (np.abs(pred_z2) > AXIAL_LENGTH / 2))
oob_mask   = oob_radial | oob_axial

n_pairs = len(err_mean)
print(f"  OOB LORs: {oob_mask.sum()} / {n_pairs} ({100*oob_mask.mean():.1f}%)")
print(f"\nLOR error stats across {len(COORD_LIST)} coord(s) — {n_pairs} total pairs (mean of both endpoints, mm):")
print(f"  Mean:   {err_mean.mean():.3f}")
print(f"  Median: {np.median(err_mean):.3f}")
print(f"  Std:    {err_mean.std():.3f}")
print(f"  95th %: {np.percentile(err_mean, 95):.3f}")

# ==============================================================================
# HELPERS
# ==============================================================================
def get_slice(image, axis, slice_mode, voxel_size):
    nx, ny, nz = image.shape
    if slice_mode == "hottest":
        depth = np.unravel_index(np.argmax(image), image.shape)[axis]
    elif slice_mode == "center":
        depth = image.shape[axis] // 2
    else:
        depth = int(slice_mode)

    half_x = nx * voxel_size / 2
    half_y = ny * voxel_size / 2
    half_z = nz * voxel_size / 2

    if axis == 0:
        return image[depth, :, :].T, [-half_y, half_y, -half_z, half_z], "Y (mm)", "Z (mm)", depth
    elif axis == 1:
        return image[:, depth, :].T, [-half_x, half_x, -half_z, half_z], "X (mm)", "Z (mm)", depth
    else:
        return image[:, :, depth].T, [-half_x, half_x, -half_y, half_y], "X (mm)", "Y (mm)", depth


def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset


# ==============================================================================
# FIGURE 1 — RECONSTRUCTION COMPARISON (3 views, one figure per coord)
# ==============================================================================
for coord, data in coord_data.items():
    gt   = data["gt"]
    pred = data["pred"]
    gt_vs   = gt["config"]["VOXEL_SIZE_IN_MM"]
    pred_vs = pred["config"]["VOXEL_SIZE_IN_MM"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Reconstruction: GT vs Predicted  [{coord}]", fontsize=14, fontweight='bold')

    for col, axis in enumerate([2, 1, 0]):
        axis_names = {2: "XY", 1: "XZ", 0: "YZ"}

        gt_slc,   gt_ext,   xl, yl, gt_d   = get_slice(gt["recon_display"],   axis, SLICE_MODE, gt_vs)
        pred_slc, pred_ext, _,  _,  pred_d = get_slice(pred["recon_display"], axis, SLICE_MODE, pred_vs)

        shared_vmax = max(gt_slc.max(), pred_slc.max())

        for row, (slc, ext, label, d, vs) in enumerate([
            (gt_slc,   gt_ext,   "GT",   gt_d,   gt_vs),
            (pred_slc, pred_ext, "Pred", pred_d, pred_vs),
        ]):
            ax = axes[row, col]
            im = ax.imshow(slc, cmap='hot', vmin=0, vmax=shared_vmax,
                           origin='lower', extent=ext)
            depth_mm = (d - slc.shape[1] // 2) * vs
            ax.set_title(f"{label} — {axis_names[axis]} (slice {d}, {depth_mm:.1f} mm)")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIGURE 2 — LINE PROFILES (GT vs Pred, through hottest voxel, along Z, per coord)
# ==============================================================================
for coord, data in coord_data.items():
    gt   = data["gt"]
    pred = data["pred"]
    gt_vs   = gt["config"]["VOXEL_SIZE_IN_MM"]
    pred_vs = pred["config"]["VOXEL_SIZE_IN_MM"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Line Profile: GT vs Predicted (through hottest voxel, along Z)  [{coord}]",
                 fontsize=13, fontweight='bold')

    for img, label, vs, color in [
        (gt["recon_display"],   "GT",   gt_vs,   "steelblue"),
        (pred["recon_display"], "Pred", pred_vs, "tomato"),
    ]:
        max_idx = np.unravel_index(np.argmax(img), img.shape)
        line = img[max_idx[0], max_idx[1], :]
        n = len(line)
        x_mm = (np.arange(n) - (n - 1) / 2) * vs
        ax.plot(x_mm, line, linewidth=2, label=f"{label} (data)", color=color)

        x_data = np.arange(n)
        p0 = [np.max(line) - np.min(line), np.argmax(line), n / 10, np.min(line)]
        try:
            popt, _ = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
            fwhm_mm = 2.355 * abs(popt[2]) * vs
            x_fit = np.linspace(0, n - 1, 500)
            x_fit_mm = (x_fit - (n - 1) / 2) * vs
            ax.plot(x_fit_mm, gaussian(x_fit, *popt), '--', linewidth=1.5, color=color,
                    label=f"{label} fit (FWHM={fwhm_mm:.2f} mm)")
        except RuntimeError:
            pass

    ax.set_xlabel("Z position (mm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================
def _scatter_binned(ax, x, y, xlabel, title, median_color, n_bins=None):
    if n_bins is None:
        n_bins = N_RADIAL_BINS
    ax.scatter(x, y, s=0.5, alpha=0.15, color='grey', rasterized=True)
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    ctrs, meds, q25s, q75s = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() > 0:
            v = y[m]
            ctrs.append((lo + hi) / 2)
            meds.append(np.median(v))
            q25s.append(np.percentile(v, 25))
            q75s.append(np.percentile(v, 75))
    ctrs, meds, q25s, q75s = map(np.array, [ctrs, meds, q25s, q75s])
    ax.plot(ctrs, meds, 'o-', color=median_color, linewidth=2, label='Median')
    ax.fill_between(ctrs, q25s, q75s, alpha=0.3, color=median_color, label='IQR')
    ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean endpoint error (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_fig3(d, title_suffix=""):
    """LOR endpoint error analysis (midpoint-based spatial plots)."""
    ep1, ep2, emean  = d['err_p1'], d['err_p2'], d['err_mean']
    radial, axial    = d['gt_radial'], d['gt_axial']
    mid_x, mid_y     = d['gt_mid_x'], d['gt_mid_y']

    gt_angle     = np.arctan2(mid_y, mid_x)
    azimuth_deg  = np.degrees(gt_angle)                           # –180 to +180
    polar_deg    = np.degrees(np.arctan2(radial, axial))          # 0° = +Z, 90° = XY plane

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"LOR Endpoint Error Analysis (GT vs Predicted){title_suffix}",
                 fontsize=14, fontweight='bold')

    # 3a — error distribution
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.hist(ep1,   bins=80, alpha=0.6, label="Endpoint 1", color="steelblue", density=True)
    ax1.hist(ep2,   bins=80, alpha=0.6, label="Endpoint 2", color="tomato",    density=True)
    ax1.hist(emean, bins=80, alpha=0.6, label="Mean",       color="seagreen",  density=True)
    ax1.set_xlabel("Error (mm)"); ax1.set_ylabel("Density")
    ax1.set_title("Overall Error Distribution")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # 3b — error vs radial midpoint
    ax2 = fig.add_subplot(2, 4, 2)
    _scatter_binned(ax2, radial, emean,
                    "Radial distance of GT midpoint (mm)", "Error vs Radial Distance", 'tomato')

    # 3c — error vs axial midpoint
    ax3 = fig.add_subplot(2, 4, 3)
    _scatter_binned(ax3, axial, emean,
                    "Axial position of GT midpoint (mm)", "Error vs Axial Position", 'steelblue',
                    n_bins=N_AXIAL_BINS)

    # 3d — error vs azimuthal angle (phi)
    ax4 = fig.add_subplot(2, 4, 4)
    _scatter_binned(ax4, azimuth_deg, emean,
                    "Azimuthal angle φ of GT midpoint (°)", "Error vs Azimuthal Angle", 'mediumpurple',
                    n_bins=N_ANGULAR_BINS)
    ax4.set_xlim(-180, 180)
    ax4.set_xticks(np.arange(-180, 181, 60))

    # 3e — bullseye
    ax5 = fig.add_subplot(2, 4, 5, projection='polar')
    r_edges  = np.linspace(radial.min(), radial.max(), N_RADIAL_BINS + 1)
    th_edges = np.linspace(-np.pi, np.pi, N_ANGULAR_BINS + 1)
    h_err_b, _, _ = np.histogram2d(gt_angle, radial, bins=[th_edges, r_edges], weights=emean)
    h_cnt_b, _, _ = np.histogram2d(gt_angle, radial, bins=[th_edges, r_edges])
    with np.errstate(invalid='ignore'):
        h_mean_b = np.where(h_cnt_b > 0, h_err_b / h_cnt_b, np.nan)
    im5 = ax5.pcolormesh(th_edges, r_edges, h_mean_b.T, cmap='hot_r')
    fig.colorbar(im5, ax=ax5, label="Mean error (mm)", shrink=0.7, pad=0.08)
    ax5.set_title("Mean LOR Error — Bullseye (XY)", pad=12)

    # 3f — RZ heatmap
    ax6 = fig.add_subplot(2, 4, 6)
    h_err_rz, re, ze = np.histogram2d(radial, axial, bins=40, weights=emean)
    h_cnt_rz, _,  _  = np.histogram2d(radial, axial, bins=40)
    with np.errstate(invalid='ignore'):
        h_mean_rz = np.where(h_cnt_rz > 0, h_err_rz / h_cnt_rz, np.nan)
    im6 = ax6.pcolormesh(re, ze, h_mean_rz.T, cmap='hot_r')
    fig.colorbar(im6, ax=ax6, label="Mean error (mm)")
    ax6.set_xlabel("Radial (mm)"); ax6.set_ylabel("Axial Z (mm)")
    ax6.set_title("Mean LOR Error in R-Z Plane")

    # 3g — CDF
    ax7 = fig.add_subplot(2, 4, 7)
    for err, lbl, col in [(ep1, "Endpoint 1", "steelblue"),
                          (ep2, "Endpoint 2", "tomato"),
                          (emean, "Mean",     "seagreen")]:
        s = np.sort(err)
        ax7.plot(s, np.arange(1, len(s) + 1) / len(s), linewidth=1.5, label=lbl, color=col)
    ax7.set_xlabel("Error (mm)"); ax7.set_ylabel("CDF")
    ax7.set_title("Cumulative Error Distribution")
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3); ax7.set_xlim(left=0)

    # 3h — error vs polar angle (theta from +Z axis)
    ax8 = fig.add_subplot(2, 4, 8)
    _scatter_binned(ax8, polar_deg, emean,
                    "Polar angle θ of GT midpoint (°)", "Error vs Polar Angle", 'darkorange',
                    n_bins=N_ANGULAR_BINS)
    ax8.set_xlim(0, 180)
    ax8.set_xticks(np.arange(0, 181, 30))

    plt.tight_layout()
    plt.show()


def plot_fig3b(d, title_suffix=""):
    """Error vs GT endpoint positions (radial & axial)."""
    emean = d['err_mean']

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Error vs GT Endpoint Positions (Radial & Axial){title_suffix}",
                 fontsize=13, fontweight='bold')

    _scatter_binned(axes[0, 0], d['gt_r1'], emean,
                    "GT endpoint 1 radial (mm)", "Error vs GT r — Endpoint 1", 'tomato')
    _scatter_binned(axes[0, 1], d['gt_r2'], emean,
                    "GT endpoint 2 radial (mm)", "Error vs GT r — Endpoint 2", 'tomato')
    _scatter_binned(axes[1, 0], d['gt_z1'], emean,
                    "GT endpoint 1 axial Z (mm)", "Error vs GT Z — Endpoint 1", 'steelblue',
                    n_bins=N_AXIAL_BINS)
    _scatter_binned(axes[1, 1], d['gt_z2'], emean,
                    "GT endpoint 2 axial Z (mm)", "Error vs GT Z — Endpoint 2", 'steelblue',
                    n_bins=N_AXIAL_BINS)
    plt.tight_layout()
    plt.show()


# Pack all per-LOR arrays into a dict for easy subsetting
def _make_data_dict(sel):
    return dict(
        err_p1=err_p1[sel], err_p2=err_p2[sel], err_mean=err_mean[sel],
        gt_radial=gt_radial[sel], gt_axial=gt_axial[sel],
        gt_mid_x=gt_mid_x[sel],  gt_mid_y=gt_mid_y[sel],
        gt_r1=gt_r1[sel],   gt_r2=gt_r2[sel],
        gt_z1=gt_z1[sel],   gt_z2=gt_z2[sel],
        pred_r1=pred_r1[sel], pred_r2=pred_r2[sel],
        pred_z1=pred_z1[sel], pred_z2=pred_z2[sel],
    )


def plot_fig4(d, title_suffix=""):
    """GT vs Pred endpoint radial and axial distributions with scanner reference lines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LOR Endpoint Distributions (GT vs Predicted){title_suffix}",
                 fontsize=13, fontweight='bold')

    for col, (gt_r, pred_r, gt_z, pred_z, ep_label) in enumerate([
        (d['gt_r1'], d['pred_r1'], d['gt_z1'], d['pred_z1'], "Endpoint 1"),
        (d['gt_r2'], d['pred_r2'], d['gt_z2'], d['pred_z2'], "Endpoint 2"),
    ]):
        ax = axes[0, col]
        ax.hist(gt_r,   bins=80, alpha=0.6, color='steelblue', density=True, label='GT')
        ax.hist(pred_r, bins=80, alpha=0.6, color='tomato',    density=True, label='Pred')
        ax.axvline(INNER_RADIUS, color='black',   linestyle='--', linewidth=1.2,
                   label=f'r_inner ({INNER_RADIUS:.1f} mm)')
        ax.axvline(OUTER_RADIUS, color='dimgrey', linestyle='--', linewidth=1.2,
                   label=f'r_outer ({OUTER_RADIUS:.1f} mm)')
        ax.set_xlabel("Radial distance (mm)")
        ax.set_ylabel("Density")
        ax.set_title(f"Radial — {ep_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.hist(gt_z,   bins=80, alpha=0.6, color='steelblue', density=True, label='GT')
        ax.hist(pred_z, bins=80, alpha=0.6, color='tomato',    density=True, label='Pred')
        ax.axvline(-AXIAL_LENGTH / 2, color='black', linestyle='--', linewidth=1.2,
                   label=f'±axial/2 (±{AXIAL_LENGTH/2:.0f} mm)')
        ax.axvline( AXIAL_LENGTH / 2, color='black', linestyle='--', linewidth=1.2)
        ax.set_xlabel("Z position (mm)")
        ax.set_ylabel("Density")
        ax.set_title(f"Axial — {ep_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIGURE 3 / 3b — ALL LORs
# ==============================================================================
all_data = _make_data_dict(np.ones(len(err_mean), dtype=bool))
plot_fig3(all_data)
plot_fig3b(all_data)
plot_fig4(all_data)

# ==============================================================================
# FIGURE 3 / 3b — OUT-OF-BOUNDS predicted endpoints only
# ==============================================================================
print(f"\nOOB subset: {oob_mask.sum()} / {len(err_mean)} LORs ({100*oob_mask.mean():.1f}%)")
oob_data = _make_data_dict(oob_mask)
plot_fig3(oob_data,  title_suffix=" — OOB Predictions")
plot_fig3b(oob_data, title_suffix=" — OOB Predictions")
plot_fig4(oob_data,  title_suffix=" — OOB Predictions")

# ==============================================================================
# FIGURE 3 / 3b — WORST 20% (highest mean error)
# ==============================================================================
worst20 = err_mean >= np.percentile(err_mean, 80)
print(f"\nWorst-20% subset: {worst20.sum()} LORs (threshold {np.percentile(err_mean, 80):.2f} mm)")
worst_data = _make_data_dict(worst20)
plot_fig3(worst_data,  title_suffix=" — Worst 20%")
plot_fig3b(worst_data, title_suffix=" — Worst 20%")
plot_fig4(worst_data,  title_suffix=" — Worst 20%")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for coord, data in coord_data.items():
    print(f"\n  [{coord}]")
    for label, run in [("GT", data["gt"]), ("Pred", data["pred"])]:
        cfg   = run["config"]
        recon = run["recon"]
        print(f"    {label}  Algorithm: {cfg['RECON_ALGORITHM']}  "
              f"Grid: {cfg['IMAGE_GRID_SIZE']}  Voxel: {cfg['VOXEL_SIZE_IN_MM']} mm  "
              f"Range: [{recon.min():.4f}, {recon.max():.4f}]")

print(f"\n  [LOR Error — {len(COORD_LIST)} coord(s), {n_pairs} total pairs]")
for label, err in [("Endpoint 1", err_p1), ("Endpoint 2", err_p2), ("Mean", err_mean)]:
    print(f"    {label:12s}: mean={err.mean():.3f} mm  "
          f"median={np.median(err):.3f} mm  "
          f"std={err.std():.3f} mm  "
          f"95th={np.percentile(err, 95):.3f} mm")
print("=" * 60)


# %%
# ==============================================================================
# SECTION 5 — LOR FILTERING REGIME ANALYSIS
# ==============================================================================
# Goal: identify filtering criteria derived *only* from predicted LOR
# coordinates (no GT access at inference time).
# GT errors are used here purely to validate that each filter actually
# reduces err_mean — they would NOT be available in production.
#
# Physical context
# ────────────────
#   INNER_RADIUS ≈ 235 mm  (inner crystal ring face)
#   OUTER_RADIUS ≈ 278 mm  (outer crystal ring face)
#   r_ring       ≈ 257 mm  (mid-crystal ring radius)
#   r_band       ≈  21 mm  (half-width of crystal ring)
#   AXIAL_LENGTH = 296 mm  (full axial FOV, ±148 mm)
# ==============================================================================

r_ring = (INNER_RADIUS + OUTER_RADIUS) / 2       # ≈ 257 mm — expected crystal radius
r_band = (OUTER_RADIUS - INNER_RADIUS) / 2       # ≈  21 mm — crystal band half-width

# ── Endpoint arrays (shorthand) ───────────────────────────────────────────────
p1 = all_pred_lors[:, :3]          # (N,3) predicted endpoint 1
p2 = all_pred_lors[:, 3:]          # (N,3) predicted endpoint 2
# pred_r1, pred_r2, pred_z1, pred_z2 already computed above

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 1 — Ring deviation
#   How far each predicted endpoint sits from the expected crystal ring.
#   Physical crystals must lie within [INNER_RADIUS, OUTER_RADIUS], so any
#   endpoint outside that band is already OOB; r_dev quantifies *how* far
#   in-band predictions are from the ring mid-line.
# ──────────────────────────────────────────────────────────────────────────────
r_dev1     = np.abs(pred_r1 - r_ring)
r_dev2     = np.abs(pred_r2 - r_ring)
r_dev_mean = (r_dev1 + r_dev2) / 2
r_dev_max  = np.maximum(r_dev1, r_dev2)

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 2 — Radial asymmetry  |r1 - r2|
#   Both endpoints should lie on the same detector ring, so their radial
#   distances should match.  Large asymmetry → one endpoint is mislocalised.
# ──────────────────────────────────────────────────────────────────────────────
r_asym = np.abs(pred_r1 - pred_r2)

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 3 — 3-D LOR chord length
#   For a ring of radius r_ring the maximum transaxial chord is 2*r_ring ≈ 514 mm.
#   Adding full axial span (296 mm) gives a 3-D maximum of ≈ ~590 mm.
#   Very short (<100 mm) or very long (>600 mm) chords are implausible.
# ──────────────────────────────────────────────────────────────────────────────
lor_vec    = p2 - p1                                  # (N,3)
lor_length = np.linalg.norm(lor_vec, axis=1)          # (N,)

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 4 — Obliqueness  (angle from the transaxial XY plane, degrees)
#   High-obliqueness LORs (large axial component) are harder for the model to
#   predict accurately and degrade resolution in the axial direction.
#   0° = purely transaxial;  90° = purely axial (physically impossible).
# ──────────────────────────────────────────────────────────────────────────────
lor_xy_len  = np.sqrt(lor_vec[:, 0]**2 + lor_vec[:, 1]**2)
oblique_deg = np.degrees(
    np.arctan2(np.abs(lor_vec[:, 2]), np.maximum(lor_xy_len, 1e-9))
)

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 5 — Minimum approach of the LOR line to the scanner z-axis  (mm)
#   A valid LOR must pass through the FOV.  Using the skew-lines formula:
#       n = d̂ × ẑ
#       dist = |P1 · n| / |n|
#   where d̂ is the unit LOR direction and ẑ = (0,0,1).
#   For a diametric LOR:  min_approach ≈ 0.
#   For endpoints on the same side (opening angle θ):
#       min_approach = r_ring · cos(θ/2)
#   Threshold INNER_RADIUS rejects LORs that don't cross the inner ring.
# ──────────────────────────────────────────────────────────────────────────────
lor_dir    = lor_vec / np.maximum(lor_length[:, None], 1e-9)
z_hat      = np.array([0., 0., 1.])
cross_dz   = np.cross(lor_dir, z_hat)                # (N,3)
cross_norm = np.linalg.norm(cross_dz, axis=1)
min_approach = np.where(
    cross_norm > 1e-6,
    np.abs(np.einsum('ij,ij->i', p1, cross_dz)) / cross_norm,
    np.sqrt(p1[:, 0]**2 + p1[:, 1]**2),   # LOR ∥ z-axis → use radial dist of P1
)

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 6 — Azimuthal opening angle  (degrees)
#   Angular separation between endpoint 1 and endpoint 2 in the XY plane.
#   Diametrically opposite → 180°.  Same-side → near 0°.
#   A LOR with opening_deg < 90° doesn't truly cross to the far side of the ring.
# ──────────────────────────────────────────────────────────────────────────────
phi1        = np.arctan2(p1[:, 1], p1[:, 0])
phi2        = np.arctan2(p2[:, 1], p2[:, 0])
dphi        = np.abs(np.arctan2(np.sin(phi2 - phi1), np.cos(phi2 - phi1)))
opening_deg = np.degrees(dphi)   # 0° = same direction, 180° = opposite

# ──────────────────────────────────────────────────────────────────────────────
# METRIC 7 — Axial span  |z2 - z1|  (mm)
#   Bounded by the axial FOV.  Very short spans correspond to nearly-transaxial
#   LORs (fine);  spans > AXIAL_LENGTH are impossible.
# ──────────────────────────────────────────────────────────────────────────────
z_span = np.abs(pred_z2 - pred_z1)

# ── Summary statistics & Pearson correlation with GT error ───────────────────
metrics_info = [
    ("r_dev_mean (mm)",    r_dev_mean,    "avg. ring deviation (both endpoints)"),
    ("r_dev_max (mm)",     r_dev_max,     "max ring deviation (worst endpoint)"),
    ("r_asym (mm)",        r_asym,        "radial asymmetry |r1-r2|"),
    ("lor_length (mm)",    lor_length,    "3-D chord length"),
    ("oblique_deg (°)",    oblique_deg,   "obliqueness from transaxial plane"),
    ("min_approach (mm)",  min_approach,  "min. distance of LOR to z-axis"),
    ("opening_deg (°)",    opening_deg,   "azimuthal opening angle"),
    ("z_span (mm)",        z_span,        "axial span |z2-z1|"),
]

print("\n── Pred-only quality metric statistics ───────────────────────────────────")
print(f"  {'Metric':<24} {'Mean':>8} {'Median':>8} {'Std':>8} {'P5':>7} {'P95':>8}")
print("  " + "─" * 66)
for name, vals, _ in metrics_info:
    print(f"  {name:<24} {vals.mean():8.2f} {np.median(vals):8.2f} "
          f"{vals.std():8.2f} {np.percentile(vals,5):7.2f} {np.percentile(vals,95):8.2f}")

print("\n── Pearson r with err_mean (GT-validated) ────────────────────────────────")
correlations = {}
for name, vals, desc in metrics_info:
    r = np.corrcoef(vals, err_mean)[0, 1]
    correlations[name] = r
    bar = "▓" * int(abs(r) * 30)
    sign = "+" if r >= 0 else "−"
    print(f"  {name:<24} {sign}{abs(r):.4f}  {bar:<30}  ({desc})")


# %%
# ==============================================================================
# FIGURE F1 — Each pred-only metric vs GT error  (scatter + binned median)
# ==============================================================================
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.suptitle(
    "Pred-only Quality Metrics vs GT Error\n"
    "(scatter = individual LORs, line = binned median ± IQR)",
    fontsize=13, fontweight='bold',
)
for ax, (name, vals, desc) in zip(axes.flat, metrics_info):
    _scatter_binned(ax, vals, err_mean, name, f"Error vs {name}", 'darkorange')
    r = correlations[name]
    ax.set_title(f"{name}\nr = {r:+.3f}", fontsize=10)
plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FILTER REGIME DEFINITIONS
# All masks use only pred-lor-derived quantities (no GT).
# True = keep the LOR.
# ==============================================================================
N = len(err_mean)
in_bounds  = ~oob_mask                                  # OOB already computed above

# --- Physical geometry thresholds (scanner-informed) -------------------------
# Ring proximity: worst endpoint within 2× crystal band of ring mid-line
_r_dev_thr_loose  = r_band * 2.0        # ≈ 42 mm
_r_dev_thr_tight  = r_band * 1.0        # ≈ 21 mm (within the crystal band itself)

# LOR length: plausible chord for this scanner
_len_lo = 100.0                          # mm — absolute minimum
_len_hi = 2 * OUTER_RADIUS * 1.1        # mm — 10% above max transaxial chord

# Approach distance: LOR must enter the inner bore (conservative) or well inside
_approach_inner   = INNER_RADIUS        # ≈ 235 mm — just crosses inner ring
_approach_fov     = INNER_RADIUS * 0.5  # ≈ 118 mm — clearly into FOV

# Opening angle: LOR must truly cross to the far side of the ring
_open_soft  = 90.0   # degrees
_open_tight = 120.0  # degrees

# Obliqueness
_oblique_soft  = 70.0  # degrees
_oblique_tight = 50.0  # degrees

# --- Population-percentile thresholds (data-driven) -------------------------
# Use 90th / 75th percentiles of each metric's pred population as cut-offs.
# These are unsupervised — we're simply removing statistical outliers.
_p90_rdev     = np.percentile(r_dev_max,    90)
_p75_rdev     = np.percentile(r_dev_max,    75)
_p90_rasym    = np.percentile(r_asym,       90)
_p75_rasym    = np.percentile(r_asym,       75)
_p90_approach = np.percentile(min_approach, 90)
_p75_approach = np.percentile(min_approach, 75)
_p5_len       = np.percentile(lor_length,    5)
_p95_len      = np.percentile(lor_length,   95)
_p10_len      = np.percentile(lor_length,   10)
_p90_len      = np.percentile(lor_length,   90)

REGIMES = {}

# R0 — No filter (baseline)
REGIMES["R0_baseline"] = np.ones(N, dtype=bool)

# R1 — OOB only  (existing hard physical bound)
REGIMES["R1_oob"] = in_bounds

# R2 — FOV transit  (LOR must pass within INNER_RADIUS of z-axis)
#      Removes same-side detector pairs and badly mislocalised endpoints.
REGIMES["R2_fov_transit"] = in_bounds & (min_approach < _approach_inner)

# R3 — Opening angle ≥ 90°
#      LOR must cross to the opposite half of the ring.
REGIMES["R3_opening90"] = in_bounds & (opening_deg >= _open_soft)

# R4 — Ring proximity  (r_dev_max < 2× band width)
#      Both endpoints must be plausibly on the detector ring.
REGIMES["R4_ring_prox"] = in_bounds & (r_dev_max < _r_dev_thr_loose)

# R5 — Percentile-90 outlier removal
#      Remove the worst 10% of r_dev_max, r_asym, and min_approach;
#      also trim the shortest 5% and longest 5% LOR lengths.
REGIMES["R5_pct90"] = (
    in_bounds
    & (r_dev_max    < _p90_rdev)
    & (r_asym       < _p90_rasym)
    & (min_approach < _p90_approach)
    & (lor_length   > _p5_len)
    & (lor_length   < _p95_len)
)

# R6 — Geometry moderate  (R1 + FOV transit + opening angle)
REGIMES["R6_geom_mod"] = in_bounds & (min_approach < _approach_fov) & (opening_deg >= _open_soft)

# R7 — Combined moderate  (R6 + ring proximity loose + pct-90)
REGIMES["R7_comb_mod"] = (
    REGIMES["R6_geom_mod"]
    & (r_dev_max < _r_dev_thr_loose)
    & (r_asym    < _p90_rasym)
    & (lor_length > _p5_len) & (lor_length < _p95_len)
)

# R8 — Combined strict  (tighter thresholds on every axis)
REGIMES["R8_comb_strict"] = (
    in_bounds
    & (min_approach < _approach_fov)
    & (opening_deg  >= _open_tight)
    & (r_dev_max    < _r_dev_thr_tight)
    & (r_asym       < _p75_rasym)
    & (lor_length   > _p10_len) & (lor_length < _p90_len)
    & (oblique_deg  < _oblique_tight)
)

# ── Print comparison table ────────────────────────────────────────────────────
baseline_mean   = err_mean.mean()
baseline_median = np.median(err_mean)

print("\n── Filter Regime Comparison ─────────────────────────────────────────────")
print(f"  {'Regime':<18} {'Kept%':>6} {'N kept':>8} {'Mean':>8} {'Δmean':>8} "
      f"{'Median':>8} {'Std':>7} {'P75':>7} {'P95':>8}")
print("  " + "─" * 84)

summary = []
for rname, mask in REGIMES.items():
    n_kept = mask.sum()
    pct    = 100 * n_kept / N
    sub    = err_mean[mask]
    row = dict(
        regime=rname, n_kept=n_kept, pct=pct,
        mean=sub.mean(), median=np.median(sub),
        std=sub.std(), p75=np.percentile(sub, 75), p95=np.percentile(sub, 95),
    )
    summary.append(row)
    delta_mean = row['mean'] - baseline_mean
    print(f"  {rname:<18} {pct:6.1f} {n_kept:8d} {row['mean']:8.3f} "
          f"{delta_mean:+8.3f} {row['median']:8.3f} {row['std']:7.3f} "
          f"{row['p75']:7.3f} {row['p95']:8.3f}")

print("\n  (Δmean = mean_filtered − baseline; negative = improvement)")
print(f"  Baseline: mean={baseline_mean:.3f}  median={baseline_median:.3f}")


# %%
# ==============================================================================
# FIGURE F2 — CDF overlay across all regimes
# ==============================================================================
fig, (ax_cdf, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("LOR Filter Regime Comparison", fontsize=14, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 0.9, len(REGIMES)))
for (rname, mask), c in zip(REGIMES.items(), colors):
    sub = np.sort(err_mean[mask])
    ax_cdf.plot(sub, np.arange(1, len(sub) + 1) / len(sub),
                linewidth=1.8, label=f"{rname} ({100*mask.mean():.0f}%)", color=c)
ax_cdf.set_xlabel("Mean endpoint error (mm)")
ax_cdf.set_ylabel("CDF")
ax_cdf.set_title("Cumulative Error Distribution")
ax_cdf.legend(fontsize=8, loc='lower right')
ax_cdf.grid(True, alpha=0.3)
ax_cdf.set_xlim(left=0)

# Bar chart: mean and median error per regime
x     = np.arange(len(summary))
width = 0.35
means   = [r['mean']   for r in summary]
medians = [r['median'] for r in summary]
labels  = [r['regime'] for r in summary]
pcts    = [r['pct']    for r in summary]

bars1 = ax_bar.bar(x - width/2, means,   width, label='Mean error',   color='tomato',    alpha=0.8)
bars2 = ax_bar.bar(x + width/2, medians, width, label='Median error', color='steelblue', alpha=0.8)
for bar, pct in zip(bars1, pcts):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{pct:.0f}%", ha='center', va='bottom', fontsize=7, color='dimgrey')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
ax_bar.set_ylabel("Error (mm)")
ax_bar.set_title("Mean & Median Error per Regime\n(% = fraction of LORs kept)")
ax_bar.legend(fontsize=9)
ax_bar.grid(True, alpha=0.3, axis='y')
ax_bar.axhline(baseline_mean, color='tomato', linestyle=':', linewidth=1, alpha=0.5)
ax_bar.axhline(baseline_median, color='steelblue', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FIGURE F3 — Error histograms per regime (3×3 grid)
# ==============================================================================
fig, axes = plt.subplots(3, 3, figsize=(21, 17))
fig.suptitle("Error Distribution per Filter Regime", fontsize=14, fontweight='bold')
for ax, (rname, mask), c in zip(axes.flat, REGIMES.items(), colors):
    sub    = err_mean[mask]
    n_kept = mask.sum()
    pct    = 100 * n_kept / N
    ax.hist(err_mean, bins=80, alpha=0.35, color='grey',  density=True, label='All LORs')
    ax.hist(sub,      bins=80, alpha=0.75, color=c,       density=True, label=f'Kept ({pct:.0f}%)')
    ax.axvline(np.median(sub), color='black', linewidth=1.5, linestyle='--',
               label=f'Median {np.median(sub):.2f} mm')
    ax.set_xlabel("Mean endpoint error (mm)")
    ax.set_ylabel("Density")
    ax.set_title(f"{rname}  |  kept={pct:.0f}%  ({n_kept})\n"
                 f"mean={sub.mean():.3f}  median={np.median(sub):.3f}  P95={np.percentile(sub,95):.3f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
axes.flat[-1].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FIGURE F4 — Spatial error heatmaps (R-Z) for selected regimes
#   Row 0: mean error per bin
#   Row 1: coverage (fraction of all LORs retained)
# Spatial coords use GT midpoint (proxy for true LOR origin).
# ==============================================================================
KEY_REGIMES = ["R0_baseline", "R1_oob", "R5_pct90", "R6_geom_mod", "R7_comb_mod", "R8_comb_strict"]
key_masks   = [REGIMES[k] for k in KEY_REGIMES]

r_range = [gt_radial.min(), gt_radial.max()]
z_range = [gt_axial.min(),  gt_axial.max()]
n_bins  = 30

h_cnt_all, re, ze = np.histogram2d(gt_radial, gt_axial, bins=n_bins,
                                    range=[r_range, z_range])

fig, axes = plt.subplots(2, len(KEY_REGIMES), figsize=(5 * len(KEY_REGIMES), 11))
fig.suptitle("Spatial Error & Coverage per Filter Regime (R-Z plane, GT midpoints)",
             fontsize=13, fontweight='bold')

# shared colour limits
all_err_vals = []
for mask in key_masks:
    h_e, _, _ = np.histogram2d(gt_radial[mask], gt_axial[mask], bins=n_bins,
                                range=[r_range, z_range], weights=err_mean[mask])
    h_c, _, _ = np.histogram2d(gt_radial[mask], gt_axial[mask], bins=n_bins,
                                range=[r_range, z_range])
    with np.errstate(invalid='ignore'):
        all_err_vals.append(np.where(h_c > 0, h_e / h_c, np.nan))
vmax_err = np.nanpercentile(np.concatenate([v.ravel() for v in all_err_vals]), 95)

for col, (rname, mask) in enumerate(zip(KEY_REGIMES, key_masks)):
    h_e, _, _ = np.histogram2d(gt_radial[mask], gt_axial[mask], bins=n_bins,
                                range=[r_range, z_range], weights=err_mean[mask])
    h_c, _, _ = np.histogram2d(gt_radial[mask], gt_axial[mask], bins=n_bins,
                                range=[r_range, z_range])
    with np.errstate(invalid='ignore'):
        h_mean_err = np.where(h_c > 0, h_e / h_c, np.nan)
        coverage   = np.where(h_cnt_all > 0, h_c / h_cnt_all, np.nan)

    im0 = axes[0, col].pcolormesh(re, ze, h_mean_err.T, cmap='hot_r',
                                   vmin=0, vmax=vmax_err)
    fig.colorbar(im0, ax=axes[0, col], label="Mean err (mm)", shrink=0.75, pad=0.02)
    axes[0, col].set_xlabel("Radial (mm)")
    axes[0, col].set_ylabel("Axial Z (mm)")
    short = rname.split('_', 1)[1]
    n_k   = mask.sum()
    axes[0, col].set_title(f"{short}\nerror  ({100*n_k/N:.0f}% LORs)", fontsize=9)

    im1 = axes[1, col].pcolormesh(re, ze, coverage.T, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im1, ax=axes[1, col], label="Coverage fraction", shrink=0.75, pad=0.02)
    axes[1, col].set_xlabel("Radial (mm)")
    axes[1, col].set_ylabel("Axial Z (mm)")
    axes[1, col].set_title(f"{short}\ncoverage", fontsize=9)

plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FIGURE F5 — Per-metric distribution for kept vs rejected LORs
#   For each regime of interest, show how each metric distributes across
#   kept (green) vs rejected (red) LORs alongside the err_mean distribution.
#   Helps confirm that the filter is actually capturing high-error LORs.
# ==============================================================================
INSPECT_REGIMES = ["R5_pct90", "R7_comb_mod", "R8_comb_strict"]

for rname in INSPECT_REGIMES:
    mask = REGIMES[rname]
    kept = mask
    rej  = ~mask

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        f"Kept vs Rejected LOR Distributions — {rname}  "
        f"(kept {100*kept.mean():.0f}% / rejected {100*rej.mean():.0f}%)",
        fontsize=13, fontweight='bold',
    )
    # First panel: err_mean (GT, for validation only)
    ax0 = axes[0, 0]
    ax0.hist(err_mean[kept], bins=60, alpha=0.6, color='seagreen', density=True, label='Kept')
    ax0.hist(err_mean[rej],  bins=60, alpha=0.6, color='tomato',   density=True, label='Rejected')
    ax0.set_xlabel("GT err_mean (mm)")
    ax0.set_title(f"GT Error  [validation only]\n"
                  f"kept mean={err_mean[kept].mean():.3f}  rej mean={err_mean[rej].mean():.3f}")
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

    # Remaining panels: pred-only metrics
    for ax, (name, vals, desc) in zip(axes.flat[1:], metrics_info):
        ax.hist(vals[kept], bins=60, alpha=0.6, color='seagreen', density=True, label='Kept')
        ax.hist(vals[rej],  bins=60, alpha=0.6, color='tomato',   density=True, label='Rejected')
        ax.set_xlabel(name)
        ax.set_title(f"{name}\nkept med={np.median(vals[kept]):.1f}  rej med={np.median(vals[rej]):.1f}")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Hide unused panels
    for ax in axes.flat[1 + len(metrics_info):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


# %%
# ==============================================================================
# FIGURE F6 — Quality–Retention trade-off curve
#   Sweeps a single scalar threshold (r_dev_max percentile) from 10% to 100%
#   to visualise the continuous quality–retention trade-off.
#   Also overlays the named regimes as labelled points.
# ==============================================================================
pct_range = np.arange(10, 101, 5)
sweep_means   = []
sweep_medians = []
sweep_p95     = []
sweep_retain  = []

for p in pct_range:
    thr  = np.percentile(r_dev_max, p)
    m    = in_bounds & (r_dev_max < thr)
    sub  = err_mean[m]
    sweep_means.append(sub.mean()       if m.any() else np.nan)
    sweep_medians.append(np.median(sub) if m.any() else np.nan)
    sweep_p95.append(np.percentile(sub, 95) if m.any() else np.nan)
    sweep_retain.append(m.mean() * 100)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Quality–Retention Trade-off  (sweep: r_dev_max percentile threshold)",
             fontsize=13, fontweight='bold')

ax_l.plot(sweep_retain, sweep_means,    'o-', color='tomato',     label='Mean error',   linewidth=2)
ax_l.plot(sweep_retain, sweep_medians,  's-', color='steelblue',  label='Median error', linewidth=2)
ax_l.plot(sweep_retain, sweep_p95,      '^-', color='darkorange', label='P95 error',    linewidth=1.5)
for row in summary:
    ax_l.scatter(row['pct'], row['mean'],   color='tomato',    s=60, zorder=5)
    ax_l.scatter(row['pct'], row['median'], color='steelblue', s=60, zorder=5)
    ax_l.annotate(row['regime'].split('_')[0], (row['pct'], row['mean']),
                  textcoords='offset points', xytext=(4, 4), fontsize=7, color='dimgrey')
ax_l.set_xlabel("% LORs retained")
ax_l.set_ylabel("Error (mm)")
ax_l.set_title("Error vs Retention  (all named regimes overlaid)")
ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)

ax_r.plot(sweep_retain, sweep_means,    'o-', color='tomato',    label='Mean error',   linewidth=2)
ax_r.plot(sweep_retain, sweep_medians,  's-', color='steelblue', label='Median error', linewidth=2)
ax_r.set_xlabel("% LORs retained")
ax_r.set_ylabel("Error (mm)")
ax_r.set_title("Error vs Retention  (mean + median only)")
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3)
for row in summary:
    ax_r.annotate(row['regime'].split('_')[0],
                  (row['pct'], row['mean']),
                  textcoords='offset points', xytext=(4, 4), fontsize=8,
                  color='dimgrey', arrowprops=dict(arrowstyle='->', color='dimgrey', lw=0.8))

plt.tight_layout()
plt.show()

# ── Final regime recommendation summary ──────────────────────────────────────
print("\n" + "=" * 70)
print("LOR FILTERING REGIME RECOMMENDATION SUMMARY")
print("=" * 70)
print(f"\n  {'Regime':<18}  {'Kept%':>5}  {'Mean':>7}  {'Median':>8}  {'P95':>7}  verdict")
print("  " + "─" * 65)
for row in summary:
    dm = row['mean'] - baseline_mean
    verdict = ("✓✓ best" if dm < -1.0 and row['pct'] > 50 else
               "✓ good"  if dm < -0.5 and row['pct'] > 60 else
               "✗ lossy" if row['pct'] < 40 else
               "~ mild")
    print(f"  {row['regime']:<18}  {row['pct']:5.1f}  {row['mean']:7.3f}  "
          f"{row['median']:8.3f}  {row['p95']:7.3f}  {verdict}")
print("=" * 70)


# %%
# ==============================================================================
# SECTION 6 — OBLIQUENESS / AXIAL-SPAN TARGETED FILTERING
# ==============================================================================
# z_span (r=+0.204) and oblique_deg (r=+0.154) are the dominant error
# predictors — 5–60× stronger than any geometry-based metric.
# This section sweeps those two axes individually, then jointly, to find
# the best threshold(s) and shows where the error "knee" sits.
# ==============================================================================

# ── 1-D sweeps ────────────────────────────────────────────────────────────────
# oblique_deg thresholds: 5° to 90° in 2° steps
oblique_thresholds = np.arange(5, 91, 2, dtype=float)
# z_span thresholds: 10 mm to 300 mm in 5 mm steps
zspan_thresholds   = np.arange(10, 301, 5, dtype=float)

def _sweep_1d(thresholds, metric, base_mask):
    """Return (retain_pct, mean_err, median_err, p75_err, p95_err) arrays."""
    retain, means, medians, p75s, p95s = [], [], [], [], []
    for thr in thresholds:
        m   = base_mask & (metric < thr)
        sub = err_mean[m]
        if m.sum() < 10:
            retain.append(np.nan); means.append(np.nan)
            medians.append(np.nan); p75s.append(np.nan); p95s.append(np.nan)
        else:
            retain.append(100 * m.mean())
            means.append(sub.mean())
            medians.append(np.median(sub))
            p75s.append(np.percentile(sub, 75))
            p95s.append(np.percentile(sub, 95))
    return (np.array(retain), np.array(means),
            np.array(medians), np.array(p75s), np.array(p95s))

ob_retain, ob_mean, ob_median, ob_p75, ob_p95 = _sweep_1d(oblique_thresholds, oblique_deg, in_bounds)
zs_retain, zs_mean, zs_median, zs_p75, zs_p95 = _sweep_1d(zspan_thresholds,   z_span,      in_bounds)

# ── 2-D joint sweep ───────────────────────────────────────────────────────────
# Grid search over oblique_deg × z_span; record mean error and retention.
OB_GRID = np.arange(10, 71, 5, dtype=float)
ZS_GRID = np.arange(30, 251, 10, dtype=float)
grid_mean    = np.full((len(OB_GRID), len(ZS_GRID)), np.nan)
grid_retain  = np.full((len(OB_GRID), len(ZS_GRID)), np.nan)

for i, ob_thr in enumerate(OB_GRID):
    for j, zs_thr in enumerate(ZS_GRID):
        m = in_bounds & (oblique_deg < ob_thr) & (z_span < zs_thr)
        if m.sum() >= 10:
            grid_mean[i, j]   = err_mean[m].mean()
            grid_retain[i, j] = 100 * m.mean()


# %%
# ==============================================================================
# FIGURE F7 — 1-D sweep: oblique_deg threshold
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("1-D Threshold Sweep — oblique_deg  (OOB filter always applied)",
             fontsize=13, fontweight='bold')

# Left: error vs threshold value
axes[0].plot(oblique_thresholds, ob_mean,   'o-', color='tomato',    linewidth=2, ms=3, label='Mean error')
axes[0].plot(oblique_thresholds, ob_median, 's-', color='steelblue', linewidth=2, ms=3, label='Median error')
axes[0].plot(oblique_thresholds, ob_p75,    '^-', color='darkorange', linewidth=1.5, ms=3, label='P75 error')
axes[0].axhline(baseline_mean,   color='tomato',    linestyle=':', alpha=0.5, label='Baseline mean')
axes[0].axhline(baseline_median, color='steelblue', linestyle=':', alpha=0.5, label='Baseline median')
axes[0].set_xlabel("oblique_deg threshold (°)")
axes[0].set_ylabel("Error (mm)")
axes[0].set_title("Error vs oblique_deg threshold")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Middle: retention vs threshold value
axes[1].plot(oblique_thresholds, ob_retain, 'o-', color='seagreen', linewidth=2, ms=3)
axes[1].set_xlabel("oblique_deg threshold (°)")
axes[1].set_ylabel("% LORs retained")
axes[1].set_title("Retention vs oblique_deg threshold")
axes[1].grid(True, alpha=0.3)
for pct_tgt in [50, 60, 70, 80, 90]:
    idx = np.searchsorted(ob_retain, pct_tgt)
    if 0 < idx < len(oblique_thresholds):
        axes[1].axhline(pct_tgt, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

# Right: error vs retention (the trade-off curve)
axes[2].plot(ob_retain, ob_mean,   'o-', color='tomato',    linewidth=2, ms=3, label='Mean')
axes[2].plot(ob_retain, ob_median, 's-', color='steelblue', linewidth=2, ms=3, label='Median')
# Annotate a few threshold values along the curve
for thr in [15, 20, 25, 30, 40, 50]:
    idx = np.argmin(np.abs(oblique_thresholds - thr))
    if not np.isnan(ob_mean[idx]):
        axes[2].annotate(f"{thr}°",
                         (ob_retain[idx], ob_mean[idx]),
                         textcoords='offset points', xytext=(4, 3),
                         fontsize=7, color='dimgrey')
axes[2].axhline(baseline_mean,   color='tomato',    linestyle=':', alpha=0.5)
axes[2].axhline(baseline_median, color='steelblue', linestyle=':', alpha=0.5)
axes[2].set_xlabel("% LORs retained")
axes[2].set_ylabel("Error (mm)")
axes[2].set_title("Quality–Retention curve  (oblique_deg sweep)")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FIGURE F8 — 1-D sweep: z_span threshold
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("1-D Threshold Sweep — z_span  (OOB filter always applied)",
             fontsize=13, fontweight='bold')

axes[0].plot(zspan_thresholds, zs_mean,   'o-', color='tomato',    linewidth=2, ms=3, label='Mean error')
axes[0].plot(zspan_thresholds, zs_median, 's-', color='steelblue', linewidth=2, ms=3, label='Median error')
axes[0].plot(zspan_thresholds, zs_p75,    '^-', color='darkorange', linewidth=1.5, ms=3, label='P75 error')
axes[0].axhline(baseline_mean,   color='tomato',    linestyle=':', alpha=0.5, label='Baseline mean')
axes[0].axhline(baseline_median, color='steelblue', linestyle=':', alpha=0.5, label='Baseline median')
axes[0].set_xlabel("z_span threshold (mm)")
axes[0].set_ylabel("Error (mm)")
axes[0].set_title("Error vs z_span threshold")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(zspan_thresholds, zs_retain, 'o-', color='seagreen', linewidth=2, ms=3)
axes[1].set_xlabel("z_span threshold (mm)")
axes[1].set_ylabel("% LORs retained")
axes[1].set_title("Retention vs z_span threshold")
axes[1].grid(True, alpha=0.3)

axes[2].plot(zs_retain, zs_mean,   'o-', color='tomato',    linewidth=2, ms=3, label='Mean')
axes[2].plot(zs_retain, zs_median, 's-', color='steelblue', linewidth=2, ms=3, label='Median')
for thr in [50, 75, 100, 125, 150, 200]:
    idx = np.argmin(np.abs(zspan_thresholds - thr))
    if not np.isnan(zs_mean[idx]):
        axes[2].annotate(f"{thr}mm",
                         (zs_retain[idx], zs_mean[idx]),
                         textcoords='offset points', xytext=(4, 3),
                         fontsize=7, color='dimgrey')
axes[2].axhline(baseline_mean,   color='tomato',    linestyle=':', alpha=0.5)
axes[2].axhline(baseline_median, color='steelblue', linestyle=':', alpha=0.5)
axes[2].set_xlabel("% LORs retained")
axes[2].set_ylabel("Error (mm)")
axes[2].set_title("Quality–Retention curve  (z_span sweep)")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FIGURE F9 — 2-D joint sweep heatmaps
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("2-D Joint Sweep: oblique_deg × z_span threshold\n"
             "(OOB filter always applied)",
             fontsize=13, fontweight='bold')

# Mean error heatmap
vmin_e = np.nanmin(grid_mean)
vmax_e = np.nanpercentile(grid_mean[~np.isnan(grid_mean)], 95)
im0 = axes[0].pcolormesh(ZS_GRID, OB_GRID, grid_mean,
                          cmap='hot_r', vmin=vmin_e, vmax=vmax_e)
fig.colorbar(im0, ax=axes[0], label="Mean error (mm)")
axes[0].set_xlabel("z_span threshold (mm)")
axes[0].set_ylabel("oblique_deg threshold (°)")
axes[0].set_title("Mean error — lower is better")

# Retention heatmap
im1 = axes[1].pcolormesh(ZS_GRID, OB_GRID, grid_retain,
                          cmap='Blues', vmin=0, vmax=100)
fig.colorbar(im1, ax=axes[1], label="% LORs retained")
axes[1].set_xlabel("z_span threshold (mm)")
axes[1].set_ylabel("oblique_deg threshold (°)")
axes[1].set_title("% LORs retained — higher is better")

# Overlay retention contours on the error map
cs = axes[0].contour(ZS_GRID, OB_GRID, grid_retain,
                      levels=[40, 50, 60, 70, 80],
                      colors='white', linewidths=0.8, alpha=0.7)
axes[0].clabel(cs, fmt="%d%%", fontsize=7)

plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# FIGURE F10 — Binned median with error band: the "knee" plots
#   Shows error as a function of the metric value (not a threshold sweep).
#   The knee in the binned median tells you where to cut.
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Error vs Metric Value — Knee Detection\n"
             "(binned median ± IQR; dashed = suggested cut)",
             fontsize=13, fontweight='bold')

for ax, (metric, name, color, suggested_cut) in zip(axes, [
    (oblique_deg, "oblique_deg (°)",  'darkorange', None),
    (z_span,      "z_span (mm)",      'mediumpurple', None),
]):
    # scatter
    ax.scatter(metric, err_mean, s=0.3, alpha=0.08, color='grey', rasterized=True)

    # binned median ± IQR ± P90
    edges = np.percentile(metric, np.linspace(0, 100, 31))
    edges = np.unique(edges)
    ctrs, meds, q25s, q75s, q90s, counts = [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (metric >= lo) & (metric < hi)
        if m.sum() > 5:
            v = err_mean[m]
            ctrs.append((lo + hi) / 2)
            meds.append(np.median(v))
            q25s.append(np.percentile(v, 25))
            q75s.append(np.percentile(v, 75))
            q90s.append(np.percentile(v, 90))
            counts.append(m.sum())
    ctrs, meds, q25s, q75s, q90s = map(np.array, [ctrs, meds, q25s, q75s, q90s])

    ax.fill_between(ctrs, q25s, q75s, alpha=0.25, color=color, label='IQR')
    ax.fill_between(ctrs, q75s, q90s, alpha=0.12, color=color, label='P75–P90')
    ax.plot(ctrs, meds, 'o-', color=color, linewidth=2.5, ms=4, label='Median error')

    # Annotate where median rises above 110% and 125% of its own minimum
    med_min = meds.min()
    for factor, style, label in [(1.10, '--', '+10% above min'), (1.25, ':', '+25% above min')]:
        idx = np.argmax(meds > med_min * factor)
        if idx > 0:
            cut_val = ctrs[idx]
            ax.axvline(cut_val, color=color, linestyle=style, linewidth=1.5,
                       label=f'{label}: {cut_val:.1f}')

    ax.axhline(baseline_mean, color='grey', linestyle='--', linewidth=1,
               alpha=0.6, label=f'Baseline mean ({baseline_mean:.2f})')
    ax.set_xlabel(name)
    ax.set_ylabel("Mean endpoint error (mm)")
    ax.set_title(f"Error vs {name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# Targeted filter comparison: oblique_deg + z_span cuts at data-driven knees
# ==============================================================================
# Read off suggested cuts from the knee plot and test a few combinations.

# Knee-informed single cuts (edit these after looking at F10)
OB_CUT_SOFT   = 20.0   # °  — roughly where median starts climbing
OB_CUT_TIGHT  = 15.0   # °  — tighter, higher error reduction
ZS_CUT_SOFT   = 130.0  # mm
ZS_CUT_TIGHT  = 80.0   # mm

TARGET_REGIMES = {
    "T0_baseline"      : in_bounds,
    "T1_ob_soft"       : in_bounds & (oblique_deg < OB_CUT_SOFT),
    "T2_ob_tight"      : in_bounds & (oblique_deg < OB_CUT_TIGHT),
    "T3_zs_soft"       : in_bounds & (z_span      < ZS_CUT_SOFT),
    "T4_zs_tight"      : in_bounds & (z_span      < ZS_CUT_TIGHT),
    "T5_ob_zs_soft"    : in_bounds & (oblique_deg < OB_CUT_SOFT)  & (z_span < ZS_CUT_SOFT),
    "T6_ob_zs_tight"   : in_bounds & (oblique_deg < OB_CUT_TIGHT) & (z_span < ZS_CUT_TIGHT),
}

print("\n── Targeted (oblique + z_span) Filter Comparison ────────────────────────")
print(f"  {'Regime':<20} {'Kept%':>6} {'N kept':>8} {'Mean':>8} {'Δmean':>8} "
      f"{'Median':>8} {'P95':>8}")
print("  " + "─" * 74)
for rname, mask in TARGET_REGIMES.items():
    sub = err_mean[mask]
    pct = 100 * mask.mean()
    dm  = sub.mean() - baseline_mean
    print(f"  {rname:<20} {pct:6.1f} {mask.sum():8d} {sub.mean():8.3f} "
          f"{dm:+8.3f} {np.median(sub):8.3f} {np.percentile(sub,95):8.3f}")
print(f"\n  Cuts used:  oblique_deg < {OB_CUT_SOFT}° (soft) / {OB_CUT_TIGHT}° (tight)")
print(f"              z_span      < {ZS_CUT_SOFT} mm (soft) / {ZS_CUT_TIGHT} mm (tight)")
print("  Adjust OB_CUT_* and ZS_CUT_* based on the knees visible in Figure F10.")
