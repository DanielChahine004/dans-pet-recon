"""Compare FWHM between GT continuous, GT binned, and Pred continuous reconstructions.

Loads GT continuous + binned reconstructions and Pred continuous reconstruction
across all point source positions. Computes Gaussian-fit FWHM and generates
comparison plots.
"""
import numpy as np
import pickle
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Processed"))
# BINNED_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Binned_new(ufb)"))
BINNED_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Binned_new(osem)"))
OUTPUT_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/binned_comparison_new(osem)"))
SKIP_DIRS = {"outside_fov", "pdf_outputs"}
BLUR_SIGMA = 1
BLUR_RADIUS = 1
MIN_R_SQUARED = 0.8     # was 0.5, discards Gaussian fits with R² below this

BINNING_LABELS = ["very_fine_500x50", "fine_400x40", "moderate_300x30", "standard_200x20", "coarse_100x10"]
# BINNING_LABELS = ["very_fine_500x50", "fine_400x40", "moderate_300x30"]
BINNING_DISPLAY = {
    "very_fine_500x50": "Very Fine (500x50, ~2.96mm)",
    "fine_400x40": "Fine (400x40, ~3.70mm)",
    "moderate_300x30": "Moderate (300x30, ~4.93mm)",
    "standard_200x20": "Standard (200x20, ~7.4mm)",
    "coarse_100x10": "Coarse (100x10, ~14.8mm)",
}


def load_recon(output_dir):
    """Load a reconstruction image and config. Returns (image, config) or (None, None)."""
    config_path = output_dir / "recon_config.pkl"
    recon_path = output_dir / "reconstruction_image.npy"
    if not config_path.exists() or not recon_path.exists():
        return None, None
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return np.load(recon_path), config


def fit_line_profile(image, axis_name, coord1, coord2, voxel_size):
    """Gaussian fit on a line profile through the hottest voxel. Returns FWHM in mm or None."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_int = axis_map[axis_name]

    if axis_int == 0:
        line = image[:, coord1, coord2]
    elif axis_int == 1:
        line = image[coord1, :, coord2]
    else:
        line = image[coord1, coord2, :]

    n = len(line)
    x_data = np.arange(n)

    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + offset

    offset_guess = np.min(line)
    amplitude_guess = np.max(line) - offset_guess
    mean_guess = np.argmax(line)
    sigma_guess = n / 10
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    try:
        popt, pcov = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
        fitted = gaussian(x_data, *popt)
        ss_res = np.sum((line - fitted) ** 2)
        ss_tot = np.sum((line - np.mean(line)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r_squared < MIN_R_SQUARED:
            return None
        return 2.355 * abs(popt[2]) * voxel_size
    except RuntimeError:
        return None


def fit_line_profile_for_plot(image, axis_name, coord1, coord2, voxel_size):
    """Like fit_line_profile but returns (fwhm_mm, x_mm, raw_vals, fit_vals), or None on failure."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_int = axis_map[axis_name]

    if axis_int == 0:
        line = image[:, coord1, coord2]
    elif axis_int == 1:
        line = image[coord1, :, coord2]
    else:
        line = image[coord1, coord2, :]

    n = len(line)
    x_data = np.arange(n)
    x_mm = x_data * voxel_size

    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) + offset

    offset_guess = np.min(line)
    amplitude_guess = np.max(line) - offset_guess
    mean_guess = np.argmax(line)
    sigma_guess = n / 10
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    try:
        popt, _ = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
        fitted = gaussian(x_data, *popt)
        ss_res = np.sum((line - fitted) ** 2)
        ss_tot = np.sum((line - np.mean(line)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r_squared < MIN_R_SQUARED:
            return None
        return 2.355 * abs(popt[2]) * voxel_size, x_mm, line, fitted
    except RuntimeError:
        return None


def get_fwhm_xyz(image, hotspot, voxel_size):
    """Get FWHM along x, y, z axes. Returns dict or None if all fits fail."""
    sx, sy, sz = hotspot
    results = {}
    for axis_name, c1, c2 in [("x", sy, sz), ("y", sx, sz), ("z", sx, sy)]:
        fwhm = fit_line_profile(image, axis_name, c1, c2, voxel_size)
        if fwhm is not None:
            results[axis_name] = fwhm
    if not results:
        return None
    results["avg"] = np.mean(list(results.values()))
    return results


def collect_data():
    """Collect FWHM for GT continuous, GT binned, and Pred continuous per position."""
    positions = sorted(
        [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name not in SKIP_DIRS],
        key=lambda d: tuple(int(x) for x in d.name.split("_")),
    )

    records = []
    for pos_dir in positions:
        pos_name = pos_dir.name
        parts = pos_name.split("_")
        x_mm, y_mm, z_mm = int(parts[0]), int(parts[1]), int(parts[2])
        radial_mm = np.sqrt(x_mm ** 2 + y_mm ** 2)

        # --- GT continuous ---
        gt_img, gt_cfg = load_recon(pos_dir / "gt")
        if gt_img is None:
            print(f"  SKIP {pos_name}: no GT continuous reconstruction")
            continue

        voxel_size = gt_cfg["VOXEL_SIZE_IN_MM"]
        gt_blurred = gaussian_filter(gt_img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
        hotspot = np.unravel_index(np.argmax(gt_blurred), gt_blurred.shape)

        gt_cont_fwhm = get_fwhm_xyz(gt_blurred, hotspot, voxel_size)
        if gt_cont_fwhm is None:
            print(f"  SKIP {pos_name}: GT continuous FWHM fit failed")
            continue

        rec = {
            "name": pos_name,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "radial_mm": radial_mm,
            "gt_continuous_fwhm_avg": gt_cont_fwhm["avg"],
            "gt_continuous_fwhm_x": gt_cont_fwhm.get("x"),
            "gt_continuous_fwhm_y": gt_cont_fwhm.get("y"),
            "gt_continuous_fwhm_z": gt_cont_fwhm.get("z"),
        }

        # --- GT binned ---
        for label in BINNING_LABELS:
            binned_img, _ = load_recon(BINNED_DIR / pos_name / f"gt_binned_{label}")
            if binned_img is None:
                rec[f"gt_{label}_fwhm_avg"] = None
                continue
            binned_blurred = gaussian_filter(binned_img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
            binned_fwhm = get_fwhm_xyz(binned_blurred, hotspot, voxel_size)
            if binned_fwhm is None:
                rec[f"gt_{label}_fwhm_avg"] = None
                continue
            rec[f"gt_{label}_fwhm_avg"] = binned_fwhm["avg"]
            rec[f"gt_{label}_fwhm_x"] = binned_fwhm.get("x")
            rec[f"gt_{label}_fwhm_y"] = binned_fwhm.get("y")
            rec[f"gt_{label}_fwhm_z"] = binned_fwhm.get("z")

        # --- Pred continuous ---
        pred_img, pred_cfg = load_recon(pos_dir / "pred")
        if pred_img is None:
            print(f"  WARN {pos_name}: no Pred continuous reconstruction")
            rec["pred_continuous_fwhm_avg"] = None
        else:
            pred_voxel_size = pred_cfg["VOXEL_SIZE_IN_MM"]
            pred_blurred = gaussian_filter(pred_img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
            pred_fwhm = get_fwhm_xyz(pred_blurred, hotspot, pred_voxel_size)
            if pred_fwhm is None:
                print(f"  WARN {pos_name}: Pred continuous FWHM fit failed")
                rec["pred_continuous_fwhm_avg"] = None
            else:
                rec["pred_continuous_fwhm_avg"] = pred_fwhm["avg"]
                rec["pred_continuous_fwhm_x"] = pred_fwhm.get("x")
                rec["pred_continuous_fwhm_y"] = pred_fwhm.get("y")
                rec["pred_continuous_fwhm_z"] = pred_fwhm.get("z")

        records.append(rec)

        line = f"  {pos_name}: gt_cont={gt_cont_fwhm['avg']:.2f}mm"
        for label in BINNING_LABELS:
            val = rec.get(f"gt_{label}_fwhm_avg")
            if val is not None:
                line += f", gt_{label.split('_')[0]}={val:.2f}mm"
        if rec.get("pred_continuous_fwhm_avg") is not None:
            line += f", pred_cont={rec['pred_continuous_fwhm_avg']:.2f}mm"
        print(line)

    return records


# Series definitions for summary plots: (record_key, display label, color, marker, linestyle)
SERIES = [
    ("gt_continuous_fwhm_avg", "GT Continuous", "#1565C0", "o", "-"),
] + [
    (f"gt_{label}_fwhm_avg", f"GT {BINNING_DISPLAY[label]}", color, marker, ls)
    for label, color, marker, ls in zip(
        BINNING_LABELS,
        ["#42A5F5", "#26C6DA", "#66BB6A", "#FFA726", "#EF5350"],
        ["s", "D", "^", "v", "P"],
        ["--", "--", "--", "--", "--"],
    )
] + [
    ("pred_continuous_fwhm_avg", "Pred Continuous", "#6A1B9A", "X", "-"),
]

# Config list for per-position figures: (subdir_name, display_label, color)
POSITION_CONFIGS = [
    ("gt", "GT Continuous", "#1565C0"),
] + [
    (f"gt_binned_{label}", f"GT {BINNING_DISPLAY[label]}", color)
    for label, color in zip(
        BINNING_LABELS,
        ["#42A5F5", "#26C6DA", "#66BB6A", "#FFA726", "#EF5350"],
    )
] + [
    ("pred", "Pred Continuous", "#6A1B9A"),
]


def _summary_bar_panel(ax, valid, value_fn, title):
    labels_disp, means, stds, colors_list, counts = [], [], [], [], []
    for key, display, color, *_ in SERIES:
        vals = [v for r in valid if (v := value_fn(r, key)) is not None]
        if not vals:
            continue
        labels_disp.append(display)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        colors_list.append(color)
        counts.append(len(vals))

    x = np.arange(len(labels_disp))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_list,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_disp, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("FWHM (mm)")
    ax.set_title(f"Mean {title}\n(blurred σ={BLUR_SIGMA})", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{m:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)


def plot_summary_all(records, output_dir):
    """Single PDF with three summary bar panels: avg, transaxial (x-y), and axial (z) FWHM."""
    valid = [r for r in records if r.get("gt_continuous_fwhm_avg") is not None]
    if not valid:
        return

    def get_avg(r, key):
        return r.get(key)

    def get_xy(r, key):
        base = key.replace("_avg", "")
        vals = [v for v in [r.get(base + "_x"), r.get(base + "_y")] if v is not None]
        return np.mean(vals) if vals else None

    def get_z(r, key):
        return r.get(key.replace("_avg", "_z"))

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    _summary_bar_panel(axes[0], valid, get_avg, "Average FWHM")
    _summary_bar_panel(axes[1], valid, get_xy, "Transaxial (x-y) FWHM")
    _summary_bar_panel(axes[2], valid, get_z, "Axial (z) FWHM")
    fig.tight_layout()
    fig.savefig(output_dir / "fwhm_summary.pdf", format="pdf", bbox_inches="tight")
    plt.close("all")


def plot_per_position_figures(output_dir):
    """For each point source, save a PDF with transaxial/axial slices and line profiles per config."""
    pos_out = output_dir / "positions"
    pos_out.mkdir(parents=True, exist_ok=True)

    positions = sorted(
        [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name not in SKIP_DIRS],
        key=lambda d: tuple(int(x) for x in d.name.split("_")),
    )

    for pos_dir in positions:
        pos_name = pos_dir.name
        print(f"  Plotting position {pos_name}...")

        hotspot = None
        loaded = []

        for subdir, label, color in POSITION_CONFIGS:
            base = BINNED_DIR / pos_name if "binned" in subdir else pos_dir
            img, cfg = load_recon(base / subdir)
            if img is None:
                continue
            vs = cfg["VOXEL_SIZE_IN_MM"]
            blurred = gaussian_filter(img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
            if subdir == "gt":
                hotspot = np.unravel_index(np.argmax(blurred), blurred.shape)
            loaded.append((label, color, blurred, vs))

        if hotspot is None or not loaded:
            print(f"    SKIP {pos_name}: no GT or no data")
            continue

        sx, sy, sz = hotspot
        n_configs = len(loaded)
        fig, axes = plt.subplots(n_configs, 4, figsize=(16, 3.2 * n_configs), squeeze=False)
        fig.suptitle(f"Position {pos_name}  |  GT hotspot voxel ({sx}, {sy}, {sz})", fontsize=11)

        for row, (label, color, blurred, vs) in enumerate(loaded):
            nx, ny, nz = blurred.shape
            half_x = nx * vs / 2
            half_y = ny * vs / 2
            half_z = nz * vs / 2
            # Hotspot in mm (centred), matching plot_static convention
            hx_mm = (sx - (nx - 1) / 2) * vs
            hy_mm = (sy - (ny - 1) / 2) * vs
            hz_mm = (sz - (nz - 1) / 2) * vs

            # Col 0: transaxial slice — same as plot_static(axis=2, depth=sz)
            axes[row, 0].imshow(
                blurred[:, :, sz].T, cmap="hot", origin="lower",
                extent=[-half_x, half_x, -half_y, half_y], aspect="equal",
            )
            axes[row, 0].axhline(hy_mm, color="cyan", linewidth=0.8, alpha=0.7)
            axes[row, 0].axvline(hx_mm, color="cyan", linewidth=0.8, alpha=0.7)
            axes[row, 0].set_title(f"{label}\nTransaxial (z={sz})", fontsize=7)
            axes[row, 0].set_xlabel("X (mm)", fontsize=6)
            axes[row, 0].set_ylabel("Y (mm)", fontsize=6)
            axes[row, 0].tick_params(labelsize=6)

            # Col 1: axial slice — same as plot_static(axis=1, depth=sy)
            axes[row, 1].imshow(
                blurred[:, sy, :].T, cmap="hot", origin="lower",
                extent=[-half_x, half_x, -half_z, half_z], aspect="equal",
            )
            axes[row, 1].axhline(hz_mm, color="cyan", linewidth=0.8, alpha=0.7)
            axes[row, 1].axvline(hx_mm, color="cyan", linewidth=0.8, alpha=0.7)
            axes[row, 1].set_title(f"Axial (y={sy})", fontsize=7)
            axes[row, 1].set_xlabel("X (mm)", fontsize=6)
            axes[row, 1].set_ylabel("Z (mm)", fontsize=6)
            axes[row, 1].tick_params(labelsize=6)

            # Col 2: transaxial line profiles (x and y overlaid)
            ax_tp = axes[row, 2]
            for axis_name, c1, c2, lp_color, lp_label in [
                ("x", sy, sz, "#E53935", "x"),
                ("y", sx, sz, "#43A047", "y"),
            ]:
                res = fit_line_profile_for_plot(blurred, axis_name, c1, c2, vs)
                if res is not None:
                    fwhm, x_mm, raw, fitted = res
                    ax_tp.plot(x_mm, raw, color=lp_color, alpha=0.45, linewidth=1)
                    ax_tp.plot(x_mm, fitted, color=lp_color, linewidth=1.5, linestyle="--",
                               label=f"{lp_label}: {fwhm:.2f}mm")
                else:
                    axis_int = {"x": 0, "y": 1}[axis_name]
                    line = blurred[:, c1, c2] if axis_int == 0 else blurred[c1, :, c2]
                    ax_tp.plot(np.arange(len(line)) * vs, line, color=lp_color, alpha=0.45,
                               linewidth=1, label=f"{lp_label} (no fit)")
            ax_tp.set_title("Transaxial profiles", fontsize=7)
            ax_tp.set_xlabel("position (mm)", fontsize=6)
            ax_tp.set_ylabel("intensity", fontsize=6)
            ax_tp.legend(fontsize=6)
            ax_tp.grid(True, alpha=0.3)
            ax_tp.tick_params(labelsize=6)

            # Col 3: axial (z) line profile
            ax_zp = axes[row, 3]
            res_z = fit_line_profile_for_plot(blurred, "z", sx, sy, vs)
            if res_z is not None:
                fwhm_z, z_mm, raw_z, fitted_z = res_z
                ax_zp.plot(z_mm, raw_z, color="#1565C0", alpha=0.45, linewidth=1)
                ax_zp.plot(z_mm, fitted_z, color="#1565C0", linewidth=1.5, linestyle="--",
                           label=f"z: {fwhm_z:.2f}mm")
            else:
                line_z = blurred[sx, sy, :]
                ax_zp.plot(np.arange(len(line_z)) * vs, line_z, color="#1565C0", alpha=0.45,
                           linewidth=1, label="z (no fit)")
            ax_zp.set_title("Axial (z) profile", fontsize=7)
            ax_zp.set_xlabel("z position (mm)", fontsize=6)
            ax_zp.set_ylabel("intensity", fontsize=6)
            ax_zp.legend(fontsize=6)
            ax_zp.grid(True, alpha=0.3)
            ax_zp.tick_params(labelsize=6)

        fig.tight_layout()
        fig.savefig(pos_out / f"{pos_name}.pdf", format="pdf", bbox_inches="tight")
        plt.close("all")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting FWHM measurements...\n")
    records = collect_data()

    if not records:
        print("No data to plot.")
        return

    print(f"\nCollected {len(records)} records. Generating summary plots...\n")
    plot_summary_all(records, OUTPUT_DIR)

    print("Generating per-position figures...")
    plot_per_position_figures(OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
