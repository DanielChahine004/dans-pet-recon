"""Compare average FWHM between unfiltered and radially-filtered LOR reconstructions.

Loads pred reconstructions from Point_Sources_Processed (unfiltered) and
Point_Sources_r235_to_257mm (filtered), computes Gaussian-fit FWHM averaged
along the z-axis through the GT hottest voxel, and plots delta FWHM vs position.
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
UNFILTERED_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Processed"))
FILTERED_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_r235_to_257mm"))
OUTPUT_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/filtered_comparison_r235_to_257mm"))
# FILTERED_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_polar_max50deg"))
# OUTPUT_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/filtered_comparison_polar_max50deg"))

BLUR_SIGMA = 1
BLUR_RADIUS = 1
R2_MIN = 0.9


def load_recon(output_dir: Path):
    """Load a reconstruction image and config. Returns (recon_image, config) or (None, None)."""
    config_path = output_dir / "recon_config.pkl"
    recon_path = output_dir / "reconstruction_image.npy"
    if not config_path.exists() or not recon_path.exists():
        return None, None
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    recon_image = np.load(recon_path)
    return recon_image, config


def fit_line_profile(image, axis_name, coord1, coord2, voxel_size):
    """Run Gaussian fit on a line profile through the hottest voxel. Returns FWHM in mm or None."""
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

    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset

    offset_guess = np.min(line)
    amplitude_guess = np.max(line) - offset_guess
    mean_guess = np.argmax(line)
    sigma_guess = n / 10
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    try:
        popt, _ = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
        residuals = line - gaussian(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((line - np.mean(line))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 < R2_MIN:
            return None
        fwhm_mm = 2.355 * abs(popt[2]) * voxel_size
        return fwhm_mm
    except RuntimeError:
        return None


def get_z_fwhm(image, hotspot, voxel_size):
    """Get FWHM along the z-axis through the given hotspot. Returns FWHM in mm or None."""
    sx, sy, _ = hotspot
    return fit_line_profile(image, "z", sx, sy, voxel_size)


def get_z_profile_data(image, hotspot, voxel_size):
    """Extract z-axis line profile and Gaussian fit through the hotspot.

    Returns a dict with keys: line, x_mm, fwhm_mm, x_fit_mm, y_fit, r2.
    Returns None if fit fails or R² < R2_MIN.
    """
    sx, sy, _ = hotspot
    line = image[sx, sy, :]
    n = len(line)
    x_data = np.arange(n)
    x_mm = (x_data - (n - 1) / 2) * voxel_size

    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset

    offset_guess = np.min(line)
    amplitude_guess = np.max(line) - offset_guess
    mean_guess = float(np.argmax(line))
    sigma_guess = n / 10
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    try:
        popt, _ = curve_fit(gaussian, x_data, line, p0=p0, maxfev=5000)
        residuals = line - gaussian(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((line - np.mean(line))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 < R2_MIN:
            return None
        x_fit = np.linspace(0, n - 1, 500)
        x_fit_mm = (x_fit - (n - 1) / 2) * voxel_size
        y_fit = gaussian(x_fit, *popt)
        fwhm_mm = 2.355 * abs(popt[2]) * voxel_size
        mean_mm = (popt[1] - (n - 1) / 2) * voxel_size
        return {"line": line, "x_mm": x_mm, "fwhm_mm": fwhm_mm, "mean_mm": mean_mm,
                "x_fit_mm": x_fit_mm, "y_fit": y_fit, "r2": r2}
    except RuntimeError:
        return None


def save_profile_plot(pos_name, label, blur_desc, voxel_size,
                      unf_data, filt_data, output_dir):
    """Save a z-axis line profile comparison PDF for one position."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for data, color, name in [
        (unf_data,  "#1f77b4", "Unfiltered"),
        (filt_data, "#d62728", "Filtered"),
    ]:
        if data is None:
            continue
        peak = np.max(data["line"])
        norm = peak if peak > 0 else 1.0
        ax.plot(data["x_mm"], data["line"] / norm, color=color, alpha=0.4, lw=1)
        ax.plot(data["x_fit_mm"], data["y_fit"] / norm, color=color, lw=1.8,
                label=f"{name}  FWHM={data['fwhm_mm']:.2f} mm"
                      f"\n  mean z={data['mean_mm']:.2f} mm  R²={data['r2']:.3f}")
        ax.axvline(data["mean_mm"], color=color, linestyle="--", lw=1, alpha=0.6)

    ax.set_xlabel("Z position (mm)")
    ax.set_ylabel("Normalised intensity")
    ax.set_title(f"{pos_name} — z-axis profile ({blur_desc})\n[GT raw hotspot]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(profile_dir / f"{pos_name}_{label}.pdf", format="pdf", bbox_inches="tight")
    plt.close("all")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find common positions
    unf_dirs = {d.name for d in UNFILTERED_DIR.iterdir()
                if d.is_dir() and d.name not in ("outside_fov", "pdf_outputs")}
    filt_dirs = {d.name for d in FILTERED_DIR.iterdir()
                 if d.is_dir() and d.name not in ("outside_fov", "pdf_outputs")}
    common_positions = sorted(unf_dirs & filt_dirs, key=lambda n: tuple(int(x) for x in n.split("_")))
    print(f"Found {len(common_positions)} common positions")

    # Collect data
    records = []  # list of dicts with position info and FWHMs
    for pos_name in common_positions:
        print(f"Processing {pos_name}...")

        unf_pred_raw, unf_config = load_recon(UNFILTERED_DIR / pos_name / "pred")
        filt_pred_raw, _ = load_recon(FILTERED_DIR / pos_name / "pred")
        gt_raw, _ = load_recon(UNFILTERED_DIR / pos_name / "gt")

        if unf_pred_raw is None or filt_pred_raw is None:
            print(f"  Skipping: missing pred reconstruction")
            continue
        if gt_raw is None:
            print(f"  Skipping: missing gt reconstruction")
            continue

        voxel_size = unf_config["VOXEL_SIZE_IN_MM"]
        unf_pred_blurred = gaussian_filter(unf_pred_raw.copy(), sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
        filt_pred_blurred = gaussian_filter(filt_pred_raw.copy(), sigma=BLUR_SIGMA, radius=BLUR_RADIUS)

        # Use raw GT hotspot as line-profile reference for both pred images
        max_idx = np.unravel_index(np.argmax(gt_raw), gt_raw.shape)
        hotspot = (max_idx[0], max_idx[1], max_idx[2])

        parts = pos_name.split("_")
        x_mm, y_mm, z_mm = int(parts[0]), int(parts[1]), int(parts[2])
        radial_mm = np.sqrt(x_mm**2 + y_mm**2)

        rec = {"name": pos_name, "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "radial_mm": radial_mm}

        for label, unf_pred, filt_pred, blur_desc in [
            ("raw",     unf_pred_raw,     filt_pred_raw,     "Raw"),
            ("blurred", unf_pred_blurred, filt_pred_blurred, f"Blurred ($\\sigma$={BLUR_SIGMA})"),
        ]:
            rec[f"unf_fwhm_{label}"] = get_z_fwhm(unf_pred, hotspot, voxel_size)
            rec[f"filt_fwhm_{label}"] = get_z_fwhm(filt_pred, hotspot, voxel_size)

            unf_profile = get_z_profile_data(unf_pred, hotspot, voxel_size)
            filt_profile = get_z_profile_data(filt_pred, hotspot, voxel_size)
            save_profile_plot(pos_name, label, blur_desc, voxel_size,
                              unf_profile, filt_profile, OUTPUT_DIR)

        records.append(rec)
        up = rec["unf_fwhm_blurred"]
        fp = rec["filt_fwhm_blurred"]
        if up and fp:
            print(f"  Pred z-FWHM (blurred): unf={up:.2f}, filt={fp:.2f}, delta={fp-up:+.2f} mm")

    if not records:
        print("No data to plot.")
        return

    # ===== Plotting =====
    for label in ["raw", "blurred"]:
        blur_desc = "Raw" if label == "raw" else f"Blurred ($\\sigma$={BLUR_SIGMA})"

        valid = [r for r in records if r[f"unf_fwhm_{label}"] is not None and r[f"filt_fwhm_{label}"] is not None]
        if not valid:
            continue

        radials = np.array([r["radial_mm"] for r in valid])
        z_vals = np.array([r["z_mm"] for r in valid])
        unf_fwhm  = np.array([r[f"unf_fwhm_{label}"] for r in valid])
        filt_fwhm = np.array([r[f"filt_fwhm_{label}"] for r in valid])
        delta_fwhm = filt_fwhm - unf_fwhm

        z_unique = sorted(set(r["z_mm"] for r in valid))
        colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(z_unique), 1)))

        radial_groups = {}
        for r in valid:
            key = (r["x_mm"], r["y_mm"])
            radial_groups.setdefault(key, []).append(r)
        radial_keys = sorted(radial_groups.keys(), key=lambda k: np.sqrt(k[0]**2 + k[1]**2))
        r_colors = plt.cm.plasma(np.linspace(0, 0.9, max(len(radial_keys), 1)))

        ylabel = r"$\Delta$ FWHM$_z$ (mm) [filtered $-$ unfiltered]"

        # --- 1) Delta FWHM vs radial distance ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for zi, z_val in enumerate(z_unique):
            mask = z_vals == z_val
            order = np.argsort(radials[mask])
            ax.plot(radials[mask][order], delta_fwhm[mask][order], 'o-', color=colors[zi],
                    label=f"z={z_val} mm", markersize=6)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Radial distance (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pred $\\Delta$ FWHM$_z$ vs Radial Distance ({blur_desc})\n[z-axis profile through GT hotspot, filtered $-$ unfiltered]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"delta_z_fwhm_vs_radial_{label}.pdf", format="pdf", bbox_inches="tight")
        plt.close("all")

        # --- 2) Delta FWHM vs axial position ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for ri, key in enumerate(radial_keys):
            pts = sorted(radial_groups[key], key=lambda p: p["z_mm"])
            r_dist = np.sqrt(key[0]**2 + key[1]**2)
            zz = [p["z_mm"] for p in pts]
            dd = [p[f"filt_fwhm_{label}"] - p[f"unf_fwhm_{label}"] for p in pts]
            ax.plot(zz, dd, 'o-', color=r_colors[ri],
                    label=f"r={r_dist:.0f} mm ({key[0]},{key[1]})", markersize=6)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel("Axial position z (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pred $\\Delta$ FWHM$_z$ vs Axial Position ({blur_desc})\n[z-axis profile through GT hotspot, filtered $-$ unfiltered]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"delta_avg_fwhm_vs_axial_{label}.pdf", format="pdf", bbox_inches="tight")
        plt.close("all")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
