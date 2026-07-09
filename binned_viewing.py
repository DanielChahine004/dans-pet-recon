#%%
"""Interactive side-by-side comparison of continuous vs binned reconstructions.

Usage (as a script or in a notebook cell):
    Change POSITION and KIND below, then run.
    A Plotly figure with 4 panels opens in browser, with a shared z-slice slider.
"""
import numpy as np
import pickle
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helpers.plotting import plot_line_profile_with_gaussian_fit

# ==============================================================================
# CONFIGURATION — change these to view different point sources
# ==============================================================================
BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Processed"))
POSITION = "0_0_32"      # point source folder name
BLUR_SIGMA = 1              # set to 0 to view raw (no blur)
BLUR_RADIUS = 1

# BINNING_LABELS = ["very_fine_500x50", "fine_400x40", "moderate_300x30", "standard_200x20", "coarse_100x10"]
BINNING_LABELS = ["very_fine_500x50", "fine_400x40"]
DISPLAY_NAMES = {
    "gt": "GT",
    "gt_binned_very_fine_500x50": "GT Very Fine (500x50)",
    "gt_binned_fine_400x40": "GT Fine (400x40)",
    # "gt_binned_moderate_300x30": "GT Moderate (300x30)",
    # "gt_binned_standard_200x20": "GT Standard (200x20)",
    # "gt_binned_coarse_100x10": "GT Coarse (100x10)",
    "pred": "Continuous Prediction",
}

# ==============================================================================
# LOAD RECONSTRUCTIONS
# ==============================================================================
pos_dir = BASE_DIR / POSITION

def load_recon(recon_dir):
    config_path = recon_dir / "recon_config.pkl"
    recon_path = recon_dir / "reconstruction_image.npy"
    if not recon_path.exists():
        return None, None
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return np.load(recon_path), config

images = {}

# GT (continuous)
img, cfg = load_recon(pos_dir / "gt")
if img is None:
    raise FileNotFoundError(f"No GT reconstruction at {pos_dir / 'gt'}")
voxel_size = cfg["VOXEL_SIZE_IN_MM"]
if BLUR_SIGMA > 0:
    img = gaussian_filter(img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
images["gt"] = img

# Binned GTs
for label in BINNING_LABELS:
    key = f"gt_binned_{label}"
    binned_img, _ = load_recon(pos_dir / key)
    if binned_img is not None:
        if BLUR_SIGMA > 0:
            binned_img = gaussian_filter(binned_img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
        images[key] = binned_img
    else:
        print(f"  Missing: {key} — skipping")

# Continuous prediction
pred_img, _ = load_recon(pos_dir / "pred")
if pred_img is not None:
    if BLUR_SIGMA > 0:
        pred_img = gaussian_filter(pred_img, sigma=BLUR_SIGMA, radius=BLUR_RADIUS)
    images["pred"] = pred_img
else:
    print(f"  Missing: pred — skipping")

panel_order = ["gt"] + [f"gt_binned_{l}" for l in BINNING_LABELS] + ["pred"]
available = [k for k in panel_order if k in images]
n_panels = len(available)
print(f"Loaded {n_panels} reconstructions for {POSITION}")

# ==============================================================================
# FIND HOTSPOT SLICE (from continuous image)
# ==============================================================================
hotspot = np.unravel_index(np.argmax(images["gt"]), images["gt"].shape)
nx, ny, nz = images["gt"].shape

# Per-panel colour scale
vmax_per = {k: img.max() for k, img in images.items()}

# mm coordinates
half_x = nx * voxel_size / 2
half_y = ny * voxel_size / 2

##%%
# ==============================================================================
# BUILD INTERACTIVE FIGURE — axial slices (z-axis), shared slider
# ==============================================================================
fig = make_subplots(
    rows=1, cols=n_panels,
    subplot_titles=[DISPLAY_NAMES.get(k, k) for k in available],
    horizontal_spacing=0.03,
)

# Add initial heatmaps (hotspot slice)
for col, key in enumerate(available, 1):
    slc = images[key][:, :, hotspot[2]].T
    fig.add_trace(
        go.Heatmap(
            z=slc,
            x0=-half_x + voxel_size / 2, dx=voxel_size,
            y0=-half_y + voxel_size / 2, dy=voxel_size,
            zmin=0, zmax=vmax_per[key],
            colorscale="Hot",
            showscale=True,
        ),
        row=1, col=col,
    )

# Build slider steps for each z-slice
z_mm = np.linspace(-nz * voxel_size / 2 + voxel_size / 2,
                     nz * voxel_size / 2 - voxel_size / 2, nz)

steps = []
for zi in range(nz):
    step_data = []
    for key in available:
        slc = images[key][:, :, zi].T
        step_data.append(slc)

    step = dict(
        method="update",
        args=[{"z": step_data}],
        label=f"{z_mm[zi]:.0f}",
    )
    steps.append(step)

fig.update_layout(
    sliders=[dict(
        active=hotspot[2],
        currentvalue={"prefix": "Z = ", "suffix": " mm"},
        steps=steps,
        pad={"t": 30},
    )],
    title=f"{POSITION} — GT, Binned GTs, Prediction — Axial Slices (blur σ={BLUR_SIGMA})",
    height=500,
    width=350 * n_panels,
)

# Set equal aspect ratio on all subplots
for i in range(1, n_panels + 1):
    fig.update_xaxes(title_text="X (mm)", scaleanchor=f"y{i}", row=1, col=i)
    fig.update_yaxes(title_text="Y (mm)" if i == 1 else "", row=1, col=i)

fig.show()

# ==============================================================================
# LINE PROFILES — through the hotspot, one per reconstruction
# ==============================================================================
print("Analyzing line profiles through hotspot...")
for key in available:
    print(f"\n  {DISPLAY_NAMES.get(key, key)}:")
    result = plot_line_profile_with_gaussian_fit(
        images[key],
        axis='z',
        coord1=hotspot[0],
        coord2=hotspot[1],
        voxel_size=voxel_size,
    )
    if result.get('fwhm_mm'):
        print(f"    FWHM: {result['fwhm_mm']:.2f} mm")
    if result.get('fwtm_mm'):
        print(f"    FWTM: {result['fwtm_mm']:.2f} mm")

# %%
