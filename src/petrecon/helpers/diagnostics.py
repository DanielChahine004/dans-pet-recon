"""
Diagnostic visualizations for PET reconstruction pipeline.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import os


# =============================================================================
# SSS Diagnostic Plots
# =============================================================================

def plot_sss_diagnostics(scatter_sino, sinogram_params, scatter_per_lor,
                         measured_lors, object_radius, output_dir='outputs'):
    """
    Generate and save SSS diagnostic plots.

    Shows scatter sinogram, measured sinogram, tail fitting, per-LOR
    scatter distribution, axial scatter, and scatter fraction profile.
    """
    from .scatter import _bin_lors_to_sinogram

    sp = sinogram_params
    s_edges = sp['s_edges']
    phi_edges = sp['phi_edges']
    z_avg_edges = sp['z_avg_edges']
    z_diff_edges = sp['z_diff_edges']

    s_centers = (s_edges[:-1] + s_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    z_avg_centers = (z_avg_edges[:-1] + z_avg_edges[1:]) / 2
    z_diff_centers = (z_diff_edges[:-1] + z_diff_edges[1:]) / 2

    # Bin measured LORs into same sinogram format
    measured_sino = _bin_lors_to_sinogram(
        measured_lors, s_edges, phi_edges, z_avg_edges, z_diff_edges
    )

    scatter_sino_raw = sp.get('scatter_sino_raw', None)
    has_raw = scatter_sino_raw is not None

    n_cols = 5 if has_raw else 4
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 11))
    fig.suptitle('SSS Scatter Correction Diagnostics', fontsize=14, fontweight='bold')

    col = 0

    # --- Panel: Raw scatter sinogram (s vs phi) ---
    if has_raw:
        ax = axes[0, col]
        raw_2d = scatter_sino_raw.sum(axis=(2, 3))
        im = ax.imshow(raw_2d.T, extent=[s_centers[0], s_centers[-1],
                       phi_centers[0], phi_centers[-1]],
                       aspect='auto', origin='lower', cmap='hot')
        ax.set_xlabel('s (mm)')
        ax.set_ylabel('phi (rad)')
        ax.set_title('Raw Scatter Sinogram (pre tail-fit)')
        plt.colorbar(im, ax=ax, label='Counts')
        if object_radius:
            ax.axvline(object_radius, color='cyan', ls='--', alpha=0.7, label=f'r={object_radius:.0f}mm')
            ax.axvline(-object_radius, color='cyan', ls='--', alpha=0.7)
            ax.legend(fontsize=7)
        col += 1

    # --- Panel: Tail-fitted scatter sinogram (s vs phi) ---
    ax = axes[0, col]
    scatter_2d = scatter_sino.sum(axis=(2, 3))
    im = ax.imshow(scatter_2d.T, extent=[s_centers[0], s_centers[-1],
                   phi_centers[0], phi_centers[-1]],
                   aspect='auto', origin='lower', cmap='hot')
    ax.set_xlabel('s (mm)')
    ax.set_ylabel('phi (rad)')
    ax.set_title('Scatter Sinogram (tail-fitted)' if has_raw else 'Scatter Sinogram (s vs phi)')
    plt.colorbar(im, ax=ax, label='Counts')
    if object_radius:
        ax.axvline(object_radius, color='cyan', ls='--', alpha=0.7, label=f'r={object_radius:.0f}mm')
        ax.axvline(-object_radius, color='cyan', ls='--', alpha=0.7)
        ax.legend(fontsize=7)
    col += 1

    # --- Panel: Measured & Corrected sinograms (s vs phi) ---
    measured_2d = measured_sino.sum(axis=(2, 3))
    corrected_2d = np.maximum(measured_2d - scatter_2d, 0)
    vmax_shared = max(measured_2d.max(), corrected_2d.max())

    ax = axes[0, col]
    im = ax.imshow(measured_2d.T, extent=[s_centers[0], s_centers[-1],
                   phi_centers[0], phi_centers[-1]],
                   aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=vmax_shared)
    ax.set_xlabel('s (mm)')
    ax.set_ylabel('phi (rad)')
    ax.set_title('Measured Sinogram (s vs phi)')
    plt.colorbar(im, ax=ax, label='Counts')
    if object_radius:
        ax.axvline(object_radius, color='cyan', ls='--', alpha=0.7)
        ax.axvline(-object_radius, color='cyan', ls='--', alpha=0.7)
    col += 1

    ax = axes[0, col]
    im = ax.imshow(corrected_2d.T, extent=[s_centers[0], s_centers[-1],
                   phi_centers[0], phi_centers[-1]],
                   aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=vmax_shared)
    ax.set_xlabel('s (mm)')
    ax.set_ylabel('phi (rad)')
    ax.set_title('Scatter-Corrected Sinogram (s vs phi)')
    plt.colorbar(im, ax=ax, label='Counts')
    if object_radius:
        ax.axvline(object_radius, color='cyan', ls='--', alpha=0.7)
        ax.axvline(-object_radius, color='cyan', ls='--', alpha=0.7)
    col += 1

    # --- Panel: Radial profiles — raw, tail-fitted, measured ---
    ax = axes[0, col]
    scatter_profile = scatter_sino.sum(axis=(1, 2, 3))
    measured_profile = measured_sino.sum(axis=(1, 2, 3))
    corrected_profile = np.maximum(measured_profile - scatter_profile, 0)
    ax.plot(s_centers, measured_profile, 'b-', label='Measured', lw=2)
    ax.plot(s_centers, scatter_profile, 'r-', label='Scatter (tail-fitted)', lw=2)
    ax.plot(s_centers, corrected_profile, 'g-', label='Corrected (meas - scat)', lw=2)
    if has_raw:
        raw_profile = scatter_sino_raw.sum(axis=(1, 2, 3))
        ax.plot(s_centers, raw_profile, 'r--', label='Scatter (raw)', lw=1.5, alpha=0.6)
    if object_radius:
        ax.axvline(object_radius, color='gray', ls='--', alpha=0.5)
        ax.axvline(-object_radius, color='gray', ls='--', alpha=0.5)
        ax.axvspan(object_radius, s_centers[-1] * 1.1, alpha=0.1, color='green', label='Tail region')
        ax.axvspan(s_centers[0] * 1.1, -object_radius, alpha=0.1, color='green')
    ax.set_xlabel('s (mm)')
    ax.set_ylabel('Counts')
    ax.set_title('Radial Profile: Measured vs Scatter')
    ax.legend(fontsize=8)
    ax.set_xlim(s_centers[0] * 1.1, s_centers[-1] * 1.1)

    # --- Panel: Per-LOR scatter histogram ---
    col = 0
    ax = axes[1, col]
    nonzero = scatter_per_lor[scatter_per_lor > 0]
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=100, color='red', alpha=0.7, edgecolor='darkred')
        ax.axvline(scatter_per_lor.mean(), color='black', ls='--', lw=1.5,
                   label=f'Mean={scatter_per_lor.mean():.2f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Scatter estimate per LOR')
    ax.set_ylabel('Count')
    n_total = len(scatter_per_lor)
    n_nonzero = len(nonzero)
    ax.set_title(f'Per-LOR Scatter Distribution\n({n_nonzero}/{n_total} non-zero, '
                 f'max={scatter_per_lor.max():.1f})')

    # --- Panel: Scatter sinogram (z_avg vs z_diff) ---
    col += 1
    ax = axes[1, col]
    scatter_z = scatter_sino.sum(axis=(0, 1))
    im = ax.imshow(scatter_z.T,
                   extent=[z_avg_centers[0], z_avg_centers[-1],
                           z_diff_centers[0], z_diff_centers[-1]],
                   aspect='auto', origin='lower', cmap='hot')
    ax.set_xlabel('z_avg (mm)')
    ax.set_ylabel('z_diff (mm)')
    ax.set_title('Scatter Sinogram (z_avg vs z_diff)')
    plt.colorbar(im, ax=ax, label='Counts')

    # --- Panel: Scatter fraction by s-bin ---
    col += 1
    ax = axes[1, col]
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(measured_profile > 0, scatter_profile / measured_profile, 0)
    ax.plot(s_centers, frac * 100, 'g-', lw=2)
    ax.set_xlabel('s (mm)')
    ax.set_ylabel('Scatter Fraction (%)')
    ax.set_title('Scatter Fraction by Radial Bin')
    if object_radius:
        ax.axvline(object_radius, color='gray', ls='--', alpha=0.5)
        ax.axvline(-object_radius, color='gray', ls='--', alpha=0.5)
    overall_frac = scatter_sino.sum() / max(measured_sino.sum(), 1) * 100
    ax.axhline(overall_frac, color='red', ls=':', alpha=0.7,
               label=f'Overall: {overall_frac:.1f}%')
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(150, frac.max() * 100 * 1.3 + 5))
    ax.set_xlim(s_centers[0] * 1.1, s_centers[-1] * 1.1)

    # --- Panel: Tail-fit scaling comparison (only when raw available) ---
    if has_raw:
        col += 1
        ax = axes[1, col]
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_by_bin = np.where(raw_profile > 0, scatter_profile / raw_profile, 0)
        ax.plot(s_centers, scale_by_bin, 'm-', lw=2)
        ax.set_xlabel('s (mm)')
        ax.set_ylabel('Scale Factor')
        ax.set_title('Tail-Fit Scale Factor by Radial Bin')
        if object_radius:
            ax.axvline(object_radius, color='gray', ls='--', alpha=0.5)
            ax.axvline(-object_radius, color='gray', ls='--', alpha=0.5)
        overall_scale = scatter_sino.sum() / max(scatter_sino_raw.sum(), 1e-30)
        ax.axhline(overall_scale, color='red', ls=':', alpha=0.7,
                   label=f'Overall: {overall_scale:.2e}')
        ax.legend(fontsize=8)
        ax.set_xlim(s_centers[0] * 1.1, s_centers[-1] * 1.1)

    # Hide any unused bottom-row axes
    for c in range(col + 1, n_cols):
        axes[1, c].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sss_diagnostics.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"SSS diagnostics saved to {save_path}")


# =============================================================================
# Pipeline State Machine Diagram
# =============================================================================

def plot_pipeline_diagram(output_dir='outputs'):
    """Generate reconstruction pipeline state machine diagram."""

    fig, ax = plt.subplots(figsize=(13, 18))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-0.5, 22)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ---- helpers ----
    def _box(cx, cy, text, color='#E3F2FD', w=3.8, h=0.85, fs=8.5, bold=True):
        r = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor='#333', lw=1.2)
        ax.add_patch(r)
        weight = 'bold' if bold else 'normal'
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
                fontweight=weight, linespacing=1.3)

    def _diamond(cx, cy, text, color='#FFF9C4', size=1.0, fs=8):
        d = size / 2
        pts = [[cx, cy + d], [cx + d * 1.8, cy], [cx, cy - d], [cx - d * 1.8, cy]]
        p = Polygon(pts, closed=True, facecolor=color, edgecolor='#333', lw=1.2)
        ax.add_patch(p)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs)

    def _arrow(x1, y1, x2, y2, label='', lo=(0, 0)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.4))
        if label:
            ax.text((x1 + x2) / 2 + lo[0], (y1 + y2) / 2 + lo[1],
                    label, fontsize=7, color='#555', ha='center', va='center')

    # colors
    C_DATA = '#BBDEFB'
    C_AC   = '#FFE0B2'
    C_SENS = '#C8E6C9'
    C_SSS  = '#FFCDD2'
    C_OSEM = '#E1BEE7'
    C_DEC  = '#FFF9C4'
    C_SKIP = '#EEEEEE'

    # ===================== MAIN FLOW =====================
    X = 5  # main column center

    # Row 1: Load data
    y = 21
    _box(X, y, 'Load LORs\nLoad Ground Truth', C_DATA)
    _arrow(X, y - 0.42, X, y - 1.1)

    # Row 2: Load mu-map
    y = 19.5
    _box(X, y, 'Load / Generate mu-map', C_DATA)
    _arrow(X, y - 0.42, X, y - 1.1)

    # Row 3: AC decision
    y = 18
    _diamond(X, y, 'AC\nenabled?', C_DEC)
    _arrow(X, y - 0.5, X, y - 1.3, 'Yes', lo=(-0.4, 0))
    _arrow(X + 0.9, y, 9.5, y, 'No', lo=(0, 0.2))
    _box(9.5, y, 'ACF = 1.0', C_SKIP, w=2.2, h=0.6, fs=7.5)

    # Row 4: Compute ACFs
    y = 16.3
    _box(X, y, 'Compute ACFs\nacf_i = exp(-integral mu along LOR_i)', C_AC, h=0.9)
    _arrow(X, y - 0.45, X, y - 1.2)

    # Row 5: Sensitivity
    y = 14.7
    _box(X, y, 'Compute Sensitivity Image\nS_j = sum_i  a_i * H_ij   (AC-weighted)\nor   S_j = sum_i  H_ij   (geometric)', C_SENS, h=1.0, w=4.5)
    _arrow(X, y - 0.5, X, y - 1.2)

    # Row 6: Scatter decision
    y = 13.1
    _diamond(X, y, 'Scatter\nenabled?', C_DEC)
    _arrow(X, y - 0.5, X, y - 1.3, 'Yes', lo=(-0.4, 0))
    _arrow(X + 0.9, y, 9.5, y, 'No', lo=(0, 0.2))
    _box(9.5, y, 's_i = 0', C_SKIP, w=2.2, h=0.6, fs=7.5)

    # Row 7: SSS setup
    y = 11.4
    _box(X, y, 'SSS Setup\nIdentify scatter points  |  Define coarse detectors', C_SSS, w=5.0)
    _arrow(X, y - 0.42, X, y - 1.0)

    # ---- OSEM LOOP ----
    loop_y_top = 10.1
    loop_y_bot = 1.6
    loop_box = FancyBboxPatch((0.6, loop_y_bot), 10.8, loop_y_top - loop_y_bot,
                               boxstyle="round,pad=0.25",
                               facecolor='#F3E5F5', edgecolor='#7B1FA2',
                               lw=2, ls='--', alpha=0.35)
    ax.add_patch(loop_box)
    ax.text(1.2, loop_y_top - 0.3, 'OSEM Iteration Loop  (iter = 1 ... N)',
            fontsize=10, fontweight='bold', color='#7B1FA2')

    # OSEM step 1: Init / Forward project
    y = 9.0
    _box(X, y, 'Initial Backprojection\n(normalized by sensitivity)', C_OSEM, w=4.2)
    _arrow(X, y - 0.42, X, y - 1.0)

    # OSEM step 2: SSS update decision
    y = 7.6
    _diamond(X, y, 'iter >= start\n& interval?', C_DEC, size=0.85)
    _arrow(X + 0.9, y, 9, y, 'Yes', lo=(0, 0.2))
    _arrow(X, y - 0.42, X, y - 1.2, 'No', lo=(-0.4, 0))

    # SSS update block
    _box(9, y, 'SSS Scatter\nEstimation', C_SSS, w=2.5, h=0.85)

    # SSS sub-steps (small annotation)
    sss_x = 9
    ax.text(sss_x, y - 0.7,
            '1. GPU ray-trace to detectors\n'
            '2. Pair-wise KN accumulation\n'
            '3. Tail-fitting normalization\n'
            '4. Interpolate to per-LOR',
            fontsize=6.5, ha='center', va='top', color='#666',
            style='italic', linespacing=1.4)

    # Arrow from SSS back to main flow
    _arrow(9, y - 0.42, 9, y - 1.8)
    _arrow(9, y - 1.8, X + 1.9, y - 1.8)

    # OSEM step 3: Forward project + ratio + backproject
    y = 6.0
    _box(X, y, 'Forward Project\npred_i = a_i * (H*x)_i + s_i', C_OSEM, w=4.2, h=0.9)
    _arrow(X, y - 0.45, X, y - 1.2)

    y = 4.5
    _box(X, y, 'Compute Ratios\nratio_i = a_i / pred_i', C_OSEM, w=4.2, h=0.85)
    _arrow(X, y - 0.42, X, y - 1.2)

    y = 3.0
    _box(X, y, 'Backproject & Update\nx_j *= [ sum_i H_ij * ratio_i ] / S_j', C_OSEM, w=4.8, h=0.85)

    # Loop-back arrow
    ax.annotate('', xy=(1.2, 7.6), xytext=(1.2, 3.0),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5,
                                connectionstyle='arc3,rad=0.3'))
    ax.text(0.4, 5.3, 'next\niter', fontsize=7, color='#7B1FA2',
            ha='center', va='center', fontweight='bold')

    # Arrow out of loop
    _arrow(X, 3.0 - 0.42, X, 3.0 - 1.1)

    # Save outputs
    y = 1.3
    _box(X, y, 'Save Outputs\nreconstruction_image.npy  |  sensitivity_image.npy', C_DATA, w=5.0)

    # ---- Legend ----
    legend_items = [
        (C_DATA, 'Data I/O'),
        (C_AC,   'Attenuation Correction'),
        (C_SENS, 'Sensitivity'),
        (C_SSS,  'Scatter Correction (SSS)'),
        (C_OSEM, 'OSEM Reconstruction'),
        (C_DEC,  'Decision'),
    ]
    for i, (c, label) in enumerate(legend_items):
        bx = 10.5
        by = 21 - i * 0.55
        r = FancyBboxPatch((bx - 0.25, by - 0.15), 0.5, 0.3,
                           boxstyle="round,pad=0.05", facecolor=c,
                           edgecolor='#333', lw=0.8)
        ax.add_patch(r)
        ax.text(bx + 0.5, by, label, fontsize=7, va='center')

    ax.set_title('PET Reconstruction Pipeline', fontsize=14,
                 fontweight='bold', pad=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pipeline_diagram.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Pipeline diagram saved to {save_path}")
