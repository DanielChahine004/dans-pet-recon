import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def plot_interactive(image, title, axis, vmin=None, vmax=None, voxel_size=None):
    if np.any(image < 0):
        print("Warning: Image contains negative values")

    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()

    nx, ny, nz = image.shape

    # Swap the two non-animation axes so x/y display correctly
    axes_to_swap = [i for i in range(3) if i != axis]
    image = np.swapaxes(image, axes_to_swap[0], axes_to_swap[1])

    plot_kwargs = dict(
        title=title,
        animation_frame=axis,
        zmin=vmin,
        zmax=image.max(),
        color_continuous_scale='Hot',
    )

    if voxel_size is not None:
        # Voxel centre coordinates, centred at 0
        x_mm = np.linspace(-(nx * voxel_size) / 2 + voxel_size / 2,
                            (nx * voxel_size) / 2 - voxel_size / 2, nx)
        y_mm = np.linspace(-(ny * voxel_size) / 2 + voxel_size / 2,
                            (ny * voxel_size) / 2 - voxel_size / 2, ny)
        z_mm = np.linspace(-(nz * voxel_size) / 2 + voxel_size / 2,
                            (nz * voxel_size) / 2 - voxel_size / 2, nz)

        # px.imshow always places row 0 at the visual top. To make the vertical
        # axis increase upward (matching plot_static with origin='lower'):
        #   1. Flip the image along the row dimension so row 0 = physical maximum.
        #   2. Pass reversed y-labels (largest first) so y[0]=+max sits at the top.
        #
        # After the axis swap the per-frame slice shapes are:
        #   axis=0: shape (nx,nz,ny)  rows=Z (dim 1)  cols=Y  frames=X
        #   axis=1: shape (nz,ny,nx)  rows=Z (dim 0)  cols=X  frames=Y
        #   axis=2: shape (ny,nx,nz)  rows=Y (dim 0)  cols=X  frames=Z
        if axis == 0:
            image = np.flip(image, axis=1)          # flip Z rows
            plot_kwargs['x'] = y_mm
            plot_kwargs['y'] = z_mm[::-1]           # +half_z at row 0 (top)
            plot_kwargs['labels'] = {'x': 'Y (mm)', 'y': 'Z (mm)', 'animation_frame': 'X (mm)'}
            frame_coords = x_mm
        elif axis == 1:
            image = np.flip(image, axis=0)          # flip Z rows
            plot_kwargs['x'] = x_mm
            plot_kwargs['y'] = z_mm[::-1]
            plot_kwargs['labels'] = {'x': 'X (mm)', 'y': 'Z (mm)', 'animation_frame': 'Y (mm)'}
            frame_coords = y_mm
        else:  # axis == 2
            image = np.flip(image, axis=0)          # flip Y rows
            plot_kwargs['x'] = x_mm
            plot_kwargs['y'] = y_mm[::-1]           # +half_y at row 0 (top)
            plot_kwargs['labels'] = {'x': 'X (mm)', 'y': 'Y (mm)', 'animation_frame': 'Z (mm)'}
            frame_coords = z_mm

    fig = px.imshow(image, **plot_kwargs)

    if voxel_size is not None:
        for i, step in enumerate(fig.layout.sliders[0].steps):
            step.label = f'{frame_coords[i]:.1f}'

    return fig


def plot_static(image, title, axis, depth, vmin=None, vmax=None, voxel_size=None):
    if np.any(image < 0):
        print("Warning: Image contains negative values")

    if vmin is None:
        vmin = image.min()

    plt.figure(figsize=(10, 8))
    vmax = image.max() if vmax is None else vmax

    nx, ny, nz = image.shape
    if voxel_size is not None:
        half_x = nx * voxel_size / 2
        half_y = ny * voxel_size / 2
        half_z = nz * voxel_size / 2

    # After .T the layout is:
    #   axis=0: rows=Z, cols=Y  extent=[left=-half_y, right=+half_y, bottom=-half_z, top=+half_z]
    #   axis=1: rows=Z, cols=X  extent=[-half_x, +half_x, -half_z, +half_z]
    #   axis=2: rows=Y, cols=X  extent=[-half_x, +half_x, -half_y, +half_y]
    if axis == 0:
        extent = [-half_y, half_y, -half_z, half_z] if voxel_size is not None else None
        plt.imshow(image[depth, :, :].T, cmap='hot', vmin=vmin, vmax=vmax,
                   origin='lower' if voxel_size is not None else 'upper', extent=extent)
        plt.xlabel('Y (mm)' if voxel_size is not None else 'Y')
        plt.ylabel('Z (mm)' if voxel_size is not None else 'Z')
    elif axis == 1:
        extent = [-half_x, half_x, -half_z, half_z] if voxel_size is not None else None
        plt.imshow(image[:, depth, :].T, cmap='hot', vmin=vmin, vmax=vmax,
                   origin='lower' if voxel_size is not None else 'upper', extent=extent)
        plt.xlabel('X (mm)' if voxel_size is not None else 'X')
        plt.ylabel('Z (mm)' if voxel_size is not None else 'Z')
    elif axis == 2:
        extent = [-half_x, half_x, -half_y, half_y] if voxel_size is not None else None
        plt.imshow(image[:, :, depth].T, cmap='hot', vmin=vmin, vmax=vmax,
                   origin='lower' if voxel_size is not None else 'upper', extent=extent)
        plt.xlabel('X (mm)' if voxel_size is not None else 'X')
        plt.ylabel('Y (mm)' if voxel_size is not None else 'Y')
    else:
        print('Axis must be 0, 1, or 2 - X, Y, or Z.')
        return None

    plt.colorbar(label='Value')
    plt.title(title)
    plt.tight_layout(pad=2.0)
    return plt


def plot_scanner_geometry(n_rings=10, n_det_per_ring=10, radius=200, axial_length=100, grid_size=(128, 128, 128), voxel_size=2.0,
                          fov_radius=None, fov_axial_half_length=None, lors=None, max_lors_to_plot=10):
    """
    Plot an interactive 3D visualization of the PET scanner geometry.

    Args:
        n_rings: Number of axial rings
        n_det_per_ring: Number of detectors per ring
        radius: Inner radius of scanner in mm
        axial_length: Axial length of scanner in mm
        grid_size: tuple (nx, ny, nz) for the image grid dimensions
        voxel_size: size of each voxel in mm
        fov_radius: Transaxial FOV radius to visualize (mm)
        fov_axial_half_length: Axial FOV half-length to visualize (mm). Defaults to image grid half-height.
        lors: Optional (N, 6) array of LORs [x1, y1, z1, x2, y2, z2] to plot
        max_lors_to_plot: Maximum number of LORs to plot (randomly sampled if exceeded)

    Returns:
        Plotly figure object
    """

    # Generate detector positions
    angles = np.linspace(0, 2 * np.pi, n_det_per_ring, endpoint=False)
    z_positions = np.linspace(-axial_length / 2, axial_length / 2, n_rings)

    n_total = n_rings * n_det_per_ring
    detector_x = np.zeros(n_total)
    detector_y = np.zeros(n_total)
    detector_z = np.zeros(n_total)

    for ring_idx in range(n_rings):
        for det_idx in range(n_det_per_ring):
            idx = ring_idx * n_det_per_ring + det_idx
            detector_x[idx] = radius * np.cos(angles[det_idx])
            detector_y[idx] = radius * np.sin(angles[det_idx])
            detector_z[idx] = z_positions[ring_idx]

    # Calculate image volume bounds (centered at origin)
    half_x = (grid_size[0] * voxel_size) / 2
    half_y = (grid_size[1] * voxel_size) / 2
    half_z = (grid_size[2] * voxel_size) / 2

    # Create figure
    fig = go.Figure()

    # Add detector point cloud
    fig.add_trace(go.Scatter3d(
        x=detector_x,
        y=detector_y,
        z=detector_z,
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.6),
        name='Detectors'
    ))

    # Plot LORs if provided
    if lors is not None and len(lors) > 0:
        # Sample LORs if there are too many
        if len(lors) > max_lors_to_plot:
            indices = np.random.choice(len(lors), max_lors_to_plot, replace=False)
            lors_to_plot = lors[indices]
        else:
            lors_to_plot = lors

        # Build line segments with None separators for efficient plotting
        x_lines = []
        y_lines = []
        z_lines = []
        for lor in lors_to_plot:
            x_lines.extend([lor[0], lor[3], None])
            y_lines.extend([lor[1], lor[4], None])
            z_lines.extend([lor[2], lor[5], None])

        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='green', width=2),
            opacity=0.5,
            name=f'LORs ({len(lors_to_plot):,} shown)'
        ))

        # Add LOR endpoints as markers
        endpoint1_x = lors_to_plot[:, 0]
        endpoint1_y = lors_to_plot[:, 1]
        endpoint1_z = lors_to_plot[:, 2]
        endpoint2_x = lors_to_plot[:, 3]
        endpoint2_y = lors_to_plot[:, 4]
        endpoint2_z = lors_to_plot[:, 5]

        fig.add_trace(go.Scatter3d(
            x=np.concatenate([endpoint1_x, endpoint2_x]),
            y=np.concatenate([endpoint1_y, endpoint2_y]),
            z=np.concatenate([endpoint1_z, endpoint2_z]),
            mode='markers',
            marker=dict(size=4, color='lime', symbol='circle'),
            name='LOR Endpoints'
        ))

    # Define the 8 corners of the image volume box
    corners = np.array([
        [-half_x, -half_y, -half_z],
        [half_x, -half_y, -half_z],
        [half_x, half_y, -half_z],
        [-half_x, half_y, -half_z],
        [-half_x, -half_y, half_z],
        [half_x, -half_y, half_z],
        [half_x, half_y, half_z],
        [-half_x, half_y, half_z],
    ])

    # Define edges of the box (pairs of corner indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    # Add box edges
    for i, (start, end) in enumerate(edges):
        fig.add_trace(go.Scatter3d(
            x=[corners[start, 0], corners[end, 0]],
            y=[corners[start, 1], corners[end, 1]],
            z=[corners[start, 2], corners[end, 2]],
            mode='lines',
            line=dict(color='red', width=4),
            name='Image Volume' if i == 0 else None,
            showlegend=(i == 0),
        ))

    # Visualize FOV cylinder
    if fov_radius is not None:
        # Use provided axial half-length or default to image grid half-height
        fov_z = fov_axial_half_length if fov_axial_half_length is not None else half_z

        # Draw circles at top and bottom of the FOV to show acceptance region
        n_circle_pts = 50
        theta = np.linspace(0, 2 * np.pi, n_circle_pts)
        circle_x = fov_radius * np.cos(theta)
        circle_y = fov_radius * np.sin(theta)

        # Bottom circle
        fig.add_trace(go.Scatter3d(
            x=circle_x, y=circle_y, z=np.full_like(circle_x, -fov_z),
            mode='lines',
            line=dict(color='orange', width=3),
            name=f'FOV (r={fov_radius}mm, z=±{fov_z}mm)'
        ))

        # Top circle
        fig.add_trace(go.Scatter3d(
            x=circle_x, y=circle_y, z=np.full_like(circle_x, fov_z),
            mode='lines',
            line=dict(color='orange', width=3),
            showlegend=False
        ))

        # Vertical lines connecting top and bottom circles
        for i in range(0, n_circle_pts, n_circle_pts // 8):
            fig.add_trace(go.Scatter3d(
                x=[circle_x[i], circle_x[i]],
                y=[circle_y[i], circle_y[i]],
                z=[-fov_z, fov_z],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False
            ))

    # Update layout with orthographic projection
    fig.update_layout(
        title='PET Scanner Geometry',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                projection=dict(type='orthographic')
            ),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def snap_lors_to_detectors(lor_path=None, n_rings=10, n_det_per_ring=10, radius=200, axial_length=100,
                           filter_distant=True, max_dist_mm=None, lor_data=None):
    """
    Snap LOR endpoints to the nearest virtual detector positions.

    Args:
        lor_path: Path to LOR file (CSV or NPZ), loaded via load_lors.
                  Not required if lor_data is provided directly.
        n_rings: Number of axial rings.
        n_det_per_ring: Number of detectors per ring.
        radius: Inner radius of scanner in mm.
        axial_length: Axial length of scanner in mm.
        filter_distant: If True and max_dist_mm is None, discard LORs where
            either endpoint is farther than the chord distance between adjacent
            detectors in a ring.
        max_dist_mm: Fixed maximum snap distance in mm. If provided, overrides
            the adaptive filter_distant threshold.
        lor_data: (N, 6) numpy array of LOR endpoints. If provided, lor_path
            is ignored.

    Returns:
        snapped_lors: (N, 6) array with endpoints moved to nearest detectors.
    """
    if lor_data is None:
        from helpers.data_io import load_lors
        lor_data = load_lors(lor_path)

    # Generate detector positions
    angles = np.linspace(0, 2 * np.pi, n_det_per_ring, endpoint=False)
    z_positions = np.linspace(-axial_length / 2, axial_length / 2, n_rings)

    n_total = n_rings * n_det_per_ring
    detectors = np.zeros((n_total, 3))
    for ring_idx in range(n_rings):
        for det_idx in range(n_det_per_ring):
            idx = ring_idx * n_det_per_ring + det_idx
            detectors[idx, 0] = radius * np.cos(angles[det_idx])
            detectors[idx, 1] = radius * np.sin(angles[det_idx])
            detectors[idx, 2] = z_positions[ring_idx]

    # Project endpoints onto the detector cylinder so only azimuthal and
    # axial distance matter (removes radial mismatch from crystal depth).
    from scipy.spatial import cKDTree

    def _project_to_cylinder(xyz, r):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rr = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
        scale = r / rr
        return np.column_stack([x * scale, y * scale, z])

    tree = cKDTree(detectors)

    proj1 = _project_to_cylinder(lor_data[:, :3], radius)
    proj2 = _project_to_cylinder(lor_data[:, 3:], radius)

    dist1, idx1 = tree.query(proj1)
    dist2, idx2 = tree.query(proj2)

    if max_dist_mm is not None:
        max_dist = max_dist_mm
    elif filter_distant:
        max_dist = radius * np.sin(np.pi / n_det_per_ring)
    else:
        max_dist = None

    if max_dist is not None:
        mask = (dist1 <= max_dist) & (dist2 <= max_dist)
        n_total = lor_data.shape[0]
        n_kept = int(mask.sum())
        n_dropped = n_total - n_kept
        print(f"LOR filtering: {n_dropped} dropped, {n_kept} remain "
              f"(of {n_total} total, max dist={max_dist:.1f} mm)")
        idx1 = idx1[mask]
        idx2 = idx2[mask]

    snapped_lors = np.hstack([detectors[idx1], detectors[idx2]])

    return snapped_lors


def create_sinogram(lors, n_radial_bins, n_angle_bins, n_axial_bins, max_radius=None,
                    weights=None, title='Sinogram (s, φ) per Axial Slice'):
    """
    Create a 3D sinogram from LOR data using standard parameterization.

    Sinogram coordinates:
    - s: Signed perpendicular distance from the LOR to the z-axis (transverse plane)
    - φ: Azimuthal angle of the LOR in the transverse plane, normalized to [0, π)
    - z: Axial position (average z-coordinate of LOR endpoints)

    Args:
        lors: (N, 6) array of LOR endpoints [x1, y1, z1, x2, y2, z2]
        n_radial_bins: Number of radial (s) bins
        n_angle_bins: Number of angle (φ) bins
        n_axial_bins: Number of axial (z) bins
        max_radius: Maximum |s| value to consider (auto-detected if None)
        weights: Optional (N,) array of weights for each LOR. If None, each LOR counts as 1.
        title: Title for the plot

    Returns:
        fig: Plotly figure of the sinogram (summed over z for 2D visualization)
        sinogram: numpy array of shape (n_radial_bins, n_angle_bins, n_axial_bins)
    """
    x1, y1, z1 = lors[:, 0], lors[:, 1], lors[:, 2]
    x2, y2, z2 = lors[:, 3], lors[:, 4], lors[:, 5]

    # LOR direction vector in XY plane
    dx = x2 - x1
    dy = y2 - y1

    # Length of LOR projection in XY plane
    lor_length_xy = np.sqrt(dx**2 + dy**2)

    # Signed perpendicular distance from origin to LOR
    # Using cross product: s = (P1 × P2) / |P2 - P1|
    s = (x1 * y2 - x2 * y1) / np.maximum(lor_length_xy, 1e-10)

    # Angle of LOR direction in XY plane
    phi = np.arctan2(dy, dx)

    # Normalize φ to [0, π) since LORs are symmetric (A→B same as B→A)
    # When flipping angle by π, also flip sign of s
    flip_mask = phi < 0
    phi[flip_mask] += np.pi
    s[flip_mask] = -s[flip_mask]

    wrap_mask = phi >= np.pi
    phi[wrap_mask] -= np.pi
    s[wrap_mask] = -s[wrap_mask]

    # Axial position: average z of endpoints
    z = (z1 + z2) / 2

    # Determine bin edges
    if max_radius is None:
        max_radius = np.max(np.abs(s)) * 1.01

    min_z, max_z = np.min(z), np.max(z)
    z_padding = (max_z - min_z) * 0.01 if max_z > min_z else 1.0
    min_z -= z_padding
    max_z += z_padding

    radial_edges = np.linspace(-max_radius, max_radius, n_radial_bins + 1)
    angle_edges = np.linspace(0, np.pi, n_angle_bins + 1)
    axial_edges = np.linspace(min_z, max_z, n_axial_bins + 1)

    # Bin the LORs
    sinogram, _ = np.histogramdd(
        np.column_stack([s, phi, z]),
        bins=[radial_edges, angle_edges, axial_edges],
        weights=weights
    )

    # Bin centers for plotting
    radial_centers = (radial_edges[:-1] + radial_edges[1:]) / 2
    axial_centers = (axial_edges[:-1] + axial_edges[1:]) / 2

    # Transpose to (n_axial, n_angle, n_radial) for animation over axial slices
    # px.imshow expects (frames, y, x)
    sinogram_for_plot = np.transpose(sinogram, (2, 1, 0))

    fig = px.imshow(
        sinogram_for_plot,
        animation_frame=0,
        x=radial_centers,
        y=np.degrees((angle_edges[:-1] + angle_edges[1:]) / 2),
        color_continuous_scale='Hot',
        zmin=0,
        zmax=sinogram.max(),
        labels={'x': 'Radial Distance s (mm)', 'y': 'Angle φ (degrees)', 'animation_frame': 'Axial Slice'},
    )

    # Update slider labels to show z positions
    for i, step in enumerate(fig.layout.sliders[0].steps):
        step.label = f'{axial_centers[i]:.1f} mm'

    fig.update_layout(
        title=title,
    )

    return fig, sinogram


def plot_intensity_histogram(image, nbins=100, title='Intensity Histogram'):

    values = image.flatten()
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=nbins)])
    fig.update_layout(
        title=title,
        xaxis_title='Intensity',
        yaxis_title='Count',
        bargap=0.05
    )

    return fig


def plot_line_profile_with_gaussian_fit(image, axis, coord1, coord2, voxel_size=1.0, show=True):
    """
    Extract a line profile along one axis, fit a Gaussian, and plot both.

    Args:
        image: 3D numpy array (x, y, z)
        axis: The axis along which to extract the line ('x', 'y', or 'z', or 0, 1, 2)
        coord1: First fixed coordinate (for the two axes not being profiled)
        coord2: Second fixed coordinate
        voxel_size: Size of each voxel in mm (for physical units)
        show: Whether to display the plot immediately

    Returns:
        dict with 'fwhm', 'fwhm_mm', 'fwtm', 'fwtm_mm', 'amplitude', 'mean', 'sigma', 'fig'

    Example:
        # Line along z-axis at x=100, y=128
        result = plot_line_profile_with_gaussian_fit(image, 'z', x=100, y=128)

        # Line along x-axis at y=128, z=64
        result = plot_line_profile_with_gaussian_fit(image, 'x', y=128, z=64)
    """
    from scipy.optimize import curve_fit

    # Convert axis to integer
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_map[axis.lower()]

    # Extract line based on axis, and compute fixed-coordinate labels in mm
    if axis == 0:  # x-axis: fix y, z
        line = image[:, coord1, coord2]
        axis_label = 'X'
        ny, nz = image.shape[1], image.shape[2]
        c1_mm = (coord1 - (ny - 1) / 2) * voxel_size
        c2_mm = (coord2 - (nz - 1) / 2) * voxel_size
        fixed_label = f'y={c1_mm:.1f}mm, z={c2_mm:.1f}mm'
    elif axis == 1:  # y-axis: fix x, z
        line = image[coord1, :, coord2]
        axis_label = 'Y'
        nx, nz = image.shape[0], image.shape[2]
        c1_mm = (coord1 - (nx - 1) / 2) * voxel_size
        c2_mm = (coord2 - (nz - 1) / 2) * voxel_size
        fixed_label = f'x={c1_mm:.1f}mm, z={c2_mm:.1f}mm'
    else:  # z-axis: fix x, y
        line = image[coord1, coord2, :]
        axis_label = 'Z'
        nx, ny = image.shape[0], image.shape[1]
        c1_mm = (coord1 - (nx - 1) / 2) * voxel_size
        c2_mm = (coord2 - (ny - 1) / 2) * voxel_size
        fixed_label = f'x={c1_mm:.1f}mm, y={c2_mm:.1f}mm'

    # Define Gaussian function
    def gaussian(x, amplitude, mean, sigma, offset):
        return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset

    # Prepare data — fit in pixel space, display in centred mm coordinates
    n = len(line)
    x_data = np.arange(n)
    # Centred mm positions: voxel i → (i - (n-1)/2) * voxel_size
    x_mm = (x_data - (n - 1) / 2) * voxel_size
    y_data = line

    # Initial guesses
    offset_guess = np.min(line)
    amplitude_guess = np.max(line) - offset_guess
    mean_guess = np.argmax(line)
    sigma_guess = n / 10
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    try:
        popt, pcov = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
        amplitude, mean, sigma, offset = popt

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        fwhm = 2.355 * abs(sigma)
        fwhm_mm = fwhm * voxel_size

        # FWTM = 2 * sqrt(2 * ln(10)) * sigma ≈ 4.292 * sigma
        fwtm = 4.292 * abs(sigma)
        fwtm_mm = fwtm * voxel_size

        # Gaussian centre in centred mm space
        mean_mm = (mean - (n - 1) / 2) * voxel_size

        fit_success = True
    except RuntimeError:
        print("Warning: Gaussian fit failed to converge")
        fwhm = fwhm_mm = fwtm = fwtm_mm = amplitude = mean = mean_mm = sigma = offset = None
        fit_success = False

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_mm, y_data, 'b-', linewidth=2, label='Data')

    if fit_success:
        # Dense x for smooth fit curve (in pixel space, convert to centred mm)
        x_fit = np.linspace(0, n - 1, 500)
        x_fit_mm = (x_fit - (n - 1) / 2) * voxel_size
        y_fit = gaussian(x_fit, *popt)
        ax.plot(x_fit_mm, y_fit, 'r--', linewidth=2,
                label=f'Gaussian fit (FWHM = {fwhm_mm:.2f} mm, FWTM = {fwtm_mm:.2f} mm)')

        # Mark FWHM on plot
        half_max = offset + amplitude / 2
        ax.axhline(y=half_max, color='gray', linestyle=':', alpha=0.7, label='Half maximum')

        # Vertical lines at FWHM boundaries (centred mm)
        left_bound_hm = mean_mm - fwhm_mm / 2
        right_bound_hm = mean_mm + fwhm_mm / 2
        ax.axvline(x=left_bound_hm, color='green', linestyle=':', alpha=0.7)
        ax.axvline(x=right_bound_hm, color='green', linestyle=':', alpha=0.7)

        # Mark FWTM on plot
        tenth_max = offset + amplitude / 10
        ax.axhline(y=tenth_max, color='purple', linestyle=':', alpha=0.7, label='Tenth maximum')

        # Vertical lines at FWTM boundaries (centred mm)
        left_bound_tm = mean_mm - fwtm_mm / 2
        right_bound_tm = mean_mm + fwtm_mm / 2
        ax.axvline(x=left_bound_tm, color='orange', linestyle=':', alpha=0.7)
        ax.axvline(x=right_bound_tm, color='orange', linestyle=':', alpha=0.7)

    ax.set_xlabel(f'{axis_label} position (mm)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(f'Line Profile along {axis_label}-axis ({fixed_label})', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.18), ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)

    if show:
        plt.show()

    return {
        'fwhm': fwhm,
        'fwhm_mm': fwhm_mm,
        'fwtm': fwtm,
        'fwtm_mm': fwtm_mm,
        'amplitude': amplitude,
        'mean': mean,
        'mean_mm': mean_mm,
        'sigma': sigma,
        'offset': offset,
        'line': line,
        'fig': fig
    }



