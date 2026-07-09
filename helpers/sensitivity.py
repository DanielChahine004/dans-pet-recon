import numpy as np


def filter_lors_by_fov(lors, fov_radius, fov_axial_half_length=None, buffer=0.0):
    """
    Filter LORs to keep only those physically passing through a cylindrical FOV.

    Computes ray-cylinder intersection to find where each LOR enters/exits the
    transaxial FOV cylinder, then checks if that portion overlaps the axial FOV.

    Args:
        lors: (N, 6) array of LOR endpoints [x1, y1, z1, x2, y2, z2]
        fov_radius: Transaxial FOV radius in mm
        fov_axial_half_length: Half the axial FOV extent in mm (optional).
        buffer: Extra margin in mm added to FOV bounds (for partial voxel overlap)

    Returns:
        Filtered LOR array
    """
    x1, y1, z1 = lors[:, 0], lors[:, 1], lors[:, 2]
    x2, y2, z2 = lors[:, 3], lors[:, 4], lors[:, 5]

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

    r_fov = fov_radius + buffer
    n = len(lors)

    # === Ray-cylinder intersection in XY plane ===
    # Parameterize LOR as P(t) = P1 + t*(P2-P1), t in [0, 1]
    # Solve: |P1_xy + t * d_xy|^2 = r_fov^2
    # Expands to: a*t^2 + b*t + c = 0
    a = dx**2 + dy**2
    b = 2 * (x1 * dx + y1 * dy)
    c = x1**2 + y1**2 - r_fov**2

    discriminant = b**2 - 4 * a * c

    # Default t range is full segment [0, 1]
    t_enter = np.zeros(n)
    t_exit = np.ones(n)

    # Vertical LORs (a ≈ 0): parallel to z-axis, check if inside cylinder
    vertical = a < 1e-10
    r1_sq = x1**2 + y1**2

    # Non-vertical LORs with valid intersection (discriminant >= 0)
    non_vert = ~vertical
    valid_disc = discriminant >= 0
    computable = non_vert & valid_disc

    # Compute intersection t-values
    sqrt_disc = np.sqrt(np.where(computable, discriminant, 0))
    a_safe = np.where(computable, a, 1.0)  # avoid div by zero

    t1_raw = np.where(computable, (-b - sqrt_disc) / (2 * a_safe), 0.0)
    t2_raw = np.where(computable, (-b + sqrt_disc) / (2 * a_safe), 1.0)

    # Clamp to segment [0, 1]
    t1 = np.clip(t1_raw, 0.0, 1.0)
    t2 = np.clip(t2_raw, 0.0, 1.0)

    t_enter = np.where(computable, t1, t_enter)
    t_exit = np.where(computable, t2, t_exit)

    # === Build validity mask ===
    # Vertical: valid if inside cylinder
    # Non-vertical: valid if has intersection and segment is non-empty
    valid = np.where(vertical, r1_sq <= r_fov**2,
                     valid_disc & (t_enter < t_exit))

    # === Axial filtering ===
    if fov_axial_half_length is not None:
        z_half = fov_axial_half_length + buffer

        # z-coordinates at entry/exit of transaxial cylinder
        z_at_enter = z1 + t_enter * dz
        z_at_exit = z1 + t_exit * dz

        z_min = np.minimum(z_at_enter, z_at_exit)
        z_max = np.maximum(z_at_enter, z_at_exit)

        # Check overlap: [z_min, z_max] ∩ [-z_half, z_half] ≠ ∅
        axial_valid = (z_min <= z_half) & (z_max >= -z_half)
        valid = valid & axial_valid

    n_filtered = n - np.sum(valid)
    print(f"Filtered {n_filtered:_} LORs ({100*n_filtered/n:.1f}%) outside FOV")

    return lors[valid]


def generate_scanner_lors(n_rings, n_det_per_ring, radius, axial_length, balance_directions=False, seed=42):
    """
    Generate all possible LORs between detector pairs in a PET scanner.

    Args:
        n_rings: Number of axial rings
        n_det_per_ring: Number of detectors per ring
        radius: Inner radius of scanner in mm
        axial_length: Axial length of scanner in mm
        balance_directions: If True, randomly swap endpoints to balance ray directions.
            No longer needed after Siddon algorithm fix, but kept for verification.
        seed: Random seed for reproducibility when balance_directions=True

    Returns:
        np.ndarray: Array of shape (n, 6) with [x1, y1, z1, x2, y2, z2] for each LOR
    """

    # Calculate all detector positions
    n_total_detectors = n_rings * n_det_per_ring
    detector_positions = np.zeros((n_total_detectors, 3), dtype=np.float32)

    # Angular positions for detectors in a ring
    angles = np.linspace(0, 2 * np.pi, n_det_per_ring, endpoint=False)

    # Axial positions for each ring (centered at z=0)
    z_positions = np.linspace(-axial_length / 2, axial_length / 2, n_rings)

    # Fill detector positions
    for ring_idx in range(n_rings):
        for det_idx in range(n_det_per_ring):
            global_idx = ring_idx * n_det_per_ring + det_idx
            detector_positions[global_idx, 0] = radius * np.cos(angles[det_idx])
            detector_positions[global_idx, 1] = radius * np.sin(angles[det_idx])
            detector_positions[global_idx, 2] = z_positions[ring_idx]

    # Generate all unique pairs (i, j) where i < j
    # Number of pairs = n * (n-1) / 2
    n_pairs = n_total_detectors * (n_total_detectors - 1) // 2
    print(f"Generating {n_pairs:_} scanner LORs from {n_total_detectors:_} detectors...")

    # Use numpy to generate pairs efficiently
    idx1, idx2 = np.triu_indices(n_total_detectors, k=1)

    # Build the (n, 6) array
    scanner_lors = np.hstack([
        detector_positions[idx1],  # (n_pairs, 3)
        detector_positions[idx2]   # (n_pairs, 3)
    ])

    # Balance ray directions by randomly swapping endpoints
    # This prevents directional bias in algorithms like Siddon that trace from point1 to point2
    if balance_directions:
        rng = np.random.default_rng(seed)
        swap_mask = rng.random(n_pairs) < 0.5
        # Swap columns [0:3] with [3:6] where mask is True
        scanner_lors[swap_mask] = np.hstack([
            scanner_lors[swap_mask, 3:6],
            scanner_lors[swap_mask, 0:3]
        ])
        print(f"Balanced LOR directions: {np.sum(swap_mask):_} endpoints swapped")

    return scanner_lors


def total_scanner_lors(n_rings, n_det_per_ring):
    """Return total number of unique LOR pairs for a scanner without allocating anything."""
    n = n_rings * n_det_per_ring
    return n * (n - 1) // 2


def iter_scanner_lors_chunked(n_rings, n_det_per_ring, radius, axial_length,
                               chunk_size=5_000_000, balance_directions=False, seed=42):
    """
    Generator that yields LOR chunks without ever building the full pair array.

    Builds detector positions once (n × 3, always small), then streams detector
    pairs row by row, yielding chunks of approximately chunk_size LORs.

    Peak CPU RAM = chunk_size × 6 × 4 bytes  (e.g. 120 MB for 5 M LORs).

    Args:
        n_rings: Number of axial rings
        n_det_per_ring: Number of detectors per ring
        radius: Inner radius of scanner in mm
        axial_length: Axial length of scanner in mm
        chunk_size: Maximum number of LOR pairs per yielded chunk
        balance_directions: If True, randomly swap endpoints (same semantics as
            generate_scanner_lors). Uses a single shared RNG so swap decisions
            are reproducible across the full stream.
        seed: Random seed used when balance_directions=True

    Yields:
        np.ndarray of shape (m, 6) float32, m <= chunk_size
    """
    n = n_rings * n_det_per_ring
    n_total_pairs = n * (n - 1) // 2
    print(f"Streaming {n_total_pairs:_} scanner LORs in chunks of {chunk_size:_}...")

    # Build detector positions — always small (n × 3 float32)
    angles = np.linspace(0, 2 * np.pi, n_det_per_ring, endpoint=False)
    z_positions = np.linspace(-axial_length / 2, axial_length / 2, n_rings)
    detector_positions = np.zeros((n, 3), dtype=np.float32)
    for ring_idx in range(n_rings):
        for det_idx in range(n_det_per_ring):
            g = ring_idx * n_det_per_ring + det_idx
            detector_positions[g, 0] = radius * np.cos(angles[det_idx])
            detector_positions[g, 1] = radius * np.sin(angles[det_idx])
            detector_positions[g, 2] = z_positions[ring_idx]

    rng = np.random.default_rng(seed) if balance_directions else None

    def _flush(parts_i, parts_j):
        idx1 = np.concatenate(parts_i)
        idx2 = np.concatenate(parts_j)
        chunk = np.hstack([detector_positions[idx1], detector_positions[idx2]])
        if balance_directions:
            swap = rng.random(len(chunk)) < 0.5
            chunk[swap] = np.hstack([chunk[swap, 3:], chunk[swap, :3]])
        return chunk

    buf_i, buf_j, buf_count = [], [], 0

    for i in range(n - 1):
        row_size = n - 1 - i  # pairs: (i, i+1), (i, i+2), ..., (i, n-1)

        # Yield accumulated buffer before it would exceed chunk_size
        if buf_count > 0 and buf_count + row_size > chunk_size:
            yield _flush(buf_i, buf_j)
            buf_i, buf_j, buf_count = [], [], 0

        if row_size > chunk_size:
            # Single row exceeds chunk_size — sub-divide along j
            for sub_start in range(i + 1, n, chunk_size):
                sub_end = min(sub_start + chunk_size, n)
                sub_j = np.arange(sub_start, sub_end, dtype=np.int32)
                sub_i = np.full(len(sub_j), i, dtype=np.int32)
                yield _flush([sub_i], [sub_j])
        else:
            buf_i.append(np.full(row_size, i, dtype=np.int32))
            buf_j.append(np.arange(i + 1, n, dtype=np.int32))
            buf_count += row_size

    if buf_count > 0:
        yield _flush(buf_i, buf_j)
