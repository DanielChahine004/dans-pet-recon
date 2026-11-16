import numpy as np
import numba
import math
from numba import cuda, njit

# Tolerance for direction checks and general floating point comparisons
EPS = 1e-10
# Small nudge for plane finding to avoid landing exactly on boundaries
PLANE_NUDGE = 1e-9
# Maximum size for t-value array (must accommodate nx + ny + nz + 2)
MAX_T_VALUES = 1024
# Maximum voxels a ray can traverse. A ray crosses a plane independently in each
# axis, so the count is bounded by nx + ny + nz (NOT the diagonal length
# sqrt(3)*max_dim). For a 256^3 grid that is ~768; 1024 covers it with margin.
MAX_VOXELS = 1024


@cuda.jit(device=True)
def siddon_cuda(x1, y1, z1, x2, y2, z2, grid_size, voxel_size, grid_origin,
                voxel_indices, lengths):
    """
    Siddon's algorithm for CUDA using the Midpoint Method.

    Instead of direction-based boundary handling, this computes the midpoint
    of each ray segment and uses that to unambiguously determine the voxel.
    This is more robust to floating point errors at voxel boundaries.

    Returns count of intersected voxels.
    """
    nx, ny, nz = grid_size
    x0, y0, z0 = grid_origin

    # Direction vector
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Ray length
    ray_length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if ray_length < EPS:
        return 0

    # Normalize direction
    dx /= ray_length
    dy /= ray_length
    dz /= ray_length

    if abs(dx) > EPS:
        tx_min = (x0 - x1) / dx
        tx_max = (x0 + nx * voxel_size - x1) / dx
        if tx_min > tx_max:
            tx_min, tx_max = tx_max, tx_min
    else:
        tx_min = -1e10
        tx_max = 1e10

    if abs(dy) > EPS:
        ty_min = (y0 - y1) / dy
        ty_max = (y0 + ny * voxel_size - y1) / dy
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min
    else:
        ty_min = -1e10
        ty_max = 1e10

    if abs(dz) > EPS:
        tz_min = (z0 - z1) / dz
        tz_max = (z0 + nz * voxel_size - z1) / dz
        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min
    else:
        tz_min = -1e10
        tz_max = 1e10

    # Overall entry and exit
    t_min = max(max(tx_min, ty_min), tz_min)
    t_max = min(min(tx_max, ty_max), tz_max)

    if t_min >= t_max or t_max < 0 or t_min > ray_length:
        return 0

    t_min = max(t_min, 0.0)
    t_max = min(t_max, ray_length)

    # Collect all t-intersection values in the valid range [t_min, t_max]
    # Maximum intersections: nx + ny + nz + 2 (for t_min and t_max)
    t_values = cuda.local.array(1024, numba.float64)
    n_t = 0

    # Add t_min
    t_values[n_t] = t_min
    n_t += 1

    # Add X-plane intersections
    if abs(dx) > EPS:
        t_delta_x = voxel_size / abs(dx)
        # Find first x-plane intersection after t_min
        if dx > 0:
            # Moving in +x direction, planes are at x0 + i*voxel_size
            first_plane_x = math.ceil((x1 + t_min * dx - x0 + PLANE_NUDGE) / voxel_size)
            t_first_x = (x0 + first_plane_x * voxel_size - x1) / dx
        else:
            # Moving in -x direction
            first_plane_x = math.floor((x1 + t_min * dx - x0 - PLANE_NUDGE) / voxel_size)
            t_first_x = (x0 + first_plane_x * voxel_size - x1) / dx

        t_x = t_first_x
        while t_x < t_max and n_t < 1023:
            if t_x > t_min:
                t_values[n_t] = t_x
                n_t += 1
            t_x += t_delta_x

    # Add Y-plane intersections
    if abs(dy) > EPS:
        t_delta_y = voxel_size / abs(dy)
        if dy > 0:
            first_plane_y = math.ceil((y1 + t_min * dy - y0 + PLANE_NUDGE) / voxel_size)
            t_first_y = (y0 + first_plane_y * voxel_size - y1) / dy
        else:
            first_plane_y = math.floor((y1 + t_min * dy - y0 - PLANE_NUDGE) / voxel_size)
            t_first_y = (y0 + first_plane_y * voxel_size - y1) / dy

        t_y = t_first_y
        while t_y < t_max and n_t < 1023:
            if t_y > t_min:
                t_values[n_t] = t_y
                n_t += 1
            t_y += t_delta_y

    # Add Z-plane intersections
    if abs(dz) > EPS:
        t_delta_z = voxel_size / abs(dz)
        if dz > 0:
            first_plane_z = math.ceil((z1 + t_min * dz - z0 + PLANE_NUDGE) / voxel_size)
            t_first_z = (z0 + first_plane_z * voxel_size - z1) / dz
        else:
            first_plane_z = math.floor((z1 + t_min * dz - z0 - PLANE_NUDGE) / voxel_size)
            t_first_z = (z0 + first_plane_z * voxel_size - z1) / dz

        t_z = t_first_z
        while t_z < t_max and n_t < 1023:
            if t_z > t_min:
                t_values[n_t] = t_z
                n_t += 1
            t_z += t_delta_z

    # Add t_max
    t_values[n_t] = t_max
    n_t += 1

    # Sort t_values using insertion sort (efficient for small arrays, no recursion)
    for i in range(1, n_t):
        key = t_values[i]
        j = i - 1
        while j >= 0 and t_values[j] > key:
            t_values[j + 1] = t_values[j]
            j -= 1
        t_values[j + 1] = key

    # Remove duplicates (values within EPS of each other)
    if n_t > 0:
        write_idx = 1
        for i in range(1, n_t):
            if t_values[i] - t_values[write_idx - 1] > EPS:
                t_values[write_idx] = t_values[i]
                write_idx += 1
        n_t = write_idx

    # Traverse segments using midpoint method
    count = 0
    max_voxels = len(voxel_indices)

    for i in range(n_t - 1):
        t_start = t_values[i]
        t_end = t_values[i + 1]
        length = t_end - t_start

        if length < EPS:
            continue

        # Compute midpoint of segment
        t_mid = 0.5 * (t_start + t_end)
        x_mid = x1 + t_mid * dx
        y_mid = y1 + t_mid * dy
        z_mid = z1 + t_mid * dz

        # Determine voxel from midpoint (simple floor - midpoint is always inside)
        ix = int(math.floor((x_mid - x0) / voxel_size))
        iy = int(math.floor((y_mid - y0) / voxel_size))
        iz = int(math.floor((z_mid - z0) / voxel_size))

        # Check bounds
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            continue

        # Store voxel and length
        if count < max_voxels:
            voxel_indices[count] = ix * ny * nz + iy * nz + iz
            lengths[count] = length
            count += 1

    return count

# ---------------------------------------------------------------------------
# Incremental traversal (used by all projection kernels)
#
# siddon_cuda() above collects every plane crossing into a 1024-element local
# array, insertion-sorts it and then walks the segments. That needs ~20 KB of
# local memory per thread (spilled to device memory) and an O(n^2) sort per ray.
# siddon_trace() produces the SAME segments without any arrays: the x-, y- and
# z-plane crossings are each already sorted, so they are merged on the fly.
# The crossing values are generated exactly as in siddon_cuda (same first-plane
# rule, same repeated addition of t_delta), segments closer than EPS are dropped
# exactly like its duplicate removal, and the voxel is chosen by the same
# midpoint rule, so the two give the same voxels and lengths. siddon_cuda() is
# kept for the scatter code and the tests.
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _first_plane_t(p1, d, t_min, origin, voxel_size):
    """t of the first plane crossing after t_min along one axis (same rule as siddon_cuda)."""
    if d > 0:
        plane = math.ceil((p1 + t_min * d - origin + PLANE_NUDGE) / voxel_size)
    else:
        plane = math.floor((p1 + t_min * d - origin - PLANE_NUDGE) / voxel_size)
    return (origin + plane * voxel_size - p1) / d


@cuda.jit(device=True)
def siddon_trace(x1, y1, z1, x2, y2, z2, nx, ny, nz, voxel_size, x0, y0, z0,
                 image, out, weight, backproject):
    """Trace one ray through the grid.

    backproject=True : out[voxel] += weight * length for every intersected voxel (atomic); returns 0.
    backproject=False: returns sum(image[voxel] * length) (forward projection); out is not used.
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    ray_length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if ray_length < EPS:
        return 0.0
    dx /= ray_length
    dy /= ray_length
    dz /= ray_length

    big = 1e10
    if abs(dx) > EPS:
        tx_min = (x0 - x1) / dx
        tx_max = (x0 + nx * voxel_size - x1) / dx
        if tx_min > tx_max:
            tx_min, tx_max = tx_max, tx_min
    else:
        tx_min = -big
        tx_max = big
    if abs(dy) > EPS:
        ty_min = (y0 - y1) / dy
        ty_max = (y0 + ny * voxel_size - y1) / dy
        if ty_min > ty_max:
            ty_min, ty_max = ty_max, ty_min
    else:
        ty_min = -big
        ty_max = big
    if abs(dz) > EPS:
        tz_min = (z0 - z1) / dz
        tz_max = (z0 + nz * voxel_size - z1) / dz
        if tz_min > tz_max:
            tz_min, tz_max = tz_max, tz_min
    else:
        tz_min = -big
        tz_max = big

    t_min = max(max(tx_min, ty_min), tz_min)
    t_max = min(min(tx_max, ty_max), tz_max)
    if t_min >= t_max or t_max < 0 or t_min > ray_length:
        return 0.0
    t_min = max(t_min, 0.0)
    t_max = min(t_max, ray_length)

    # Heads of the three crossing sequences; inf = exhausted. Crossings <= t_min are skipped
    # and crossings >= t_max are not used, as in siddon_cuda.
    inf = math.inf
    tx = inf
    sx = 0.0
    if abs(dx) > EPS:
        sx = voxel_size / abs(dx)
        tx = _first_plane_t(x1, dx, t_min, x0, voxel_size)
        while tx <= t_min:
            tx += sx
        if tx >= t_max:
            tx = inf
    ty = inf
    sy = 0.0
    if abs(dy) > EPS:
        sy = voxel_size / abs(dy)
        ty = _first_plane_t(y1, dy, t_min, y0, voxel_size)
        while ty <= t_min:
            ty += sy
        if ty >= t_max:
            ty = inf
    tz = inf
    sz = 0.0
    if abs(dz) > EPS:
        sz = voxel_size / abs(dz)
        tz = _first_plane_t(z1, dz, t_min, z0, voxel_size)
        while tz <= t_min:
            tz += sz
        if tz >= t_max:
            tz = inf

    # Voxel index: from the midpoint of the first segment (as siddon_cuda does for every segment), then stepped by
    # +-1 on the axis of every crossing consumed after it, including crossings dropped as duplicates (they are real
    # plane crossings; the midpoint of the next segment lies beyond them), instead of three floors per segment.
    stx = 1 if dx > 0 else -1
    sty = 1 if dy > 0 else -1
    stz = 1 if dz > 0 else -1
    have_idx = False
    ix = 0
    iy = 0
    iz = 0
    acc = 0.0
    t_prev = t_min
    done = False
    while not done:
        # next crossing in sorted order, or the exit point
        if tx <= ty and tx <= tz:
            t_next = tx
            axis = 0
        elif ty <= tz:
            t_next = ty
            axis = 1
        else:
            t_next = tz
            axis = 2
        if t_next == inf:
            t_next = t_max
            axis = 3
            done = True
        else:
            if axis == 0:
                tx += sx
                if tx >= t_max:
                    tx = inf
            elif axis == 1:
                ty += sy
                if ty >= t_max:
                    ty = inf
            else:
                tz += sz
                if tz >= t_max:
                    tz = inf
        length = t_next - t_prev
        # length <= EPS: duplicate crossing (or exit point equal to the last crossing): no segment, t_prev kept
        if length > EPS:
            # Axes without tracked crossings (|d| <= EPS) keep the midpoint rule on every segment: such a ray can lie
            # within rounding of a voxel boundary plane (e.g. diametric grid LORs on x = 0) and siddon_cuda assigns
            # each segment by the sign of that rounding, which stepping alone would not reproduce.
            if not have_idx or sx == 0.0 or sy == 0.0 or sz == 0.0:
                t_mid = 0.5 * (t_prev + t_next)
                if not have_idx or sx == 0.0:
                    ix = int(math.floor((x1 + t_mid * dx - x0) / voxel_size))
                if not have_idx or sy == 0.0:
                    iy = int(math.floor((y1 + t_mid * dy - y0) / voxel_size))
                if not have_idx or sz == 0.0:
                    iz = int(math.floor((z1 + t_mid * dz - z0) / voxel_size))
                have_idx = True
            t_prev = t_next
            if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                v = ix * ny * nz + iy * nz + iz
                if backproject:
                    cuda.atomic.add(out, v, weight * length)
                else:
                    acc += image[v] * length
        # the crossing just consumed moves the ray into the next voxel along its axis
        if have_idx:
            if axis == 0:
                ix += stx
            elif axis == 1:
                iy += sty
            elif axis == 2:
                iz += stz
    return acc


@cuda.jit
def backproject_kernel(lors, backproj, nx, ny, nz, voxel_size, x0, y0, z0):
    """Backproject ones: accumulate the intersection lengths of every LOR."""
    idx = cuda.grid(1)
    if idx >= lors.shape[0]:
        return
    siddon_trace(lors[idx, 0], lors[idx, 1], lors[idx, 2], lors[idx, 3], lors[idx, 4], lors[idx, 5],
                 nx, ny, nz, voxel_size, x0, y0, z0, backproj, backproj, 1.0, True)


@cuda.jit
def forward_project_kernel(lors, image, projections, nx, ny, nz, voxel_size, x0, y0, z0):
    """Forward project: compute line integrals through image for each LOR."""
    idx = cuda.grid(1)
    if idx >= lors.shape[0]:
        return
    projections[idx] = siddon_trace(lors[idx, 0], lors[idx, 1], lors[idx, 2], lors[idx, 3], lors[idx, 4], lors[idx, 5],
                                    nx, ny, nz, voxel_size, x0, y0, z0, image, image, 0.0, False)


@cuda.jit
def backproject_weighted_kernel(lors, weights, backproj, nx, ny, nz, voxel_size, x0, y0, z0):
    """Backproject with per-LOR weights (for OSEM ratio backprojection)."""
    idx = cuda.grid(1)
    if idx >= lors.shape[0]:
        return
    siddon_trace(lors[idx, 0], lors[idx, 1], lors[idx, 2], lors[idx, 3], lors[idx, 4], lors[idx, 5],
                 nx, ny, nz, voxel_size, x0, y0, z0, backproj, backproj, weights[idx], True)
