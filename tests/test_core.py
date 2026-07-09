"""
Unit tests for core reconstruction functions.

Tests the deterministic, pure-logic components that are critical to correctness
but hard to debug visually: Siddon ray-tracing, sinogram coordinate computation,
scanner geometry generation, FOV filtering, and attenuation correction factors.

Run from the Reconstructions/ directory:
    python -m tests.test_core
"""

import unittest
import numpy as np
import sys
import os
import math

# Ensure helpers/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numba
from numba import cuda
from helpers.siddon import (
    siddon_cuda, forward_project_kernel,
    backproject_kernel, backproject_weighted_kernel,
)
from helpers.sensitivity import generate_scanner_lors, filter_lors_by_fov
from helpers.scatter import _bin_lors_to_sinogram, scatter_sinogram_to_lor


# ---------------------------------------------------------------------------
# Helper: run Siddon on GPU for a single LOR and return voxel indices + lengths
# ---------------------------------------------------------------------------
@cuda.jit
def _test_siddon_kernel(lors, grid_size, voxel_size, grid_origin,
                        out_indices, out_lengths, out_counts):
    """Test harness kernel: run siddon_cuda and write results back."""
    idx = cuda.grid(1)
    if idx >= lors.shape[0]:
        return

    voxels = cuda.local.array(1024, numba.int32)
    lens = cuda.local.array(1024, numba.float64)

    count = siddon_cuda(
        lors[idx, 0], lors[idx, 1], lors[idx, 2],
        lors[idx, 3], lors[idx, 4], lors[idx, 5],
        grid_size, voxel_size, grid_origin,
        voxels, lens,
    )

    out_counts[idx] = count
    for i in range(count):
        out_indices[idx, i] = voxels[i]
        out_lengths[idx, i] = lens[i]


def run_siddon(lors_np, grid_size, voxel_size):
    """Run Siddon on GPU for an array of LORs. Returns (indices, lengths, counts)."""
    nx, ny, nz = grid_size
    x0 = -nx * voxel_size / 2.0
    y0 = -ny * voxel_size / 2.0
    z0 = -nz * voxel_size / 2.0

    n = lors_np.shape[0]
    max_voxels = 1024

    d_lors = cuda.to_device(lors_np.astype(np.float32))
    d_indices = cuda.device_array((n, max_voxels), dtype=np.int32)
    d_lengths = cuda.device_array((n, max_voxels), dtype=np.float64)
    d_counts = cuda.device_array(n, dtype=np.int32)

    threads = 256
    blocks = (n + threads - 1) // threads
    _test_siddon_kernel[blocks, threads](
        d_lors, grid_size, voxel_size, (x0, y0, z0),
        d_indices, d_lengths, d_counts,
    )

    return d_indices.copy_to_host(), d_lengths.copy_to_host(), d_counts.copy_to_host()


def run_forward_project(lors_np, image, grid_size, voxel_size):
    """Forward project through an image for a set of LORs."""
    nx, ny, nz = grid_size
    x0 = -nx * voxel_size / 2.0
    y0 = -ny * voxel_size / 2.0
    z0 = -nz * voxel_size / 2.0

    n = lors_np.shape[0]
    d_lors = cuda.to_device(lors_np.astype(np.float32))
    d_image = cuda.to_device(image.flatten().astype(np.float32))
    d_proj = cuda.device_array(n, dtype=np.float32)

    threads = 256
    blocks = (n + threads - 1) // threads
    forward_project_kernel[blocks, threads](
        d_lors, d_image, d_proj,
        nx, ny, nz, voxel_size, x0, y0, z0,
    )
    return d_proj.copy_to_host()


def run_backproject(lors_np, grid_size, voxel_size):
    """Backproject LORs (unweighted) into an image."""
    nx, ny, nz = grid_size
    x0 = -nx * voxel_size / 2.0
    y0 = -ny * voxel_size / 2.0
    z0 = -nz * voxel_size / 2.0
    n_voxels = nx * ny * nz
    n = lors_np.shape[0]

    d_lors = cuda.to_device(lors_np.astype(np.float32))
    d_bp = cuda.to_device(np.zeros(n_voxels, dtype=np.float32))

    threads = 256
    blocks = (n + threads - 1) // threads
    backproject_kernel[blocks, threads](
        d_lors, d_bp, nx, ny, nz, voxel_size, x0, y0, z0,
    )
    return d_bp.copy_to_host().reshape(grid_size)


# ===========================================================================
# Test Cases
# ===========================================================================

class TestSiddonRayTracing(unittest.TestCase):
    """Tests for the core Siddon algorithm (siddon_cuda)."""

    def test_axial_ray_through_center(self):
        """A ray along the x-axis through center of a small grid."""
        grid_size = (4, 4, 4)
        voxel_size = 10.0
        # Ray from (-100, 0, 0) to (100, 0, 0) — along x through center
        # Grid spans [-20, 20] in each axis. Ray crosses 4 voxels in x.
        lors = np.array([[-100, 0, 0, 100, 0, 0]], dtype=np.float32)
        indices, lengths, counts = run_siddon(lors, grid_size, voxel_size)

        count = counts[0]
        self.assertEqual(count, 4, f"Expected 4 voxels crossed, got {count}")

        # Each voxel should have length = voxel_size
        total_length = np.sum(lengths[0, :count])
        self.assertAlmostEqual(total_length, 4 * voxel_size, places=2,
                               msg=f"Total path length should be {4*voxel_size}, got {total_length}")

        for i in range(count):
            self.assertAlmostEqual(lengths[0, i], voxel_size, places=2,
                                   msg=f"Voxel {i} length should be {voxel_size}, got {lengths[0, i]}")

    def test_ray_misses_grid(self):
        """A ray that completely misses the grid returns count=0."""
        grid_size = (4, 4, 4)
        voxel_size = 10.0
        # Ray far above the grid
        lors = np.array([[-100, 500, 500, 100, 500, 500]], dtype=np.float32)
        _, _, counts = run_siddon(lors, grid_size, voxel_size)
        self.assertEqual(counts[0], 0)

    def test_diagonal_ray(self):
        """A diagonal ray through the grid has correct total path length."""
        grid_size = (4, 4, 4)
        voxel_size = 10.0
        half = 4 * 10 / 2  # 20mm
        # Ray from corner to corner of the grid
        lors = np.array([[-half, -half, -half, half, half, half]], dtype=np.float32)
        indices, lengths, counts = run_siddon(lors, grid_size, voxel_size)

        count = counts[0]
        self.assertGreater(count, 0, "Diagonal ray should hit some voxels")

        total_length = np.sum(lengths[0, :count])
        expected = math.sqrt(3) * 4 * voxel_size  # diagonal of 4x4x4 grid
        self.assertAlmostEqual(total_length, expected, places=1,
                               msg=f"Diagonal total length should be ~{expected:.1f}, got {total_length:.1f}")

    def test_reverse_direction_same_result(self):
        """Reversing LOR endpoints gives same voxels and total path length."""
        grid_size = (8, 8, 8)
        voxel_size = 5.0
        lors_fwd = np.array([[-50, 3, 7, 50, -5, 12]], dtype=np.float32)
        lors_rev = np.array([[50, -5, 12, -50, 3, 7]], dtype=np.float32)

        _, lengths_fwd, counts_fwd = run_siddon(lors_fwd, grid_size, voxel_size)
        _, lengths_rev, counts_rev = run_siddon(lors_rev, grid_size, voxel_size)

        self.assertEqual(counts_fwd[0], counts_rev[0],
                         "Forward and reverse should cross same number of voxels")

        total_fwd = np.sum(lengths_fwd[0, :counts_fwd[0]])
        total_rev = np.sum(lengths_rev[0, :counts_rev[0]])
        self.assertAlmostEqual(total_fwd, total_rev, places=3,
                               msg="Forward and reverse total lengths should match")

    def test_zero_length_ray(self):
        """A zero-length ray (same start and end) returns count=0."""
        grid_size = (4, 4, 4)
        voxel_size = 10.0
        lors = np.array([[5, 5, 5, 5, 5, 5]], dtype=np.float32)
        _, _, counts = run_siddon(lors, grid_size, voxel_size)
        self.assertEqual(counts[0], 0)

    def test_long_oblique_not_truncated(self):
        """An off-axis oblique ray through a 256^3 grid crosses >512 voxels.

        Regression test for the voxel-buffer truncation bug. The number of
        voxels a ray traverses is bounded by nx+ny+nz (~768 here), not by the
        diagonal length sqrt(3)*256 (~443). A *perfect* body diagonal collapses
        to 256 voxels (crossings coincide at corners), so this uses an off-axis
        ray whose x/y/z plane crossings interleave — it crosses ~725 voxels.
        With the old 512-element buffer the far end of the ray was silently
        dropped, biasing long oblique line integrals; the full chord length
        must be recovered.
        """
        grid_size = (256, 256, 256)
        voxel_size = 1.0
        p1 = np.array([-200.0, -190.0, -180.0])
        p2 = np.array([200.0, 188.0, 175.0])
        lors = np.array([[*p1, *p2]], dtype=np.float32)
        _, lengths, counts = run_siddon(lors, grid_size, voxel_size)

        count = counts[0]
        self.assertGreater(count, 512,
                           f"Oblique ray should cross >512 voxels, got {count}")

        # Analytic expected in-grid chord length via slab (ray-box) clipping.
        half = 256 * voxel_size / 2.0
        d = p2 - p1
        t_enter, t_exit = 0.0, 1.0
        for axis in range(3):
            t_lo = (-half - p1[axis]) / d[axis]
            t_hi = (half - p1[axis]) / d[axis]
            if t_lo > t_hi:
                t_lo, t_hi = t_hi, t_lo
            t_enter = max(t_enter, t_lo)
            t_exit = min(t_exit, t_hi)
        expected = (t_exit - t_enter) * np.linalg.norm(d)

        total_length = np.sum(lengths[0, :count])
        self.assertAlmostEqual(total_length, expected, delta=1.0,
                               msg=f"In-grid chord should be ~{expected:.1f} "
                                   f"(no truncation), got {total_length:.1f}")


class TestForwardBackproject(unittest.TestCase):
    """Tests for forward projection and backprojection kernels."""

    def test_forward_project_uniform_image(self):
        """Forward projecting through a uniform image gives value * path length."""
        grid_size = (8, 8, 8)
        voxel_size = 5.0
        value = 3.0
        image = np.full(grid_size, value, dtype=np.float32)

        # Axial ray through center: path length = 8 * 5 = 40mm
        lors = np.array([[-100, 0, 0, 100, 0, 0]], dtype=np.float32)
        proj = run_forward_project(lors, image, grid_size, voxel_size)

        expected = value * 8 * voxel_size  # 3.0 * 40 = 120.0
        self.assertAlmostEqual(proj[0], expected, places=1,
                               msg=f"Forward projection through uniform image: expected {expected}, got {proj[0]}")

    def test_forward_project_empty_image(self):
        """Forward projecting through an empty image gives zero."""
        grid_size = (8, 8, 8)
        voxel_size = 5.0
        image = np.zeros(grid_size, dtype=np.float32)
        lors = np.array([[-100, 0, 0, 100, 0, 0]], dtype=np.float32)
        proj = run_forward_project(lors, image, grid_size, voxel_size)
        self.assertAlmostEqual(proj[0], 0.0, places=5)

    def test_forward_project_single_hot_voxel(self):
        """Forward projecting through a single hot voxel gives value * path length through that voxel."""
        grid_size = (8, 8, 8)
        voxel_size = 5.0
        image = np.zeros(grid_size, dtype=np.float32)
        image[4, 4, 4] = 10.0  # Hot voxel at center-ish

        # Axial ray through the hot voxel row (y=center, z=center)
        # Voxel [4,4,4] center is at x=2.5, y=2.5, z=2.5 (grid origin at -20)
        # y=2.5 means LOR at y=2.5, z=2.5
        lors = np.array([[-100, 2.5, 2.5, 100, 2.5, 2.5]], dtype=np.float32)
        proj = run_forward_project(lors, image, grid_size, voxel_size)

        # Ray should cross exactly that voxel with path length = voxel_size
        expected = 10.0 * voxel_size
        self.assertAlmostEqual(proj[0], expected, places=1,
                               msg=f"Expected {expected}, got {proj[0]}")

    def test_backproject_symmetry(self):
        """Backprojecting symmetric LORs produces a symmetric image."""
        grid_size = (8, 8, 8)
        voxel_size = 5.0
        # Two LORs: one along +x, one along -x through center
        # They should produce identical contributions (same voxels, same lengths)
        lors = np.array([
            [-100, 0, 0, 100, 0, 0],
            [100, 0, 0, -100, 0, 0],
        ], dtype=np.float32)
        bp = run_backproject(lors, grid_size, voxel_size)

        # Image should be symmetric along x
        center_slice = bp[:, 4, 4]
        self.assertTrue(np.allclose(center_slice, center_slice[::-1], atol=1e-4),
                        "Backprojection should be symmetric for opposite-direction LORs")

    def test_forward_backproject_adjointness(self):
        """<Hx, y> == <x, H^T y> (adjoint test for system matrix)."""
        grid_size = (8, 8, 8)
        voxel_size = 5.0
        nx, ny, nz = grid_size
        x0 = -nx * voxel_size / 2.0
        y0 = -ny * voxel_size / 2.0
        z0 = -nz * voxel_size / 2.0

        rng = np.random.default_rng(42)

        # Random image x
        x = rng.random(grid_size).astype(np.float32)

        # Random LORs that cross the grid
        n_lors = 200
        angles = rng.uniform(0, 2 * np.pi, n_lors)
        z_vals = rng.uniform(-15, 15, n_lors)
        radius = 50.0
        lors = np.column_stack([
            radius * np.cos(angles), radius * np.sin(angles), z_vals,
            -radius * np.cos(angles), -radius * np.sin(angles), z_vals + rng.uniform(-5, 5, n_lors),
        ]).astype(np.float32)

        # Forward project: Hx
        Hx = run_forward_project(lors, x, grid_size, voxel_size)

        # Random sinogram weights y
        y = rng.random(n_lors).astype(np.float32)

        # Backproject weighted: H^T y
        d_lors = cuda.to_device(lors)
        d_y = cuda.to_device(y)
        d_bp = cuda.to_device(np.zeros(nx * ny * nz, dtype=np.float32))
        threads = 256
        blocks = (n_lors + threads - 1) // threads
        backproject_weighted_kernel[blocks, threads](
            d_lors, d_y, d_bp, nx, ny, nz, voxel_size, x0, y0, z0,
        )
        HTy = d_bp.copy_to_host().reshape(grid_size)

        # Adjoint: <Hx, y> should equal <x, H^T y>
        lhs = np.dot(Hx.astype(np.float64), y.astype(np.float64))
        rhs = np.sum(x.astype(np.float64) * HTy.astype(np.float64))

        rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        self.assertLess(rel_error, 0.01,
                        f"Adjoint test failed: <Hx,y>={lhs:.6f}, <x,H^Ty>={rhs:.6f}, "
                        f"relative error={rel_error:.6f}")


class TestScannerGeometry(unittest.TestCase):
    """Tests for scanner LOR generation."""

    def test_detector_count(self):
        """Number of LORs = n_detectors * (n_detectors - 1) / 2."""
        n_rings = 3
        n_det_per_ring = 10
        lors = generate_scanner_lors(n_rings, n_det_per_ring, radius=50.0, axial_length=30.0)
        n_det = n_rings * n_det_per_ring
        expected_pairs = n_det * (n_det - 1) // 2
        self.assertEqual(lors.shape[0], expected_pairs)
        self.assertEqual(lors.shape[1], 6)

    def test_detectors_at_correct_radius(self):
        """All detector positions should be at the specified radius."""
        radius = 100.0
        lors = generate_scanner_lors(2, 20, radius=radius, axial_length=50.0)

        # Check endpoint 1
        r1 = np.sqrt(lors[:, 0]**2 + lors[:, 1]**2)
        np.testing.assert_allclose(r1, radius, atol=0.01,
                                   err_msg="Endpoint 1 should be at scanner radius")

        # Check endpoint 2
        r2 = np.sqrt(lors[:, 3]**2 + lors[:, 4]**2)
        np.testing.assert_allclose(r2, radius, atol=0.01,
                                   err_msg="Endpoint 2 should be at scanner radius")

    def test_axial_span(self):
        """Detector z-positions should span [-axial_length/2, axial_length/2]."""
        axial_length = 100.0
        lors = generate_scanner_lors(5, 10, radius=50.0, axial_length=axial_length)

        all_z = np.concatenate([lors[:, 2], lors[:, 5]])
        self.assertAlmostEqual(all_z.min(), -axial_length / 2, places=1)
        self.assertAlmostEqual(all_z.max(), axial_length / 2, places=1)


class TestFOVFiltering(unittest.TestCase):
    """Tests for LOR FOV filtering."""

    def test_lor_inside_fov_kept(self):
        """A LOR passing through the center should be kept."""
        lors = np.array([[50, 0, 0, -50, 0, 0]], dtype=np.float32)
        filtered = filter_lors_by_fov(lors, fov_radius=40.0)
        self.assertEqual(len(filtered), 1)

    def test_lor_outside_fov_removed(self):
        """A LOR far outside the FOV should be removed."""
        # LOR at y=200mm, well outside any reasonable FOV
        lors = np.array([[50, 200, 0, -50, 200, 0]], dtype=np.float32)
        filtered = filter_lors_by_fov(lors, fov_radius=40.0)
        self.assertEqual(len(filtered), 0)

    def test_axial_filtering(self):
        """LORs outside axial range should be removed."""
        lors = np.array([
            [50, 0, 0, -50, 0, 0],       # z_avg = 0, should pass
            [50, 0, 500, -50, 0, 500],    # z_avg = 500, should fail
        ], dtype=np.float32)
        filtered = filter_lors_by_fov(lors, fov_radius=100.0, fov_axial_half_length=50.0)
        self.assertEqual(len(filtered), 1)

    def test_buffer_extends_fov(self):
        """Buffer should keep borderline LORs that would otherwise be filtered."""
        # LOR at y = 41mm, just outside fov_radius=40
        lors = np.array([[50, 41, 0, -50, 41, 0]], dtype=np.float32)

        filtered_no_buffer = filter_lors_by_fov(lors, fov_radius=40.0)
        filtered_with_buffer = filter_lors_by_fov(lors, fov_radius=40.0, buffer=5.0)

        self.assertEqual(len(filtered_no_buffer), 0, "Should be filtered without buffer")
        self.assertEqual(len(filtered_with_buffer), 1, "Should be kept with buffer")


class TestSinogramCoordinates(unittest.TestCase):
    """Tests for LOR-to-sinogram coordinate computation."""

    def test_horizontal_lor_through_origin(self):
        """A horizontal LOR through origin has s=0."""
        lors = np.array([[100, 0, 0, -100, 0, 0]], dtype=np.float32)

        s_edges = np.linspace(-120, 120, 33)
        phi_edges = np.linspace(0, np.pi, 17)
        z_avg_edges = np.linspace(-150, 150, 11)
        z_diff_edges = np.linspace(-300, 300, 11)

        sino = _bin_lors_to_sinogram(lors, s_edges, phi_edges, z_avg_edges, z_diff_edges)

        # Find the bin where the count landed
        nonzero = np.argwhere(sino > 0)
        self.assertEqual(len(nonzero), 1, "Exactly one bin should have a count")

        s_bin = nonzero[0, 0]
        s_center = (s_edges[s_bin] + s_edges[s_bin + 1]) / 2
        self.assertAlmostEqual(s_center, 0.0, delta=s_edges[1] - s_edges[0],
                               msg=f"s should be ~0 for LOR through origin, got bin center={s_center}")

    def test_offset_lor_has_nonzero_s(self):
        """A LOR offset from origin has |s| > 0."""
        # LOR parallel to x-axis at y=50mm
        lors = np.array([[100, 50, 0, -100, 50, 0]], dtype=np.float32)

        s_edges = np.linspace(-120, 120, 33)
        phi_edges = np.linspace(0, np.pi, 17)
        z_avg_edges = np.linspace(-150, 150, 11)
        z_diff_edges = np.linspace(-300, 300, 11)

        sino = _bin_lors_to_sinogram(lors, s_edges, phi_edges, z_avg_edges, z_diff_edges)
        nonzero = np.argwhere(sino > 0)
        self.assertEqual(len(nonzero), 1)

        s_bin = nonzero[0, 0]
        s_center = (s_edges[s_bin] + s_edges[s_bin + 1]) / 2
        # s should be close to 50mm (or -50mm depending on direction convention)
        self.assertAlmostEqual(abs(s_center), 50.0, delta=s_edges[1] - s_edges[0],
                               msg=f"|s| should be ~50 for LOR at y=50mm, got |bin center|={abs(s_center)}")

    def test_roundtrip_sinogram_interpolation(self):
        """LORs binned into sinogram and interpolated back should recover counts."""
        rng = np.random.default_rng(42)
        n = 5000
        angles = rng.uniform(0, 2 * np.pi, n)
        radius = 100.0
        z_vals = rng.uniform(-100, 100, n)

        lors = np.column_stack([
            radius * np.cos(angles), radius * np.sin(angles), z_vals,
            -radius * np.cos(angles), -radius * np.sin(angles), z_vals + rng.uniform(-20, 20, n),
        ]).astype(np.float32)

        s_edges = np.linspace(-120, 120, 33)
        phi_edges = np.linspace(0, np.pi, 17)
        z_avg_edges = np.linspace(-150, 150, 16)
        z_diff_edges = np.linspace(-300, 300, 11)

        sino = _bin_lors_to_sinogram(lors, s_edges, phi_edges, z_avg_edges, z_diff_edges)

        self.assertAlmostEqual(sino.sum(), n, delta=n * 0.05,
                               msg=f"Total sinogram counts should be ~{n}, got {sino.sum()}")


class TestAttenuationCorrection(unittest.TestCase):
    """Tests for ACF computation via forward projection through mu-map."""

    def test_acf_through_water(self):
        """ACF through 160mm of water should be exp(-0.009597 * 160) ~ 0.215."""
        grid_size = (128, 128, 128)
        voxel_size = 2.0
        mu_water = 0.009597

        # Create a water cylinder (radius 80mm, fill the full grid axially)
        mu_map = np.zeros(grid_size, dtype=np.float32)
        nx, ny, nz = grid_size
        half = nx * voxel_size / 2.0
        for ix in range(nx):
            for iy in range(ny):
                x = -half + (ix + 0.5) * voxel_size
                y = -half + (iy + 0.5) * voxel_size
                if x*x + y*y <= 80.0**2:
                    mu_map[ix, iy, :] = mu_water

        # LOR through center along x-axis (crosses ~160mm of water)
        lors = np.array([[-200, 0, 0, 200, 0, 0]], dtype=np.float32)
        mu_integral = run_forward_project(lors, mu_map, grid_size, voxel_size)
        acf = np.exp(-mu_integral[0])

        expected_acf = np.exp(-mu_water * 160.0)  # ~0.215
        self.assertAlmostEqual(acf, expected_acf, places=2,
                               msg=f"ACF through 160mm water: expected {expected_acf:.4f}, got {acf:.4f}")

    def test_acf_through_air(self):
        """ACF through empty grid (air) should be ~1.0."""
        grid_size = (16, 16, 16)
        voxel_size = 5.0
        mu_map = np.zeros(grid_size, dtype=np.float32)

        lors = np.array([[-100, 0, 0, 100, 0, 0]], dtype=np.float32)
        mu_integral = run_forward_project(lors, mu_map, grid_size, voxel_size)
        acf = np.exp(-mu_integral[0])

        self.assertAlmostEqual(acf, 1.0, places=5,
                               msg=f"ACF through air should be 1.0, got {acf}")


class TestKleinNishina(unittest.TestCase):
    """Tests for Klein-Nishina cross section values at known angles.

    scatter.py computes cos_theta = cos(∠ASB) = dot(S→A, S→B) / (|SA|·|SB|).
    Since the Compton angle θ_c satisfies cos(θ_c) = -cos(∠ASB), the formula
    E' = 511 / (2 + cos(∠ASB)) is equivalent to E' = 511 / (2 - cos(θ_c)).

    All tests below use cos_asb (the geometric angle ∠ASB) as input to match
    the convention used in scatter.py.
    """

    @staticmethod
    def kn_relative(cos_asb):
        """KN cross section as computed in scatter.py (cos_asb = cos ∠ASB)."""
        E_scattered = 511.0 / (2.0 + cos_asb)
        ratio = E_scattered / 511.0
        sin2 = 1.0 - cos_asb * cos_asb
        return 0.5 * ratio * ratio * (ratio + 1.0 / ratio - sin2)

    def test_forward_scatter(self):
        """Forward scatter (θ_c=0): A,B on opposite sides of S → ∠ASB=π, E'=511."""
        cos_asb = -1.0  # cos(π)
        E = 511.0 / (2.0 + cos_asb)
        self.assertAlmostEqual(E, 511.0, places=1,
                               msg="Forward scatter should give E'=511 keV")
        kn = self.kn_relative(cos_asb)
        self.assertAlmostEqual(kn, 1.0, places=3,
                               msg="KN at forward scatter should be 1.0")

    def test_backscatter(self):
        """Backscatter (θ_c=π): A,B on same side of S → ∠ASB=0, E'=170.3."""
        cos_asb = 1.0  # cos(0)
        E = 511.0 / (2.0 + cos_asb)  # 511/3 = 170.3
        self.assertAlmostEqual(E, 170.33, places=1)

        kn = self.kn_relative(cos_asb)
        self.assertGreater(kn, 0, "KN at backscatter should be positive")
        self.assertLess(kn, self.kn_relative(-1.0), "KN at backscatter < forward scatter")

    def test_ninety_degrees(self):
        """90° Compton (θ_c=π/2): ∠ASB=π/2, E'=255.5."""
        cos_asb = 0.0
        E = 511.0 / (2.0 + cos_asb)  # 511/2 = 255.5
        self.assertAlmostEqual(E, 255.5, places=1)

        kn = self.kn_relative(cos_asb)
        self.assertGreater(kn, 0)

    def test_energy_window(self):
        """Verify energy window (350-650 keV) with the scatter.py formula."""
        # Forward scatter (∠ASB=π, cos=-1): E'=511/(2+(-1))=511 -> inside
        self.assertTrue(350 < 511.0 / (2.0 + (-1.0)) < 650)

        # 90° Compton (∠ASB=π/2, cos=0): E'=511/(2+0)=255.5 -> outside
        self.assertFalse(350 < 511.0 / (2.0 + 0.0) < 650)

        # Backscatter (∠ASB=0, cos=1): E'=511/(2+1)=170.3 -> outside
        self.assertFalse(350 < 511.0 / (2.0 + 1.0) < 650)

        # Small Compton angle (∠ASB≈145°, cos≈-0.7): E'=511/1.3=393 -> inside
        self.assertTrue(350 < 511.0 / (2.0 + (-0.7)) < 650)

    def test_geometric_angle_to_energy(self):
        """Verify energy for concrete A, S, B positions."""
        # A at (-R, 0, 0), S at origin, B at (R, 0, 0):
        # photon passes through S with no deflection → forward scatter → E'=511
        A = np.array([-100.0, 0.0, 0.0])
        S = np.array([0.0, 0.0, 0.0])
        B = np.array([100.0, 0.0, 0.0])

        sa = A - S
        sb = B - S
        cos_asb = np.dot(sa, sb) / (np.linalg.norm(sa) * np.linalg.norm(sb))
        self.assertAlmostEqual(cos_asb, -1.0, places=5)

        E = 511.0 / (2.0 + cos_asb)
        self.assertAlmostEqual(E, 511.0, places=1,
                               msg="Collinear A-S-B (forward scatter) should give E'=511 keV")

        # A and B at same position (backscatter ∠ASB=0): E'=170.3
        A2 = np.array([100.0, 0.0, 0.0])
        B2 = np.array([100.0, 0.01, 0.0])  # nearly same direction from S
        sa2 = A2 - S
        sb2 = B2 - S
        cos_asb2 = np.dot(sa2, sb2) / (np.linalg.norm(sa2) * np.linalg.norm(sb2))
        self.assertAlmostEqual(cos_asb2, 1.0, places=3)

        E2 = 511.0 / (2.0 + cos_asb2)
        self.assertAlmostEqual(E2, 170.33, places=0,
                               msg="A,B on same side of S (backscatter) should give E'≈170 keV")

        # Right angle: A at (R,0,0), B at (0,R,0) → ∠ASB=90° → E'=255.5
        A3 = np.array([100.0, 0.0, 0.0])
        B3 = np.array([0.0, 100.0, 0.0])
        sa3 = A3 - S
        sb3 = B3 - S
        cos_asb3 = np.dot(sa3, sb3) / (np.linalg.norm(sa3) * np.linalg.norm(sb3))
        self.assertAlmostEqual(cos_asb3, 0.0, places=5)

        E3 = 511.0 / (2.0 + cos_asb3)
        self.assertAlmostEqual(E3, 255.5, places=1)


class TestScatterPerLorNormalization(unittest.TestCase):
    """Per-LOR scatter estimate = scatter fraction (scatter / occupancy).

    scatter_sinogram_to_lor() must divide the per-bin scatter total by the
    measured-LOR occupancy so the result is a per-event quantity on the same
    O(1) scale as the trues term, and is invariant to sinogram bin size.
    """

    def _make_params(self, n_s=8, n_phi=4, n_z_avg=4, n_z_diff=4):
        return {
            's_edges': np.linspace(-120, 120, n_s + 1),
            'phi_edges': np.linspace(0, np.pi, n_phi + 1),
            'z_avg_edges': np.linspace(-150, 150, n_z_avg + 1),
            'z_diff_edges': np.linspace(-300, 300, n_z_diff + 1),
        }

    def test_per_lor_equals_scatter_fraction(self):
        """s_i = scatter[bin] / occupancy[bin] (uniform field → exact)."""
        params = self._make_params()
        shape = (8, 4, 4, 4)
        # Uniform fields so linear interpolation returns the constant ratio
        # regardless of where in the bin the LOR lands.
        scatter_sino = np.full(shape, 300.0)   # 300 scatter counts per bin
        occupancy = np.full(shape, 1000.0)     # out of 1000 measured → 30%
        params['occupancy_sino'] = occupancy

        lors = np.array([[0, 100, 0, 0, -100, 0]], dtype=np.float32)
        s_per_lor = scatter_sinogram_to_lor(scatter_sino, params, lors)
        self.assertAlmostEqual(float(s_per_lor[0]), 0.30, places=4,
                               msg="Per-LOR scatter should equal scatter fraction 0.3")

    def test_bin_size_invariance(self):
        """Per-LOR estimate is invariant to sinogram bin count.

        Finer bins split the per-bin scatter AND occupancy proportionally, so
        the ratio (the per-LOR estimate) is unchanged — unlike the raw per-bin
        total, which would scale with bin size and force a different
        SSS_SCALE_FACTOR.
        """
        lors = np.array([[0, 100, 0, 0, -100, 0]], dtype=np.float32)

        def per_lor(n_s):
            params = self._make_params(n_s=n_s)
            shape = (n_s, 4, 4, 4)
            # Uniform 40% scatter fraction everywhere, but occupancy per bin
            # falls as bins get finer (fewer events per bin).
            occ = np.full(shape, 1000.0 * 8 / n_s)
            scatter = 0.4 * occ
            params['occupancy_sino'] = occ
            return float(scatter_sinogram_to_lor(scatter, params, lors)[0])

        coarse = per_lor(n_s=8)
        fine = per_lor(n_s=32)
        self.assertAlmostEqual(coarse, 0.4, places=4)
        self.assertAlmostEqual(fine, 0.4, places=4)
        self.assertAlmostEqual(coarse, fine, places=4,
                               msg="Per-LOR scatter must not depend on bin size")

    def test_empty_bin_gives_zero(self):
        """Bins with zero occupancy yield zero scatter (no div-by-zero)."""
        params = self._make_params()
        shape = (8, 4, 4, 4)
        scatter_sino = np.ones(shape) * 5.0      # scatter present everywhere
        params['occupancy_sino'] = np.zeros(shape)  # but no measured LORs

        lors = np.array([[0, 100, 0, 0, -100, 0]], dtype=np.float32)
        s_per_lor = scatter_sinogram_to_lor(scatter_sino, params, lors)
        self.assertTrue(np.all(np.isfinite(s_per_lor)))
        self.assertAlmostEqual(float(s_per_lor[0]), 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
