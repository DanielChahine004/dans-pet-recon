"""Invariant tests for the dual-engine (petrecon + OMEGA) + openSSS pipeline.

These encode what "working" means physically, so unit/normalization bugs fail
loudly instead of producing plausible-looking wrong images:

  - geometry truth: an off-center point source reconstructs at its known
    position on BOTH engines, and the engines agree with each other
  - guards: the openSSS bridge rejects mismatched/oversized/wrong-unit maps,
    the tail fit refuses degenerate support, and both runners abort when
    scatter is requested without geometric sensitivity
  - provenance: runners drop a manifest next to their output

Requires the GPU + the Opengate datasets; every test skips cleanly when the
environment lacks them.  Run:  python -m unittest tests.test_engines -v
Slow scatter-direction test gated behind RECON_SLOW_TESTS=1.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
sys.path.insert(0, str(EXAMPLES))

POINT_CSV = Path("/home/h/Code/Opengate/ragy_files/Point_Sources_Processed/"
                 "96_96_96/gt_processed_96_96_96.csv")
HAVE_DATA = POINT_CSV.exists()
try:
    import cupy
    HAVE_GPU = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAVE_GPU = False

GRID, VOXEL, N_LORS = 64, 4.0, 3000
TMP = Path(tempfile.mkdtemp(prefix="engine_tests_"))


def _run(runner, cfg):
    p = TMP / f"{runner}.pkl"
    pickle.dump(cfg, open(p, "wb"))
    return subprocess.run([sys.executable, "-u", str(EXAMPLES / runner), str(p)],
                          capture_output=True, text=True, timeout=600)


def _expected_peak(lors):
    """Least-squares LOR convergence point -> expected voxel index."""
    p1, p2 = lors[:, :3], lors[:, 3:]
    d = p2 - p1
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A, b = np.zeros((3, 3)), np.zeros(3)
    for i in range(0, len(p1), max(1, len(p1) // 500)):
        P = np.eye(3) - np.outer(d[i], d[i])
        A += P
        b += P @ p1[i]
    src_mm = np.linalg.solve(A, b)
    return src_mm / VOXEL + GRID / 2 - 0.5


@unittest.skipUnless(HAVE_DATA and HAVE_GPU, "needs GPU + Opengate point-source data")
class TestGeometryTruth(unittest.TestCase):
    """Both engines put a known off-center point source where it belongs."""

    @classmethod
    def setUpClass(cls):
        lors = np.loadtxt(POINT_CSV, delimiter=",", skiprows=1,
                          dtype=np.float32)[:N_LORS]
        cls.expected = _expected_peak(lors)
        cls.lors_npy = TMP / "lors.npy"
        np.save(cls.lors_npy, lors)

        r = _run("petrecon_runner.py", {
            "lors_npy": str(cls.lors_npy), "out_dir": str(TMP / "pr"),
            "grid": GRID, "voxel": VOXEL, "iterations": 2, "subsets": 2,
            "sensitivity": False, "scatter_npy": "", "sens_npy": ""})
        assert r.returncode == 0, f"petrecon runner failed:\n{r.stdout[-800:]}"
        cls.pr = np.load(TMP / "pr" / "reconstruction_image.npy")

        r = _run("omega_runner.py", {
            "lors_npy": str(cls.lors_npy), "out_npy": str(TMP / "om.npy"),
            "grid": GRID, "iterations": 2, "subsets": 2,
            "sens_npy": "", "scatter_npy": ""})
        assert r.returncode == 0, f"omega runner failed:\n{r.stdout[-800:]}"
        cls.om = np.load(TMP / "om.npy")

    def _assert_peak(self, vol, name):
        peak = np.array(np.unravel_index(np.argmax(vol), vol.shape), dtype=float)
        err = np.abs(peak - self.expected)
        self.assertTrue((err <= 2.0).all(),
                        f"{name} peak {peak} vs expected {self.expected} "
                        f"(voxel error {err})")

    def test_petrecon_peak_position(self):
        self._assert_peak(self.pr, "petrecon")

    def test_omega_peak_position(self):
        self._assert_peak(self.om, "OMEGA")

    def test_engines_agree(self):
        self.assertEqual(np.unravel_index(np.argmax(self.pr), self.pr.shape),
                         np.unravel_index(np.argmax(self.om), self.om.shape),
                         "engines disagree on the peak voxel")
        from scipy.ndimage import gaussian_filter as g
        a = g(self.pr / self.pr.max(), 1.5).ravel()
        b = g(self.om / self.om.max(), 1.5).ravel()
        r = float(np.corrcoef(a, b)[0, 1])
        self.assertGreater(r, 0.90, f"cross-engine correlation {r:.3f} < 0.90")

    def test_binned_peak_position(self):
        r = _run("omega_sinogram_runner.py", {
            "lors_npy": str(self.lors_npy), "out_npy": str(TMP / "bin.npy"),
            "grid": GRID, "iterations": 2, "subsets": 2})
        self.assertEqual(r.returncode, 0, f"binned runner failed:\n{r.stdout[-800:]}")
        vol = np.load(TMP / "bin.npy")
        peak = np.array(np.unravel_index(np.argmax(vol), vol.shape), dtype=float)
        # compare against the unbinned reconstruction's peak: the difference
        # IS the binning cost, expected ~1 crystal pitch (LOR-convergence
        # "expected" is scatter-biased, so it's not the right reference here)
        ref = np.array(np.unravel_index(np.argmax(self.pr), self.pr.shape), dtype=float)
        err = np.abs(peak - ref)
        self.assertTrue((err <= 2.5).all(),
                        f"binned peak {peak} vs unbinned peak {ref} "
                        f"(voxel error {err}; tolerance 2.5 = ~1 crystal pitch)")

    def test_manifests_written(self):
        m = json.loads((TMP / "pr" / "manifest.json").read_text())
        self.assertEqual(m["engine"], "petrecon")
        m = json.loads((TMP / "om.manifest.json").read_text())
        self.assertEqual(m["engine"], "omega")


class TestBridgeGuards(unittest.TestCase):
    """openSSS bridge rejects the input classes that previously hung/corrupted."""

    def test_mismatched_pool_shapes_raise(self):
        import opensss_bridge as B
        act = np.zeros((64,) * 3, dtype=np.float32)
        mu = np.full((100,) * 3, 0.0096, dtype=np.float32)   # pools to 33^3 vs 32^3
        with self.assertRaises(ValueError):
            B.estimate_scatter(act, mu)

    def test_oversized_pool_raises(self):
        import opensss_bridge as B
        act = np.zeros((256,) * 3, dtype=np.float32)
        mu = np.full((256,) * 3, 0.0096, dtype=np.float32)
        with self.assertRaises(ValueError):
            B.estimate_scatter(act, mu, downsample_to_mm=1.0)   # would stay 256^3

    def test_wrong_mu_units_raise(self):
        import opensss_bridge as B
        act = np.zeros((64,) * 3, dtype=np.float32)
        mu_cm = np.full((64,) * 3, 0.096, dtype=np.float32)    # 1/cm, not 1/mm
        with self.assertRaises(ValueError):
            B.estimate_scatter(act, mu_cm)

    def test_tail_fit_refuses_blurry_support(self):
        import opensss_bridge as B
        table = np.ones((401, 200, 1600), dtype=np.float32)
        lors = np.zeros((100, 6), dtype=np.float32)
        act = np.ones((64,) * 3, dtype=np.float32)             # support = whole FOV
        scale, tail_frac = B.tail_fit_scale(table, lors, act)
        self.assertIsNone(scale)


@unittest.skipUnless(HAVE_DATA, "needs Opengate point-source data")
class TestRunnerGuards(unittest.TestCase):
    """Runners abort loudly on physically-invalid configurations."""

    @classmethod
    def setUpClass(cls):
        lors = np.loadtxt(POINT_CSV, delimiter=",", skiprows=1,
                          dtype=np.float32)[:200]
        cls.lors_npy = TMP / "guard_lors.npy"
        np.save(cls.lors_npy, lors)
        cls.scatter_npy = TMP / "guard_scatter.npy"
        np.save(cls.scatter_npy, np.full(len(lors), 0.25, dtype=np.float32))

    def test_petrecon_rejects_scatter_without_sens(self):
        r = _run("petrecon_runner.py", {
            "lors_npy": str(self.lors_npy), "out_dir": str(TMP / "g1"),
            "grid": GRID, "voxel": VOXEL, "iterations": 1, "subsets": 1,
            "sensitivity": False, "scatter_npy": str(self.scatter_npy),
            "sens_npy": ""})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requires geometric", r.stdout + r.stderr)

    def test_omega_rejects_scatter_without_sens(self):
        r = _run("omega_runner.py", {
            "lors_npy": str(self.lors_npy), "out_npy": str(TMP / "g2.npy"),
            "grid": GRID, "iterations": 1, "subsets": 1,
            "sens_npy": "", "scatter_npy": str(self.scatter_npy)})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requires a geometric", r.stdout + r.stderr)


class TestScannerConstants(unittest.TestCase):
    """scanner.py stays in lockstep with petrecon's defaults."""

    def test_matches_petrecon_defaults(self):
        import scanner as SC
        from petrecon import ReconConfig
        cfg = ReconConfig()
        self.assertEqual(SC.RINGS, cfg.n_axial_rings)
        self.assertEqual(SC.DETS_PER_RING, cfg.n_detectors_per_ring)
        self.assertEqual(SC.INNER_RADIUS_MM, cfg.inner_radius)
        self.assertEqual(SC.AXIAL_MM, cfg.axial_length)
        self.assertEqual(SC.SECTORS_TRANS * SC.MODULES_TRANS * SC.CRYSTALS_TRANS,
                         cfg.n_detectors_per_ring)
        self.assertEqual(SC.SECTORS_AXIAL * SC.MODULES_AXIAL * SC.CRYSTALS_AXIAL,
                         cfg.n_axial_rings)


@unittest.skipUnless(os.environ.get("RECON_SLOW_TESTS") == "1",
                     "slow; set RECON_SLOW_TESTS=1 (needs sens pass, ~3 min)")
class TestScatterDirection(unittest.TestCase):
    """Scatter correction suppresses background more than hot regions (IEC)."""

    def test_direction(self):
        iec = Path("/home/h/Code/Opengate/ragy_files/IEC/iec_groundtruth_filtered.csv")
        sens256 = Path("/home/h/Code/Opengate/ragy_files/IEC/gt/sensitivity_image.npy")
        if not (iec.exists() and sens256.exists() and HAVE_GPU):
            self.skipTest("needs IEC data + cached sensitivity + GPU")
        lors = np.loadtxt(iec, delimiter=",", dtype=np.float32)
        lors_npy = TMP / "iec.npy"
        np.save(lors_npy, lors)
        s = np.load(sens256).astype(np.float32)
        n = 64
        sens = s.reshape(n, 4, n, 4, n, 4).mean(axis=(1, 3, 5))
        sens_npy = TMP / "iec_sens.npy"
        np.save(sens_npy, sens)
        import opensss_bridge as B
        act = np.load("/home/h/Code/Opengate/ragy_files/IEC/gt/reconstruction_image.npy")
        table = B.estimate_scatter(act, np.load(str(REPO / "water_attenuation_map.npy")))
        s_lor = B.scale_to_fraction(B.map_to_lors(table, lors), 0.25)
        scatter_npy = TMP / "iec_scatter.npy"
        np.save(scatter_npy, s_lor)
        vols = {}
        for tag, sc in (("off", ""), ("on", str(scatter_npy))):
            r = _run("petrecon_runner.py", {
                "lors_npy": str(lors_npy), "out_dir": str(TMP / f"iec_{tag}"),
                "grid": n, "voxel": 4.0, "iterations": 3, "subsets": 3,
                "sensitivity": True, "scatter_npy": sc, "sens_npy": str(sens_npy)})
            self.assertEqual(r.returncode, 0, r.stdout[-500:])
            vols[tag] = np.load(TMP / f"iec_{tag}" / "reconstruction_image.npy")
        a, b = vols["off"], vols["on"]
        # regression: with full-scanner sensitivity + additive scatter, the
        # count-scale normalization must prevent iterate collapse (~1e-20)
        self.assertGreater(float(b.max()), 1e-6,
                           "scatter-corrected image collapsed — count-scale "
                           "normalization of the sensitivity is broken")
        hot = a > a.max() * 0.5
        bg = (a > a.max() * 0.02) & (a < a.max() * 0.15)
        hot_change = b[hot].sum() / a[hot].sum()
        bg_change = b[bg].sum() / a[bg].sum()
        self.assertLess(bg_change, hot_change,
                        "scatter correction should suppress background more than hot")


if __name__ == "__main__":
    unittest.main()
