"""petrecon vs OMEGA — side-by-side reconstruction comparison on a danvas canvas.

Runs the same LOR data through both packages (each in its own killable
subprocess), times them, and compares the outputs: two Model3D volumes,
synced slice viewers with a normalized-difference map, through-peak line
profiles, and a timing table.

  python examples/danvas_compare.py      ->  http://127.0.0.1:8001

petrecon reconstructs with its own pipeline (run_reconstruction); OMEGA runs
list-mode OSEM with its Siddon projector via the custom-algorithm CuPy path
(see omega_runner.py). Geometry, grid, FOV, iterations, subsets and the LOR
subset are identical for both.
"""
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import danvas

from petrecon.helpers.data_io import load_lors

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# ---- Inputs ------------------------------------------------------------------
DATASETS = {
    "point 0_0_0": ("/home/h/Opengate/ragy_files/Point_Sources_Processed/"
                    "0_0_0/gt_processed_0_0_0.csv"),
    "IEC gt": "/home/h/Opengate/ragy_files/IEC/iec_groundtruth_filtered.csv",
    "IEC pred": "/home/h/Opengate/ragy_files/IEC/iec_prediction_filtered.csv",
}
MU_MAP = "/home/h/pet-recon/water_attenuation_map.npy"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "example_output_compare"
PR_OUT = OUT / "petrecon"
PR_PRE = OUT / "petrecon_prepass"          # activity pre-pass (never displayed)
OM_NPY = OUT / "omega_recon.npy"
BIN_NPY = OUT / "omega_binned.npy"
TMP = Path(tempfile.gettempdir())
SUBSET_NPY = TMP / "danvas_compare_lors.npy"
N_LINE_LORS = 400
MM = 1e-3
FOV_MM = 256.0                                # image cube, both packages

OUT.mkdir(exist_ok=True)
data = {"lors": None, "perm": None}


_dataset_cache = {}


def load_dataset(name):
    if name not in _dataset_cache:
        lors = load_lors(DATASETS[name]).astype(np.float32)
        _dataset_cache[name] = (lors, np.random.default_rng(0).permutation(len(lors)))
    data["lors"], data["perm"] = _dataset_cache[name]


load_dataset("point 0_0_0")

# ---- Canvas: two 3D viewers + three slice viewers ------------------------------
canvas = danvas.Canvas()
viewer_pr = canvas.model3d("petrecon_3d", label="petrecon", x=40, y=40, w=620, h=520)
viewer_om = canvas.model3d("omega_3d", label="OMEGA", right_of=viewer_pr, w=620, h=520)
viewer_bin = canvas.model3d("binned_3d", label="OMEGA binned (conventional)",
                            right_of=viewer_om, w=620, h=520)

hm_pr = canvas.heatmap("slice_petrecon", label="petrecon slice", below=viewer_pr, w=400, h=360)
hm_om = canvas.heatmap("slice_omega", label="OMEGA slice", right_of=hm_pr, w=400, h=360)
hm_diff = canvas.heatmap("slice_diff", label="pr–om difference", right_of=hm_om, w=400, h=360)
hm_bin = canvas.heatmap("slice_binned", label="OMEGA binned slice", right_of=hm_diff, w=400, h=360)

PLANES = {"xy": 2, "xz": 1, "yz": 0}
plane_sel = canvas.toggle(list(PLANES), name="slice_plane", label="Slice plane", below=hm_pr)
slice_idx = canvas.slider("slice_idx", label="Slice index", min=0, max=255, default=128,
                          right_of=plane_sel, w=400)
slice_pos = canvas.label("slice_pos", "no volumes yet — Run a comparison",
                         right_of=slice_idx, w=380, h="auto")
smooth_sel = canvas.toggle(["off", "on"], name="display_filter", default="off",
                           label="Display post-filter (Gaussian, all views)",
                           right_of=slice_pos, w=280)
smooth_fwhm = canvas.slider("filter_fwhm", label="Post-filter FWHM (mm)",
                            min=1.0, max=12.0, step=0.5, default=4.0,
                            right_of=smooth_sel, w=300)

profile_plot = (canvas.plot(name="profiles", label="line profiles (normalized)",
                            below=plane_sel, w=800, h=340) if go else None)
if profile_plot is not None:
    prof_axis = canvas.toggle(["x", "y", "z"], name="profile_axis",
                              label="Profile axis", right_of=profile_plot, w=300)
    prof_mode = canvas.toggle(["peak", "manual"], name="profile_mode",
                              label="Profile through", below=prof_axis, w=300)
    prof_px = canvas.slider("profile_x", label="x (mm)", min=-127, max=127,
                            default=0, below=prof_mode, w=300)
    prof_py = canvas.slider("profile_y", label="y (mm)", min=-127, max=127,
                            default=0, below=prof_px, w=300)
    prof_pz = canvas.slider("profile_z", label="z (mm)", min=-127, max=127,
                            default=0, below=prof_py, w=300)

vols = {"pr": None, "om": None, "bin": None, "voxel": 1.0}
_filter_cache = {}
_filter_lock = threading.Lock()


def display_vol(key):
    """Volume for the slice displays: raw, or (explicitly enabled) with a
    Gaussian post-filter of the selected FWHM — the standard, declared
    post-smoothing of publications. Saved outputs are never touched."""
    vol = vols[key]
    if vol is None or smooth_sel.value == "off":
        return vol
    fwhm = float(smooth_fwhm.value)
    ck = (key, fwhm, id(vol))
    with _filter_lock:                       # serialize: no duplicate filtering
        if ck not in _filter_cache:
            from scipy.ndimage import gaussian_filter
            sigma = fwhm / 2.355 / vols["voxel"]
            if len(_filter_cache) > 6:
                _filter_cache.clear()
            _filter_cache[ck] = gaussian_filter(vol, sigma)
        return _filter_cache[ck]


def update_lor_layers(n):
    subset = data["lors"][data["perm"][:n]]
    endpoints = np.concatenate([subset[:, :3], subset[:, 3:]], axis=0) * MM
    lines = subset[: min(N_LINE_LORS, n)].reshape(-1, 2, 3) * MM
    for v in (viewer_pr, viewer_om):
        v.layer("endpoints").points(endpoints, color_by=endpoints[:, 2])
        v.layer("lors").lines(lines, color=(120, 160, 220, 50))


def display_vmax(vol):
    """Publication convention: raw linear window [0, max]. No heuristics, no
    smoothing anywhere — what you see is the reconstruction as saved. (A
    volume whose display is dominated by isolated hot voxels is showing you
    a real property of that reconstruction; the runners' guards and the test
    suite exist to catch genuinely broken runs.)"""
    return float(vol.max()) or 1.0


def show_volumes(voxel_mm, fresh=True):
    if fresh:
        _filter_cache.clear()
    spacing = (voxel_mm * MM,) * 3
    for key, viewer in (("pr", viewer_pr), ("om", viewer_om), ("bin", viewer_bin)):
        vol = display_vol(key)
        if vol is None:
            continue
        origin = tuple(-n * voxel_mm * MM / 2 for n in vol.shape)
        viewer.layer("recon").volume(vol, spacing=spacing, origin=origin,
                                     vmin=0.0, vmax=display_vmax(vol), cmap="hot")
    sync_slice_controls()


def update_slices():
    ref = vols["pr"] if vols["pr"] is not None else vols["om"]
    if ref is None:
        return
    axis = PLANES[plane_sel.value]
    k = min(int(slice_idx.value), ref.shape[axis] - 1)
    a, b = (i for i in range(3) if i != axis)
    vox = vols["voxel"]
    half_a, half_b = (ref.shape[i] * vox / 2 for i in (a, b))
    extent = (-half_a, half_a, half_b, -half_b)   # heatmap y labels top row first
    norm = {}
    for key, hm in (("pr", hm_pr), ("om", hm_om), ("bin", hm_bin)):
        vol = display_vol(key)
        if vol is None:
            continue
        sl = np.take(vol, k, axis=axis).T[::-1]
        vmax = display_vmax(vol)
        hm.update(sl, vmin=0.0, vmax=vmax, cmap="hot", extent=extent)
        norm[key] = np.clip(sl / vmax, 0, 1)
    if "pr" in norm and "om" in norm:
        d = norm["pr"] - norm["om"]
        lim = float(np.abs(d).max()) or 1.0
        hm_diff.update(d, vmin=-lim, vmax=lim, cmap="gray", extent=extent)
    mm = (k + 0.5 - ref.shape[axis] / 2) * vox
    slice_pos.update(f"{plane_sel.value} slice {k}/{ref.shape[axis] - 1} — "
                     f"{'xyz'[axis]} = {mm:+.1f} mm")


def sync_slice_controls():
    ref = vols["pr"] if vols["pr"] is not None else vols["om"]
    if ref is None:
        return
    dim = ref.shape[PLANES[plane_sel.value]]
    slice_idx.max = dim - 1
    if int(slice_idx.value) >= dim:
        slice_idx.update(dim // 2)
    update_slices()


def update_profiles():
    ref = display_vol("pr") if vols["pr"] is not None else display_vol("om")
    if profile_plot is None or ref is None:
        return
    vox = vols["voxel"]
    n = ref.shape[0]
    if prof_mode.value == "peak":
        pt = np.unravel_index(np.argmax(ref), ref.shape)
    else:
        pt = tuple(int(np.clip(round(float(s.value) / vox + n / 2 - 0.5), 0, n - 1))
                   for s in (prof_px, prof_py, prof_pz))
    axis = {"x": 0, "y": 1, "z": 2}[prof_axis.value]
    fig = go.Figure()
    for key, name in (("pr", "petrecon (unbinned)"), ("om", "OMEGA (unbinned)"),
                      ("bin", "OMEGA binned")):
        vol = display_vol(key)
        if vol is None:
            continue
        idx = list(pt)
        idx[axis] = slice(None)
        prof = vol[tuple(idx)] / (float(vol.max()) or 1.0)
        coords = (np.arange(vol.shape[axis]) + 0.5 - vol.shape[axis] / 2) * vox
        fig.add_trace(go.Scatter(x=coords, y=prof, name=name, mode="lines"))
    fixed = [f"{'xyz'[i]}={((pt[i] + 0.5 - n / 2) * vox):+.0f}mm"
             for i in range(3) if i != axis]
    fig.update_layout(
        title=dict(text=f"{prof_axis.value}-profile through {', '.join(fixed)} "
                        f"({prof_mode.value})", font=dict(size=13)),
        xaxis_title=f"{prof_axis.value} (mm)", yaxis_title="normalized intensity",
        margin=dict(l=50, r=20, t=40, b=40), legend=dict(x=0.02, y=0.98))
    profile_plot.update(fig)


# ---- Controls column -------------------------------------------------------------
def section(col, text):
    slug = text.lower().replace(" ", "_").replace("&", "and")
    col.add(canvas.markdown(f"#### {text}", name=f"sec_{slug}", h="auto"))


PROGRESS_JSX = """
function Component({ props }) {
  const pct = props.pct ?? 0;
  const stage = props.stage ?? "idle";
  const color = props.failed ? "#ef4444" : (props.active ? "#f59e0b" : "#22c55e");
  return (
    <div style={{padding: "2px 4px"}}>
      <div style={{display: "flex", justifyContent: "space-between",
                   fontSize: 13, marginBottom: 6, opacity: 0.85}}>
        <span>{stage}</span><span>{Math.round(pct)}%</span>
      </div>
      <div style={{height: 10, borderRadius: 5, overflow: "hidden",
                   background: "rgba(128,128,128,0.25)"}}>
        <div style={{width: pct + "%", height: "100%", background: color,
                     transition: "width 0.25s"}} />
      </div>
    </div>
  );
}
"""

# controls go to the right of the three 620-px viewers (was a fixed x=1330,
# which sat underneath the third viewer)
CTRL_X = 40 + 3 * (620 + 16) + 24
col = canvas.column(x=CTRL_X, y=40, w=380, gap=14)
section(col, "Data & run")
with col:
    dataset_sel = canvas.toggle(list(DATASETS), name="dataset", label="LOR dataset")
    n_lors = canvas.slider("n_lors", label="LORs used / shown",
                           min=1, max=len(data["lors"]), default=len(data["lors"]))
    run_btn = canvas.button("run", text="▶ Run comparison", color=(0, 170, 90))
    progress = canvas.react(source=PROGRESS_JSX, name="progress",
                            props={"pct": 0, "stage": "idle", "active": False, "failed": False})
    status = canvas.label("status", "idle — tune the config, then Run", h="auto")

section(col, "Shared configuration")
with col:
    iters = canvas.slider("iterations", label="OSEM iterations", min=1, max=10, default=4)
    subsets = canvas.slider("subsets", label="OSEM subsets", min=1, max=8, default=3)
    grid = canvas.toggle(["64", "128", "192", "256"], name="grid_size", default="256",
                         label=f"Grid (voxel = {FOV_MM:g} mm / grid)")
    sens = canvas.toggle(["off", "on"], name="sensitivity", default="off",
                         label="Sensitivity correction (petrecon-computed, shared)")

section(col, "Scatter correction (openSSS)")
with col:
    scatter_sel = canvas.toggle(["off", "openSSS"], name="scatter", default="off",
                                label="Scatter correction — forces sensitivity on")
    scatter_scale = canvas.toggle(["manual", "tail fit"], name="scatter_scale",
                                  default="manual", label="Scatter scale")
    scatter_frac = canvas.slider("scatter_fraction", label="Manual scatter fraction",
                                 min=0.05, max=0.6, step=0.05, default=0.25)

section(col, "Timings & results")
with col:
    timing_table = canvas.table([{"package": "—", "core (s)": "", "wall (s)": "",
                                  "peak": "", "peak voxel": ""}], name="timings", h="auto")

with col:
    canvas.markdown(f"""
#### Notes
- The dataset toggle reloads the LOR set (point source / IEC phantom, ground
  truth or ML-predicted); the slider subsets both packages identically
- **core** = the reconstruction call itself; **wall** = whole subprocess, so
  petrecon pays its numba JIT (~5 s) and OMEGA its CuPy init (~1 s) there
- petrecon runs its normal pipeline; OMEGA runs list-mode OSEM with its Siddon
  projector via the custom-algorithm CuPy path (`omega_runner.py`) — the
  prebuilt OMEGA binaries need the ArrayFire runtime, absent on this machine
- **Sensitivity on**: petrecon streams the full-scanner sensitivity (that cost
  lands in *its* time) and the image is shared with OMEGA for identical
  normalization — off by default so the OSEM cores compare symmetrically
- The difference map is each volume normalized to its own max, subtracted
- **Scatter on** runs the openSSS pipeline: a 2-iteration petrecon pre-pass for
  the activity estimate, openSSS single-scatter simulation (~1 min), then both
  engines reconstruct with the same per-LOR additive scatter. Sensitivity is
  forced on (the correction needs count-scale normalization). Tail-fit scaling
  is experimental at low counts — manual fraction is the safe default
- **OMEGA binned** is the conventional counterpart: events snapped to the
  crystal grid, span-3 sinograms via OMEGA's own saveSinogram, reconstructed
  with the fully-supported turnkey path (`reconstructions_main`). Binned vs
  unbinned on identical events is the thesis's core comparison — expect ~±3 mm
  position error and a few % of events lost to sinogram acceptance. Scatter
  correction applies to the unbinned legs only (sinogram scatter would need
  per-bin injection)
- Run button turns into **■ Cancel** while a comparison is in flight
""", name="notes", h="auto")

# ---- Conditional panel visibility ------------------------------------------------
def show_panels(show, *panels):
    for p in panels:
        (canvas.unhide if show else canvas.hide)(p)


# panels that only apply under an enabling toggle start hidden
if profile_plot is not None:
    show_panels(False, prof_px, prof_py, prof_pz)      # only in "manual" mode
show_panels(False, smooth_fwhm)                        # only when filter is on
show_panels(False, scatter_scale, scatter_frac)        # only when scatter is on


# ---- Wiring -------------------------------------------------------------------
run_lock = threading.Lock()
current_proc = [None]
state_lock = threading.Lock()


@n_lors.on_change(dedicated=True, queue="latest")
def _(v):
    update_lor_layers(int(v))
    if not run_lock.locked():
        status.update(f"{int(v):,} LORs selected — hit Run to compare")


@dataset_sel.on_change(dedicated=True)
def _(v):
    if run_lock.locked():
        status.update("can't switch dataset during a run")
        return
    status.update(f"loading {v} …")
    load_dataset(v)
    n = len(data["lors"])
    n_lors.max = n
    n_lors.update(n)
    update_lor_layers(n)
    status.update(f"{v}: {n:,} LORs loaded — hit Run to compare")


@scatter_sel.on_change
def _(v):
    show_panels(v == "openSSS", scatter_scale, scatter_frac)


@plane_sel.on_change(dedicated=True, queue="latest")
def _(v):
    sync_slice_controls()


@slice_idx.on_change(dedicated=True, queue="latest")
def _(v):
    update_slices()


def _refresh_filtered():
    if vols["pr"] is None and vols["om"] is None:
        return
    if smooth_sel.value == "on":
        # scipy's gaussian_filter releases the GIL: warm all three volumes'
        # cache entries concurrently before any push
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(display_vol, ("pr", "om", "bin")))
    show_volumes(vols["voxel"], fresh=False)   # includes slices via sync
    update_profiles()


@smooth_sel.on_change(dedicated=True)
def _(v):
    show_panels(v == "on", smooth_fwhm)
    _refresh_filtered()


@smooth_fwhm.on_change(dedicated=True, queue="latest")
def _(v):
    if smooth_sel.value == "on":
        _refresh_filtered()


if profile_plot is not None:
    @prof_axis.on_change(dedicated=True, queue="latest")
    def _(v):
        update_profiles()

    @prof_mode.on_change(dedicated=True)
    def _(v):
        show_panels(v == "manual", prof_px, prof_py, prof_pz)
        update_profiles()

    for _s in (prof_px, prof_py, prof_pz):
        _s.on_change(dedicated=True, queue="latest")(lambda v: update_profiles())


class ProgressTap:
    """Parse tqdm bars / 'Iteration k/n' prints from a child's output and
    mirror them onto the progress panel, prefixed with the package name."""

    def __init__(self, prefix):
        self._prefix = prefix
        self._iter = None
        self._sent = None

    def feed(self, s):
        sys.stdout.write(s)
        it = re.search(r"Iteration (\d+)/(\d+)", s)
        if it:
            self._iter = (int(it.group(1)), int(it.group(2)))
        for desc, pct in re.findall(r"([A-Za-z][^|\r\n]*?):\s*(\d+)%\|", s):
            desc = desc.strip()
            if desc == "Subsets" and self._iter:
                desc = "OSEM iteration {}/{}".format(*self._iter)
            if (desc, int(pct)) != self._sent:
                self._sent = (desc, int(pct))
                progress.update(stage=f"{self._prefix}: {desc}", pct=int(pct),
                                active=True, failed=False)
        return len(s)


def run_child(argv, prefix):
    """Run one package's subprocess; returns (rc, wall_s, core_s, tail)."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0)
    with state_lock:
        current_proc[0] = proc
    tap = ProgressTap(prefix)
    tail = deque(maxlen=40)
    t0 = time.time()
    fd = proc.stdout.fileno()
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        text = chunk.decode(errors="replace")
        tail.append(text)
        tap.feed(text)
    rc = proc.wait()
    with state_lock:
        current_proc[0] = None
    wall = time.time() - t0
    m = re.search(r"CORE_SECONDS=([\d.]+)", "".join(tail))
    return rc, wall, float(m.group(1)) if m else None, "".join(tail)


def result_row(name, wall, core, vol):
    peak = np.unravel_index(np.argmax(vol), vol.shape)
    return {"package": name, "core (s)": f"{core:.2f}" if core else "?",
            "wall (s)": f"{wall:.1f}", "peak": f"{vol.max():.4g}",
            "peak voxel": str(tuple(int(p) for p in peak))}


@run_btn.on_click(threaded=True)
def run_or_cancel():
    with state_lock:
        if current_proc[0] is not None:
            status.update("cancelling…")
            current_proc[0].terminate()
            return
    if not run_lock.acquire(blocking=False):
        return
    t0 = time.time()
    try:
        run_btn.text = "■ Cancel comparison"
        run_btn.color = (220, 60, 60)
        n = int(n_lors.value)
        g = int(grid.value)
        voxel = FOV_MM / g
        np.save(SUBSET_NPY, data["lors"][data["perm"][:n]])
        rows = []
        scatter_on = scatter_sel.value == "openSSS"
        sens_on = sens.value == "on" or scatter_on   # scatter needs count-scale sens
        scatter_npy = ""
        sens_path = ""
        status.color = (255, 180, 0)

        def pr_run(tag, n_iter, out_dir=None):
            cfg = {"lors_npy": str(SUBSET_NPY), "out_dir": str(out_dir or PR_OUT), "grid": g,
                   "voxel": voxel, "iterations": n_iter, "subsets": int(subsets.value),
                   "sensitivity": sens_on, "scatter_npy": scatter_npy,
                   "sens_npy": sens_path}
            p = TMP / "danvas_compare_pr.pkl"
            pickle.dump(cfg, open(p, "wb"))
            return run_child([sys.executable, "-u",
                              str(HERE / "petrecon_runner.py"), str(p)], tag)

        def check(rc, tail, what):
            if rc < 0:
                raise InterruptedError("cancelled")
            if rc != 0:
                lines = [l for l in tail.strip().splitlines() if l.strip()]
                raise RuntimeError(f"{what} failed: {lines[-1] if lines else rc}")

        # ---- openSSS scatter pipeline (optional) ----
        if scatter_on:
            status.update(f"activity pre-pass for openSSS ({n:,} LORs, grid {g}³) …")
            progress.update(stage="petrecon: activity pre-pass", pct=0,
                            active=True, failed=False)
            rc, wall, core, tail = pr_run("pre-pass", 2, out_dir=PR_PRE)
            check(rc, tail, "activity pre-pass")
            sens_path = str(PR_PRE / "sensitivity_image.npy")
            rows.append({"package": "pre-pass (2 it.)", "core (s)": f"{core:.1f}" if core else "?",
                         "wall (s)": f"{wall:.1f}", "peak": "—", "peak voxel": "—"})
            timing_table.update(rows)

            status.update("openSSS: estimating scatter …")
            progress.update(stage="openSSS: single scatter simulation", pct=0,
                            active=True, failed=False)
            scatter_npy = str(TMP / "danvas_compare_scatter.npy")
            sss_cfg = {"activity_npy": str(PR_PRE / "reconstruction_image.npy"),
                       "mumap_npy": MU_MAP, "lors_npy": str(SUBSET_NPY),
                       "out_npy": scatter_npy,
                       "scale": "tailfit" if scatter_scale.value == "tail fit" else "manual",
                       "scatter_fraction": float(scatter_frac.value)}
            p = TMP / "danvas_compare_sss.pkl"
            pickle.dump(sss_cfg, open(p, "wb"))
            rc, wall, core, tail = run_child(
                [sys.executable, "-u", str(HERE / "opensss_runner.py"), str(p)], "openSSS")
            check(rc, tail, "openSSS")
            rows.append({"package": "openSSS SSS", "core (s)": f"{wall:.1f}",
                         "wall (s)": f"{wall:.1f}", "peak": "—", "peak voxel": "—"})
            timing_table.update(rows)

        # ---- petrecon ----
        status.update(f"petrecon: {n:,} LORs, grid {g}³ …")
        progress.update(stage="petrecon: starting…", pct=0, active=True, failed=False)
        rc, wall, core, tail = pr_run("petrecon", int(iters.value))
        check(rc, tail, "petrecon")
        sens_path = str(PR_OUT / "sensitivity_image.npy")
        vols["pr"] = np.load(PR_OUT / "reconstruction_image.npy").astype(np.float32)
        vols["voxel"] = voxel
        rows.append(result_row("petrecon", wall, core, vols["pr"]))
        timing_table.update(rows)

        # ---- OMEGA ----
        status.update(f"OMEGA: {n:,} LORs, grid {g}³ …")
        progress.update(stage="OMEGA: starting…", pct=0, active=True, failed=False)
        om_cfg = {"lors_npy": str(SUBSET_NPY), "out_npy": str(OM_NPY), "grid": g,
                  "iterations": int(iters.value), "subsets": int(subsets.value),
                  "sens_npy": sens_path if sens_on else "",
                  "scatter_npy": scatter_npy}
        om_pkl = TMP / "danvas_compare_om.pkl"
        pickle.dump(om_cfg, open(om_pkl, "wb"))
        rc, wall, core, tail = run_child(
            [sys.executable, "-u", str(HERE / "omega_runner.py"), str(om_pkl)], "OMEGA")
        check(rc, tail, "OMEGA")
        vols["om"] = np.load(OM_NPY).astype(np.float32)
        rows.append(result_row("OMEGA", wall, core, vols["om"]))
        timing_table.update(rows)

        # ---- OMEGA binned (conventional baseline; no scatter/sens args —
        # sinogram-mode reconstruction handles its own normalization) ----
        status.update(f"OMEGA binned: {n:,} LORs, grid {g}³ …")
        progress.update(stage="OMEGA binned: sinogram + turnkey OSEM", pct=0,
                        active=True, failed=False)
        bin_cfg = {"lors_npy": str(SUBSET_NPY), "out_npy": str(BIN_NPY), "grid": g,
                   "iterations": int(iters.value), "subsets": int(subsets.value)}
        bin_pkl = TMP / "danvas_compare_bin.pkl"
        pickle.dump(bin_cfg, open(bin_pkl, "wb"))
        rc, wall, core, tail = run_child(
            [sys.executable, "-u", str(HERE / "omega_sinogram_runner.py"),
             str(bin_pkl)], "OMEGA binned")
        check(rc, tail, "OMEGA binned")
        vols["bin"] = np.load(BIN_NPY).astype(np.float32)
        rows.append(result_row("OMEGA binned", wall, core, vols["bin"]))
        timing_table.update(rows)

        # ---- present ----
        show_volumes(voxel)
        update_profiles()
        progress.update(stage="done", pct=100, active=False, failed=False)
        status.color = (0, 200, 90)
        status.update(f"Comparison done in {time.time() - t0:.0f}s total — "
                      f"volumes, slices, diff and profiles updated")
    except InterruptedError:
        progress.update(stage="cancelled", pct=0, active=False, failed=False)
        status.color = (255, 180, 0)
        status.update(f"Cancelled after {time.time() - t0:.0f}s")
    except Exception as exc:
        progress.update(stage="failed", active=False, failed=True)
        status.color = (255, 80, 80)
        status.update(f"Failed after {time.time() - t0:.0f}s: {exc}")
    finally:
        run_btn.text = "▶ Run comparison"
        run_btn.color = (0, 170, 90)
        run_lock.release()


# dispatch-trace panel: live view of which handlers each action fires, where
# they run (inline/threaded/dedicated) and how long they take — the
# observability twin of the threading layout above. deep=False keeps the
# per-call overhead out of the heavy filter/update paths.
canvas.trace(deep=False, below=profile_plot if profile_plot else hm_pr,
             w=800, h=380)

# ---- Initial state -------------------------------------------------------------
# the binned panel hosts a CAD model of its equivalent scanner: the exact
# crystal grid (400 x 40, real pitches and radii from scanner.py) that the
# sinogram binning quantizes onto — toggle it in the panel's Items list
try:
    from scanner_mesh import crystal_grid_glb
    viewer_bin.layer("scanner").update(
        crystal_grid_glb(OUT / "scanner_crystals.glb"),
        mesh_color=(115, 125, 145, 80))
except Exception as _exc:
    print(f"scanner CAD model skipped: {_exc}")

update_lor_layers(len(data["lors"]))
if (PR_OUT / "reconstruction_image.npy").exists() and OM_NPY.exists():
    vols["pr"] = np.load(PR_OUT / "reconstruction_image.npy").astype(np.float32)
    vols["om"] = np.load(OM_NPY).astype(np.float32)
    if BIN_NPY.exists():
        vols["bin"] = np.load(BIN_NPY).astype(np.float32)
    vols["voxel"] = FOV_MM / vols["pr"].shape[0]
    show_volumes(vols["voxel"])
    update_profiles()
    status.update("showing last saved comparison — tune the config, then Run")

canvas.serve(port=8001, tunnel=True, password="petrecon")
