"""Interactive petrecon dashboard on a danvas canvas.

Every ReconConfig attribute is a panel; a LOR-count slider controls both how
many LORs are shown on the Model3D panel and how many feed the reconstruction;
the Run button reconstructs on the GPU and swaps the new volume into the
viewer automatically.

  python examples/danvas_view.py     ->  http://127.0.0.1:8000
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
from dataclasses import asdict, fields as dc_fields
from pathlib import Path

import numpy as np
import danvas

from petrecon import ReconConfig
from petrecon.helpers.data_io import load_lors

# ---- Inputs ------------------------------------------------------------------
LOR_CSV = ("/home/h/Code/Opengate/ragy_files/Point_Sources_Processed/"
           "0_0_0/gt_processed_0_0_0.csv")
OUT_DIR = Path(__file__).resolve().parent.parent / "example_output"
SUBSET_NPY = Path(tempfile.gettempdir()) / "danvas_lor_subset.npy"  # OUT_DIR is wiped per run
N_LINE_LORS = 400   # at most this many LORs drawn as lines
MM = 1e-3           # Model3D world units are meters; petrecon coordinates are mm

all_lors = load_lors(LOR_CSV).astype(np.float32)          # same filtering as petrecon
perm = np.random.default_rng(0).permutation(len(all_lors))  # fixed shuffle: any prefix is uniform

# ---- Canvas + Model3D ----------------------------------------------------------
canvas = danvas.Canvas()
viewer = canvas.model3d("pet", x=40, y=40, w=900, h=760)


def update_lor_layers(n):
    subset = all_lors[perm[:n]]
    endpoints = np.concatenate([subset[:, :3], subset[:, 3:]], axis=0) * MM
    viewer.layer("endpoints").points(endpoints, color_by=endpoints[:, 2])
    lines = subset[: min(N_LINE_LORS, n)].reshape(-1, 2, 3) * MM
    viewer.layer("lors").lines(lines, color=(120, 160, 220, 60))


current_vol = {"vol": None, "voxel": 1.0}   # full-res volume, shared with the slice viewer


def show_volume(recon, voxel_mm):
    # Native resolution: danvas quantizes the volume to uint8 on the wire
    # (256³ ≈ 17 MB, a one-shot push per recon), so no downsampling — the
    # Items row then always reads the true reconstruction grid.
    recon = recon.astype(np.float32)
    current_vol["vol"] = recon
    current_vol["voxel"] = voxel_mm
    spacing = (voxel_mm * MM,) * 3
    origin = tuple(-n * voxel_mm * MM / 2 for n in recon.shape)  # grid is scanner-centered
    viewer.layer("recon").volume(recon, spacing=spacing, origin=origin, cmap="hot")
    sync_slice_controls()


# ---- Slice viewer (full-resolution, beneath the 3D panel) ---------------------------
PLANES = {"xy": 2, "xz": 1, "yz": 0}   # plane -> axis index to slice along

slice_hm = canvas.heatmap("slice_view", below=viewer, w=520, h=460)
plane_sel = canvas.toggle(list(PLANES), name="slice_plane", label="Slice plane",
                          right_of=slice_hm)
slice_idx = canvas.slider("slice_idx", label="Slice index", min=0, max=255, default=128,
                          below=plane_sel, w=380)
slice_pos = canvas.label("slice_pos", "no volume yet — Run a reconstruction",
                         below=slice_idx, w=380, h="auto")


def update_slice():
    vol = current_vol["vol"]
    if vol is None:
        return
    axis = PLANES[plane_sel.value]
    k = min(int(slice_idx.value), vol.shape[axis] - 1)
    sl = np.take(vol, k, axis=axis)
    a, b = (i for i in range(3) if i != axis)       # in-plane axes: a → right, b → up
    vox = current_vol["voxel"]
    half_a, half_b = (vol.shape[i] * vox / 2 for i in (a, b))
    # rows flipped so +b runs up; heatmap extent y labels top row first
    slice_hm.update(sl.T[::-1], vmin=0.0, vmax=float(vol.max()) or 1.0,
                    cmap="hot", extent=(-half_a, half_a, half_b, -half_b))
    mm = (k + 0.5 - vol.shape[axis] / 2) * current_vol["voxel"]
    slice_pos.update(f"{plane_sel.value} slice {k}/{vol.shape[axis] - 1} — "
                     f"{'xyz'[axis]} = {mm:+.1f} mm (full resolution)")


def sync_slice_controls():
    vol = current_vol["vol"]
    if vol is None:
        return
    dim = vol.shape[PLANES[plane_sel.value]]
    slice_idx.max = dim - 1
    if int(slice_idx.value) >= dim:
        slice_idx.update(dim // 2)
    update_slice()


# ---- Config panels ---------------------------------------------------------------
def section(col, text):
    slug = text.lower().replace(" ", "_").replace("&", "and")
    col.add(canvas.markdown(f"#### {text}", name=f"sec_{slug}", h="auto"))


ALGOS = {"OSEM": "OSEM", "Backprojection": "Unfiltered_backprojection", "None": "None"}
ON_OFF = ["on", "off"]
CFG_DEFAULTS = {f.name: f.default for f in dc_fields(ReconConfig)}

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

col_run = canvas.column(x=980, y=40, w=360, gap=14)
section(col_run, "Data & run")
with col_run:
    n_lors = canvas.slider("n_lors", label="LORs used / shown",
                           min=1, max=len(all_lors), default=len(all_lors))
    run_btn = canvas.button("run", text="▶ Run reconstruction", color=(0, 170, 90))
    reset_btn = canvas.button("reset", text="↺ Restore defaults")
    progress = canvas.react(source=PROGRESS_JSX, name="progress",
                            props={"pct": 0, "stage": "idle", "active": False, "failed": False})
    status = canvas.label("status", "idle — tune the config, then Run", h="auto")

section(col_run, "Algorithm")
with col_run:
    algo = canvas.toggle(list(ALGOS), name="algorithm", label="Algorithm")
    iters = canvas.slider("iterations", label="OSEM iterations", min=1, max=10, default=4)
    subsets = canvas.slider("subsets", label="OSEM subsets", min=1, max=8, default=3)

section(col_run, "Image grid")
with col_run:
    grid = canvas.toggle(["64", "128", "192", "256"], name="grid_size", default="256",
                         label="Grid size (per axis)")
    voxel = canvas.slider("voxel_size", label="Voxel size (mm)",
                          min=0.5, max=4.0, step=0.25, default=1.0)

col_geo = canvas.column(x=1380, y=40, w=360, gap=14)
section(col_geo, "Scanner geometry")
with col_geo:
    rings = canvas.slider("rings", label="Axial rings", min=8, max=64, default=40)
    dets = canvas.slider("detectors", label="Detectors per ring", min=100, max=800, default=400)
    axial = canvas.slider("axial_length", label="Axial length (mm)",
                          min=100.0, max=400.0, step=0.001, default=296.0)
    r_in = canvas.slider("inner_radius", label="Inner radius (mm)",
                         min=150.0, max=300.0, step=0.001, default=235.422)
    r_out = canvas.slider("outer_radius", label="Outer radius (mm)",
                          min=150.0, max=350.0, step=0.001, default=278.296)

section(col_geo, "Corrections")
with col_geo:
    sens = canvas.toggle(ON_OFF, name="sensitivity", label="Sensitivity correction")
    smooth = canvas.toggle(ON_OFF, name="smooth_sensitivity", label="Smooth sensitivity")
    sigma = canvas.slider("gaussian_sigma", label="Gaussian sigma",
                          min=0.5, max=10.0, step=0.5, default=5.0)
    g_radius = canvas.slider("gaussian_radius", label="Gaussian radius", min=1, max=10, default=5)
    oob = canvas.toggle(ON_OFF, name="filter_oob", default="off", label="Filter OOB LORs")
    scatter = canvas.toggle(ON_OFF, name="scatter", default="off", label="Scatter correction (SSS)")
    mu_map = canvas.text_field("mu_map", label="Mu-map .npy (enables attenuation)",
                               placeholder="/path/to/mu_map.npy — blank = off")
    chunk = canvas.slider("sens_chunk", label="Sensitivity chunk size (M LORs)",
                          min=1, max=10, default=5)
    batch = canvas.slider("sens_batch", label="Sensitivity batch size (M LORs)",
                          min=1, max=10, default=5)

col_sss = canvas.column(x=1780, y=40, w=360, gap=14)
section(col_sss, "Single scatter simulation")
with col_sss:
    sss_coarse = canvas.slider("sss_coarse", label="Coarse factor", min=2, max=16, default=8)
    sss_ang = canvas.slider("sss_angular", label="Angular detectors", min=8, max=64, default=32)
    sss_ax = canvas.slider("sss_axial", label="Axial detectors", min=5, max=30, default=15)
    sss_start = canvas.slider("sss_start", label="Start iteration", min=0, max=8, default=2)
    sss_int = canvas.slider("sss_interval", label="Update interval", min=1, max=4, default=2)
    sss_scale = canvas.slider("sss_scale", label="Scale factor",
                              min=0.1, max=3.0, step=0.1, default=1.0)

with col_sss:
    canvas.markdown(f"""
#### Notes
- **{len(all_lors):,} LORs** loaded from `{Path(LOR_CSV).name}`
- The LOR slider re-picks a uniform random subset live and caps what Run feeds OSEM
- While a run is in flight the Run button becomes **■ Cancel** — it kills the
  reconstruction subprocess immediately (the output folder may be left incomplete)
- Output lands in `{OUT_DIR.name}/`; the folder is wiped each run
- Sensitivity correction streams every scanner LOR — the dominant cost
  (~2 min at the defaults). Fewer rings/detectors or a smaller grid makes runs much faster
- Scatter correction needs a mu-map path (there's `water_attenuation_map.npy` in the repo root)
""", name="notes", h="auto")

# ---- Wiring -----------------------------------------------------------------------
run_lock = threading.Lock()


class ProgressTap:
    """Pass-through stdout/stderr wrapper that mirrors tqdm bars and
    "Iteration k/n" prints onto the canvas progress panel."""

    def __init__(self, real, state):
        self._real = real
        self._state = state
        self._sent = None

    def write(self, s):
        self._real.write(s)
        it = re.search(r"Iteration (\d+)/(\d+)", s)
        if it:
            self._state["iter"] = (int(it.group(1)), int(it.group(2)))
        for desc, pct in re.findall(r"([A-Za-z][^|\r\n]*?):\s*(\d+)%\|", s):
            desc = desc.strip()
            if desc == "Subsets" and self._state.get("iter"):
                desc = "OSEM iteration {}/{}".format(*self._state["iter"])
            if (desc, int(pct)) != self._sent:
                self._sent = (desc, int(pct))
                progress.update(stage=desc, pct=int(pct), active=True, failed=False)
        return len(s)

    def flush(self):
        self._real.flush()


@reset_btn.on_click
def restore_defaults():
    if run_lock.locked():
        status.update("can't restore defaults while a reconstruction is running")
        return
    d = CFG_DEFAULTS
    algo.update("OSEM")
    iters.update(d["osem_iterations"]); subsets.update(d["osem_subsets"])
    grid.update(str(d["image_grid_size"][0])); voxel.update(d["voxel_size_in_mm"])
    rings.update(d["n_axial_rings"]); dets.update(d["n_detectors_per_ring"])
    axial.update(d["axial_length"]); r_in.update(d["inner_radius"]); r_out.update(d["outer_radius"])
    sens.update("on" if d["sensitivity_correction"] else "off")
    smooth.update("on" if d["smooth_sensitivity"] else "off")
    sigma.update(d["gaussian_sigma"]); g_radius.update(d["gaussian_radius"])
    oob.update("on" if d["filter_oob_lors"] else "off")
    scatter.update("on" if d["scatter_correction"] else "off")
    mu_map.update("")
    chunk.update(d["sensitivity_chunk_size"] // 1_000_000)
    batch.update(d["sensitivity_batch_size"] // 1_000_000)
    sss_coarse.update(d["sss_coarse_factor"])
    sss_ang.update(d["sss_n_angular_detectors"]); sss_ax.update(d["sss_n_axial_detectors"])
    sss_start.update(d["sss_start_iteration"]); sss_int.update(d["sss_update_interval"])
    sss_scale.update(d["sss_scale_factor"])
    n_lors.update(len(all_lors)); update_lor_layers(len(all_lors))
    status.color = None
    status.update("defaults restored — all panels back to ReconConfig defaults")


@n_lors.on_change(dedicated=True, queue="latest")
def _(v):
    update_lor_layers(int(v))
    if not run_lock.locked():
        status.update(f"{int(v):,} LORs selected — hit Run to reconstruct")


@plane_sel.on_change
def _(v):
    sync_slice_controls()


@slice_idx.on_change(dedicated=True, queue="latest")
def _(v):
    update_slice()


def build_config():
    return ReconConfig(
        path_to_lors=str(SUBSET_NPY),
        output_dir=str(OUT_DIR),
        mu_map_path=(mu_map.value or "").strip() or None,
        recon_algorithm=ALGOS[algo.value],
        osem_iterations=int(iters.value),
        osem_subsets=int(subsets.value),
        n_axial_rings=int(rings.value),
        n_detectors_per_ring=int(dets.value),
        axial_length=float(axial.value),
        inner_radius=float(r_in.value),
        outer_radius=float(r_out.value),
        image_grid_size=(int(grid.value),) * 3,
        voxel_size_in_mm=float(voxel.value),
        sensitivity_correction=sens.value == "on",
        sensitivity_chunk_size=int(chunk.value) * 1_000_000,
        sensitivity_batch_size=int(batch.value) * 1_000_000,
        smooth_sensitivity=smooth.value == "on",
        gaussian_sigma=float(sigma.value),
        gaussian_radius=int(g_radius.value),
        scatter_correction=scatter.value == "on",
        sss_coarse_factor=int(sss_coarse.value),
        sss_n_angular_detectors=int(sss_ang.value),
        sss_n_axial_detectors=int(sss_ax.value),
        sss_start_iteration=int(sss_start.value),
        sss_update_interval=int(sss_int.value),
        sss_scale_factor=float(sss_scale.value),
        filter_oob_lors=oob.value == "on",
    )


# The recon runs in a subprocess (not a thread) so Cancel can kill it cleanly,
# GPU context included. This snippet is the child's entry point.
RUNNER = ("import pickle, sys; "
          "from petrecon.config import ReconConfig; "
          "from petrecon.recon import run_reconstruction; "
          "run_reconstruction(ReconConfig(**pickle.load(open(sys.argv[1], 'rb'))))")

current_proc = [None]
state_lock = threading.Lock()


@run_btn.on_click(threaded=True)
def run_or_cancel():
    with state_lock:
        if current_proc[0] is not None:      # a run is in flight -> this click cancels
            status.update("cancelling…")
            current_proc[0].terminate()
            return
    if not run_lock.acquire(blocking=False):
        return  # run is starting up in another thread; try again in a moment
    t0 = time.time()
    try:
        n = int(n_lors.value)
        np.save(SUBSET_NPY, all_lors[perm[:n]])
        cfg = build_config()
        cfg_pkl = Path(tempfile.gettempdir()) / "danvas_recon_cfg.pkl"
        with open(cfg_pkl, "wb") as f:
            pickle.dump(asdict(cfg), f)

        proc = subprocess.Popen([sys.executable, "-u", "-c", RUNNER, str(cfg_pkl)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        with state_lock:
            current_proc[0] = proc
        run_btn.text = "■ Cancel reconstruction"
        run_btn.color = (220, 60, 60)
        status.color = (255, 180, 0)
        status.update(f"Running {cfg.recon_algorithm} on {n:,} LORs, "
                      f"grid {cfg.image_grid_size[0]}³ …")
        progress.update(stage="starting…", pct=0, active=True, failed=False)

        tap = ProgressTap(sys.stdout, {"iter": None})
        tail = deque(maxlen=40)              # keep recent output for error reporting
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            tail.append(text)
            tap.write(text)
        rc = proc.wait()

        if rc == 0:
            recon = np.load(OUT_DIR / "reconstruction_image.npy").astype(np.float32)
            show_volume(recon, cfg.voxel_size_in_mm)
            progress.update(stage="done", pct=100, active=False, failed=False)
            status.color = (0, 200, 90)
            status.update(f"Done in {time.time() - t0:.0f}s — volume updated "
                          f"(max {recon.max():.3g}). Saved to {OUT_DIR.name}/")
        elif rc < 0:                          # killed by a signal -> user cancel
            progress.update(stage="cancelled", pct=0, active=False, failed=False)
            status.color = (255, 180, 0)
            status.update(f"Cancelled after {time.time() - t0:.0f}s — "
                          f"{OUT_DIR.name}/ may be incomplete")
        else:
            lines = [l for l in "".join(tail).strip().splitlines() if l.strip()]
            progress.update(stage="failed", active=False, failed=True)
            status.color = (255, 80, 80)
            status.update(f"Failed after {time.time() - t0:.0f}s: "
                          f"{lines[-1] if lines else f'exit code {rc}'}")
    except Exception as exc:
        progress.update(stage="failed", active=False, failed=True)
        status.color = (255, 80, 80)
        status.update(f"Failed after {time.time() - t0:.0f}s: {exc}")
    finally:
        with state_lock:
            current_proc[0] = None
        run_btn.text = "▶ Run reconstruction"
        run_btn.color = (0, 170, 90)
        run_lock.release()


# ---- Initial state -----------------------------------------------------------------
update_lor_layers(len(all_lors))
last = OUT_DIR / "reconstruction_image.npy"
if last.exists():
    show_volume(np.load(last).astype(np.float32), voxel_mm=1.0)
    status.update("showing last saved reconstruction — tune the config, then Run")

canvas.serve(port=8000, tunnel=True, password="petrecon")
