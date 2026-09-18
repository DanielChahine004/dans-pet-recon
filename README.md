# dans-pet-recon

GPU-accelerated PET image reconstruction from list-mode LOR (line-of-response)
data. Supports OSEM and unfiltered backprojection, with sensitivity,
attenuation, and single-scatter-simulation (SSS) corrections. Forward and back
projections use a CUDA Siddon ray-tracer implemented with Numba.

---

## Requirements

- Python ≥ 3.10
- An NVIDIA GPU with the CUDA toolkit (the projection kernels run via
  `numba.cuda` — there is no CPU fallback)

## Installation

```bash
git clone https://github.com/DanielChahine004/dans-pet-recon.git
cd dans-pet-recon
pip install -e .
```

This installs the Python package `petrecon` and puts a `petrecon` command on
your PATH. Check it worked:

```bash
petrecon --version
python -m unittest discover -s tests    # optional: run the test suite
```

---

## Quickstart

```bash
# 1. Reconstruct
petrecon run --lors /data/lors.csv --output /data/recon_out

# 2. Inspect the results interactively
petrecon view /data/recon_out
```

That runs 4-iteration / 3-subset OSEM with sensitivity correction on the
default scanner geometry, and writes `reconstruction_image.npy` (and friends)
to `/data/recon_out`.

## Launchers (`pixi run …`)

The project runs from the `base` pixi env at `~/pixi-envs/base`; a `pixi()`
wrapper in `~/.bashrc` points `pixi run` at that manifest from any directory
without its own, so these work from the repo root:

| Task | What it launches |
|------|------------------|
| `pixi run scanner` | Thesis scanner CAD model + 25 point sources on a danvas Model3D panel (`:8002`) |
| `pixi run view` | Interactive petrecon dashboard: config panels, LOR viewer, run button (`:8000`) |
| `pixi run compare` | petrecon vs OMEGA vs OMEGA-binned comparison (`:8001`) |
| `pixi run dashboards` | All three canvases at once; Ctrl-C stops them all |
| `pixi run recon …` | The `petrecon` CLI, e.g. `pixi run recon run --config examples/recon_config.toml` |
| `pixi run batch-binned` | `scripts/batch_recon_binned.py` |
| `pixi run batch-pdf` | `scripts/batch_recon_pdf.py` |
| `pixi run test` | Unit tests (`unittest`; engine tests need a GPU) |

Before redoing the thesis §5.7 discretisation study, read
[examples/DISCRETISATION_STUDY_RERUN.md](examples/DISCRETISATION_STUDY_RERUN.md).

`pixi task list` prints the same table. Tasks are defined in
`~/pixi-envs/base/pixi.toml`, not in this repo, because the env (torch,
CUDA, OMEGA) lives there.

## Input data

**LOR file** (`--lors`) — one row per coincidence, endpoint coordinates in mm:

- **CSV** with a header row and 6 columns: `x1,y1,z1,x2,y2,z2`
- **NPZ** containing either a single `(N, 6)` array, or two `(N, 3)` arrays
  named `xyz1` and `xyz2`

Rows with a `(0,0,0)` endpoint or identical endpoints are dropped automatically.

**Mu-map** (`--mu-map`, optional) — a `.npy` volume of linear attenuation
coefficients in 1/mm whose shape matches the image grid (`--grid`). Passing it
switches on attenuation correction; it is also required for scatter correction.

---

## The four subcommands

### `petrecon run` — reconstruct one dataset

```bash
petrecon run \
    --lors /data/lors.csv \
    --output /data/out \
    --mu-map /data/mu_map.npy \
    --algorithm OSEM --iterations 4 --subsets 3 \
    --grid 256 256 256 --voxel-size 1.0 \
    --rings 40 --detectors-per-ring 400 \
    --inner-radius 235.422 --outer-radius 278.296 --axial-length 296 \
    --scatter \
    --filter-oob
```

| Option | Default | Meaning |
|---|---|---|
| `--lors PATH` | *(required)* | LOR CSV/NPZ file |
| `--output DIR` | *(required)* | Output directory — **cleared before the run** |
| `--mu-map PATH` | none | Mu-map `.npy`; enables attenuation correction |
| `--config TOML` | none | Config file; any CLI flag overrides it |
| `--algorithm` | `OSEM` | `OSEM`, `Unfiltered_backprojection`, or `None` |
| `--iterations N` | 4 | OSEM iterations |
| `--subsets N` | 3 | OSEM subsets |
| `--grid NX NY NZ` | 256 256 256 | Image grid size (voxels) |
| `--voxel-size MM` | 1.0 | Isotropic voxel size |
| `--rings N` | 40 | Axial detector rings |
| `--detectors-per-ring N` | 400 | Detectors per ring |
| `--inner-radius MM` | 235.422 | Scanner inner radius |
| `--outer-radius MM` | 278.296 | Scanner outer radius |
| `--axial-length MM` | 296 | Scanner axial length |
| `--sensitivity / --no-sensitivity` | on | Sensitivity correction |
| `--smooth-sensitivity / --no-…` | on | Gaussian-smooth the sensitivity image |
| `--attenuation / --no-attenuation` | auto | On iff `--mu-map` is given |
| `--scatter / --no-scatter` | off | SSS scatter correction (needs `--mu-map`) |
| `--filter-oob / --no-filter-oob` | off | Drop LORs with endpoints outside the scanner |

SSS tuning parameters (coarse factor, detector binning, start iteration,
update interval, scale factor) and the sensitivity streaming chunk/batch sizes
are settable via the config file — see `petrecon config`.

**Outputs** written to `--output`:

| File | Content |
|---|---|
| `reconstruction_image.npy` | Final reconstructed volume |
| `sensitivity_image.npy` | Sensitivity image used for normalisation |
| `backprojection_image.npy` | Plain backprojection of the LORs |
| `recon_config.pkl` | Snapshot of every setting used for the run |
| `scatter_*.npy`, `sino_params.npz` | Scatter sinograms/estimates (only with `--scatter`) |

### `petrecon config` — config file template

```bash
petrecon config                      # print a commented TOML template
petrecon config --write recon.toml   # or write it to a file
```

Edit the file, then pass it with `--config`. Precedence is
**defaults < TOML < CLI flags**, so a config file can hold your scanner
geometry while `--lors/--output` vary per run:

```bash
petrecon run --config recon.toml --lors /data/scan1.csv --output /data/scan1_out
```

A ready-made copy of the template lives at
[examples/recon_config.toml](examples/recon_config.toml).

### `petrecon batch` — reconstruct a folder of datasets

Iterates over position subfolders and reconstructs each `gt`/`pred` dataset it
finds. Skips anything that already has a `reconstruction_image.npy`, so it is
resumable. Every job runs in its own subprocess, and all `run` options are
forwarded to each job:

```bash
petrecon batch /data/point_sources --kinds gt pred --config recon.toml
```

Expected folder convention:

```
/data/point_sources/
└── <position>/
    ├── gt_processed_<position>*.csv          # input  (kind: gt)
    ├── prediction_processed_<position>*.csv  # input  (kind: pred)
    ├── gt/                                   # output (created)
    └── pred/                                 # output (created)
```

### `petrecon view` — inspect a finished reconstruction

```bash
petrecon view /data/out                    # full interactive tour
petrecon view /data/out --smooth-sigma 2   # heavier display smoothing
petrecon view /data/out --no-smooth
petrecon view /data/out --lors /new/path/lors.csv   # if the LOR file moved
```

Opens interactive plotly/matplotlib views of the scanner geometry, sinograms
(with scatter comparison when available), backprojection, sensitivity, mu-map,
and reconstruction, plus a line profile through the hottest voxel with a
Gaussian fit (FWHM/FWTM printed to the terminal).

> Tip: `napari /data/out/reconstruction_image.npy` is a nice dedicated volume
> viewer if you have napari installed.

---

## Python API

Everything the CLI does is callable from Python:

```python
from petrecon import ReconConfig
from petrecon.recon import run_reconstruction
from petrecon.viewing import view_reconstruction

cfg = ReconConfig(
    path_to_lors="/data/lors.csv",
    output_dir="/data/out",
    mu_map_path="/data/mu_map.npy",   # auto-enables attenuation correction
    osem_iterations=4,
    image_grid_size=(256, 256, 256),
)
recon = run_reconstruction(cfg)       # returns the reconstructed volume (np.ndarray)

view_reconstruction("/data/out")      # same plots as `petrecon view`
```

`ReconConfig.from_toml("recon.toml", overrides={...})` loads a config file with
optional programmatic overrides; `cfg.validate()` raises `ValueError` with a
list of problems if the config isn't runnable.

## Repository layout

| Path | Purpose |
|---|---|
| `src/petrecon/cli.py` | `petrecon` command (run / batch / view / config) |
| `src/petrecon/config.py` | `ReconConfig` dataclass — all settings + TOML loading |
| `src/petrecon/recon.py` | Reconstruction pipeline |
| `src/petrecon/batch.py` | Batch driver |
| `src/petrecon/viewing.py` | Result viewer |
| `src/petrecon/helpers/siddon.py` | CUDA Siddon forward/back-projection kernels |
| `src/petrecon/helpers/osem.py` | OSEM iteration loop |
| `src/petrecon/helpers/sensitivity.py` | Scanner LOR generation + sensitivity image |
| `src/petrecon/helpers/attenuation.py` | ACFs and attenuation-weighted sensitivity |
| `src/petrecon/helpers/scatter.py` | Single scatter simulation (SSS) |
| `scripts/` | Research/analysis scripts (FWHM, contrast recovery, GT-vs-pred, binned studies, shiny viewer) |
| `examples/recon_config.toml` | Config template (same as `petrecon config`) |
| `tests/test_core.py` | Unit tests (`python -m unittest discover -s tests`) |

The analysis scripts in `scripts/` still carry hardcoded dataset paths from the
original research workflow — edit the constants at the top of each before use.

## Troubleshooting

- **`CUDA available: False`** printed at the start of a run — Numba can't see
  your GPU. Check `nvidia-smi` works and that the CUDA toolkit matches your
  driver; on WSL2 you need a Windows NVIDIA driver with WSL support.
- **Out-of-memory during sensitivity computation** — lower
  `sensitivity_batch_size` (GPU VRAM) and/or `sensitivity_chunk_size` (CPU RAM)
  in your config TOML. At the default 5M LORs per batch, a 256³ grid uses
  ~184 MB of VRAM per launch.
- **`mu_map shape ... does not match IMAGE_GRID_SIZE`** — the mu-map volume
  must have exactly the shape passed as `--grid`.
- **Output folder vanished** — `--output` is deleted and recreated at the start
  of every run by design; point each run at its own directory.
