# dans-pet-recon

GPU-accelerated PET image reconstruction (OSEM / unfiltered backprojection) from
list-mode LOR data, with sensitivity, attenuation, and single-scatter-simulation
(SSS) corrections. Projections use a CUDA Siddon ray-tracer via Numba.

## Install

```bash
pip install -e .        # needs a CUDA-capable GPU + toolkit for numba.cuda
```

This provides the `petrecon` command.

## Usage

### Run a reconstruction

```bash
petrecon run --lors /data/lors.csv --output /data/recon_out
```

Common options:

```bash
petrecon run \
    --lors /data/lors.csv --output /data/out \
    --mu-map /data/mu_map.npy \          # enables attenuation correction
    --algorithm OSEM --iterations 4 --subsets 3 \
    --grid 256 256 256 --voxel-size 1.0 \
    --rings 40 --detectors-per-ring 400 \
    --inner-radius 235.422 --outer-radius 278.296 --axial-length 296 \
    --scatter \                          # SSS scatter correction
    --filter-oob                         # drop LORs outside the scanner
```

See `petrecon run --help` for the full list. Defaults can also live in a TOML
file (CLI flags override it):

```bash
petrecon config --write recon.toml      # dump a commented template
petrecon run --config recon.toml --lors /data/lors.csv --output /data/out
```

### Batch reconstruction

Reconstruct every position folder under a base directory (skips completed
outputs, so it is resumable). Each job runs in its own subprocess:

```bash
petrecon batch /data/point_sources --kinds gt pred --config recon.toml
```

Expects `BASE_DIR/<pos>/gt_processed_<pos>*.csv` and/or
`prediction_processed_<pos>*.csv`; results go to `<pos>/gt` and `<pos>/pred`.

### View results

```bash
petrecon view /data/recon_out                 # interactive plotly/matplotlib views
petrecon view /data/recon_out --smooth-sigma 2
```

## Inputs and outputs

**Inputs**
- LOR data: `(N, 6)` CSV (header row) or NPZ of endpoint coordinates
  `x1,y1,z1,x2,y2,z2` in mm
- Mu-map (optional): `.npy` volume matching the image grid, in 1/mm —
  enables attenuation correction (and is required for scatter correction)

**Outputs** (written to `--output`, which is cleared first)
- `reconstruction_image.npy`, `sensitivity_image.npy`, `backprojection_image.npy`
- `recon_config.pkl` — snapshot of the run configuration
- scatter sinograms + per-LOR estimates when `--scatter` is on

## Layout

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
original research workflow — edit their constants before use.

## Python API

```python
from petrecon import ReconConfig
from petrecon.recon import run_reconstruction

cfg = ReconConfig(path_to_lors="/data/lors.csv", output_dir="/data/out",
                  osem_iterations=4, image_grid_size=(256, 256, 256))
recon = run_reconstruction(cfg)   # returns the reconstructed volume
```
