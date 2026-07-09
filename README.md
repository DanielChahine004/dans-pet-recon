# pet-recon

GPU-accelerated PET image reconstruction (OSEM / unfiltered backprojection) from
list-mode LOR data, with sensitivity, attenuation, and single-scatter-simulation
(SSS) corrections. Projections use a CUDA Siddon ray-tracer via Numba.

## Layout

| Path | Purpose |
|---|---|
| `run_recon.py` | Main entry point — runs a full reconstruction from `recon_config.py` |
| `recon_config.py` | All configuration: paths, algorithm, scanner geometry, corrections |
| `helpers/siddon.py` | CUDA Siddon forward/back-projection kernels |
| `helpers/osem.py` | OSEM iteration loop |
| `helpers/backprojection.py` | GPU backprojection (streaming variants) |
| `helpers/sensitivity.py` | Scanner LOR generation + sensitivity image |
| `helpers/attenuation.py` | ACFs and attenuation-weighted sensitivity |
| `helpers/scatter.py` | Single scatter simulation (SSS) estimation |
| `helpers/data_io.py` | LOR / mu-map loading (CSV or NPZ) |
| `helpers/plotting.py`, `helpers/diagnostics.py` | Figures and SSS diagnostics |
| `batch_recon*.py` | Batch drivers over folders of datasets |
| `*_viewing.py`, `compare_*.py`, `calculate_contrast_recovery.py` | Result viewers and analysis (FWHM, contrast recovery, GT-vs-pred) |
| `tests/test_core.py` | Unit tests |

## Setup

```bash
pip install -r requirements.txt   # needs a CUDA-capable GPU for numba.cuda
```

## Usage

Edit `recon_config.py` (input LOR CSV/NPZ, output dir, scanner geometry,
corrections), then:

```bash
python run_recon.py
```

Outputs (`reconstruction_image.npy`, `sensitivity_image.npy`, config snapshot)
are written to `OUTPUT_DIR`.

## Inputs

- **LOR data**: `(N, 6)` CSV or NPZ of endpoint coordinates `x1,y1,z1,x2,y2,z2` (mm)
- **Mu-map** (optional, for attenuation/scatter correction): `.npy` volume,
  e.g. `water_attenuation_map.npy` (not tracked in git — 65 MB)
