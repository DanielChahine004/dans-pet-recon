"""petrecon command-line interface.

Subcommands:
    petrecon run    — run a reconstruction
    petrecon batch  — reconstruct every dataset under a folder
    petrecon view   — display the outputs of a reconstruction
    petrecon config — print a TOML config template

Config precedence for `run`: dataclass defaults < --config TOML < CLI flags.
"""
import argparse
import sys
from pathlib import Path

from petrecon import __version__
from petrecon.config import ReconConfig

CONFIG_TEMPLATE = """\
# petrecon configuration — pass with: petrecon run --config this_file.toml
# Any CLI flag overrides the value set here. All keys optional.

# ---- Paths (usually given on the CLI instead: --lors / --output / --mu-map) ----
# path_to_lors = "/data/lors.csv"     # (N,6) CSV or NPZ: x1,y1,z1,x2,y2,z2 in mm
# output_dir = "/data/recon_out"
# mu_map_path = "/data/mu_map.npy"    # setting this enables attenuation correction

# ---- Algorithm ----
recon_algorithm = "OSEM"              # "OSEM" | "Unfiltered_backprojection" | "None"
osem_iterations = 4
osem_subsets = 3

# ---- Scanner geometry (mm) ----
n_axial_rings = 40
n_detectors_per_ring = 400
axial_length = 296.0
inner_radius = 235.422
outer_radius = 278.296

# ---- Image grid ----
image_grid_size = [256, 256, 256]
voxel_size_in_mm = 1.0
# fov_radius = 223.65                 # default: 0.95 * inner_radius
# fov_axial_half_length = 128.0       # default: grid_z * voxel_size // 2

# ---- Corrections ----
sensitivity_correction = true
sensitivity_chunk_size = 5000000      # LORs per CPU chunk during scanner LOR streaming
sensitivity_batch_size = 5000000      # LORs per GPU kernel launch — tune to VRAM
smooth_sensitivity = true
gaussian_sigma = 5.0
gaussian_radius = 5
# attenuation_correction = true       # default: auto (on iff mu_map_path is set)
scatter_correction = false

# ---- Single scatter simulation (SSS) ----
sss_coarse_factor = 8
sss_n_angular_detectors = 32
sss_n_axial_detectors = 15
sss_start_iteration = 2
sss_update_interval = 2
sss_scale_factor = 1.0

# ---- LOR filtering ----
filter_oob_lors = false               # drop LORs with endpoints outside the scanner
"""


def _add_run_args(p: argparse.ArgumentParser):
    p.add_argument("--config", metavar="TOML", help="TOML config file (CLI flags override it)")
    # Paths
    p.add_argument("--lors", dest="path_to_lors", metavar="PATH",
                   help="LOR file: (N,6) CSV or NPZ of endpoints x1,y1,z1,x2,y2,z2 in mm")
    p.add_argument("--output", dest="output_dir", metavar="DIR",
                   help="Output directory (cleared before the run)")
    p.add_argument("--mu-map", dest="mu_map_path", metavar="PATH",
                   help="Mu-map .npy volume; enables attenuation correction")
    # Algorithm
    p.add_argument("--algorithm", dest="recon_algorithm",
                   choices=["OSEM", "Unfiltered_backprojection", "None"],
                   help="Reconstruction algorithm (default: OSEM)")
    p.add_argument("--iterations", dest="osem_iterations", type=int, help="OSEM iterations")
    p.add_argument("--subsets", dest="osem_subsets", type=int, help="OSEM subsets")
    # Geometry
    p.add_argument("--rings", dest="n_axial_rings", type=int, help="Number of axial detector rings")
    p.add_argument("--detectors-per-ring", dest="n_detectors_per_ring", type=int)
    p.add_argument("--axial-length", dest="axial_length", type=float, help="Scanner axial length (mm)")
    p.add_argument("--inner-radius", dest="inner_radius", type=float, help="Scanner inner radius (mm)")
    p.add_argument("--outer-radius", dest="outer_radius", type=float, help="Scanner outer radius (mm)")
    # Grid
    p.add_argument("--grid", dest="image_grid_size", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
                   help="Image grid size, e.g. --grid 256 256 256")
    p.add_argument("--voxel-size", dest="voxel_size_in_mm", type=float, help="Voxel size (mm)")
    # Corrections (BooleanOptionalAction gives --x / --no-x pairs)
    p.add_argument("--sensitivity", dest="sensitivity_correction",
                   action=argparse.BooleanOptionalAction, help="Sensitivity correction")
    p.add_argument("--smooth-sensitivity", dest="smooth_sensitivity",
                   action=argparse.BooleanOptionalAction, help="Gaussian-smooth the sensitivity image")
    p.add_argument("--attenuation", dest="attenuation_correction",
                   action=argparse.BooleanOptionalAction,
                   help="Attenuation correction (default: on iff --mu-map is given)")
    p.add_argument("--scatter", dest="scatter_correction",
                   action=argparse.BooleanOptionalAction, help="SSS scatter correction")
    p.add_argument("--filter-oob", dest="filter_oob_lors",
                   action=argparse.BooleanOptionalAction,
                   help="Drop LORs with endpoints outside the physical scanner")


def _collect_overrides(args: argparse.Namespace) -> dict:
    """CLI flags that were explicitly set, keyed by ReconConfig field name."""
    names = ReconConfig.field_names()
    overrides = {}
    for name, value in vars(args).items():
        if name in names and value is not None:
            overrides[name] = tuple(value) if name == "image_grid_size" else value
    return overrides


def _build_config(args: argparse.Namespace) -> ReconConfig:
    overrides = _collect_overrides(args)
    if args.config:
        return ReconConfig.from_toml(args.config, overrides)
    return ReconConfig(**overrides)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="petrecon",
        description="GPU-accelerated PET image reconstruction (OSEM / backprojection).",
    )
    parser.add_argument("--version", action="version", version=f"petrecon {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    p_run = sub.add_parser("run", help="Run a reconstruction",
                           description="Run a reconstruction. Requires --lors and --output "
                                       "(directly or via --config).")
    _add_run_args(p_run)

    # ---- batch ----
    p_batch = sub.add_parser(
        "batch", help="Reconstruct every dataset under a folder",
        description="Iterate over position subfolders of BASE_DIR, reconstructing "
                    "gt/pred LOR CSVs into gt/ and pred/ output subfolders. "
                    "Positions with existing outputs are skipped (resumable). "
                    "All run options (--config, --algorithm, geometry flags, ...) are "
                    "forwarded to each job.")
    p_batch.add_argument("base_dir", help="Folder containing one subfolder per position")
    p_batch.add_argument("--kinds", nargs="+", choices=["gt", "pred"], default=["gt", "pred"],
                         help="Dataset kinds to reconstruct (default: gt pred)")
    _add_run_args(p_batch)

    # ---- view ----
    p_view = sub.add_parser("view", help="Display the outputs of a reconstruction",
                            description="Open interactive plots for a completed reconstruction.")
    p_view.add_argument("output_dir", help="Reconstruction output directory")
    p_view.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="Gaussian-smooth the recon image for display (default: on)")
    p_view.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Display smoothing sigma (default: 1.0)")
    p_view.add_argument("--lors", dest="lors_path", metavar="PATH", default=None,
                        help="Override the LOR file path recorded in the config pickle")

    # ---- config ----
    p_cfg = sub.add_parser("config", help="Print a TOML config template",
                           description="Print a commented TOML config template to stdout, "
                                       "or write it to a file with --write.")
    p_cfg.add_argument("--write", metavar="PATH", help="Write the template to PATH instead of stdout")

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = _build_config(args)
        try:
            cfg.validate()
        except ValueError as e:
            p_run.error(str(e))
        from petrecon.recon import run_reconstruction
        run_reconstruction(cfg)

    elif args.command == "batch":
        # Forward every explicitly-set run option to the per-job subprocesses
        extra = []
        if args.config:
            extra += ["--config", args.config]
        for name, value in _collect_overrides(args).items():
            if name in ("path_to_lors", "output_dir"):
                continue  # set per job by the batch driver
            flag = "--" + name.replace("_", "-")
            # Map field names back to their CLI flag spellings
            flag_map = {
                "--mu-map-path": "--mu-map", "--recon-algorithm": "--algorithm",
                "--osem-iterations": "--iterations", "--osem-subsets": "--subsets",
                "--n-axial-rings": "--rings", "--n-detectors-per-ring": "--detectors-per-ring",
                "--image-grid-size": "--grid", "--voxel-size-in-mm": "--voxel-size",
                "--sensitivity-correction": "--sensitivity",
                "--attenuation-correction": "--attenuation",
                "--scatter-correction": "--scatter", "--filter-oob-lors": "--filter-oob",
            }
            flag = flag_map.get(flag, flag)
            if isinstance(value, bool):
                extra.append(flag if value else flag.replace("--", "--no-", 1))
            elif isinstance(value, tuple):
                extra += [flag, *map(str, value)]
            else:
                extra += [flag, str(value)]
        from petrecon.batch import run_batch
        run_batch(args.base_dir, kinds=args.kinds, extra_run_args=extra)

    elif args.command == "view":
        if not Path(args.output_dir, "recon_config.pkl").exists():
            p_view.error(f"No recon_config.pkl in {args.output_dir} — is this a reconstruction output folder?")
        from petrecon.viewing import view_reconstruction
        view_reconstruction(args.output_dir, smooth=args.smooth,
                            smooth_sigma=args.smooth_sigma, lors_path=args.lors_path)

    elif args.command == "config":
        if args.write:
            Path(args.write).write_text(CONFIG_TEMPLATE)
            print(f"Wrote config template to {args.write}")
        else:
            sys.stdout.write(CONFIG_TEMPLATE)


if __name__ == "__main__":
    main()
