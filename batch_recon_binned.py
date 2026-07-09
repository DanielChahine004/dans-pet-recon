"""Batch reconstruction with detector-binned LOR endpoints.

Phase 1: Snap GT and Pred LOR endpoints to 3 clinically relevant detector
         discretisation grids using KD-tree nearest-neighbour matching.
Phase 2: Run OSEM reconstruction for each binned dataset, patching
         recon_config.py with matching scanner geometry per job.

Skips positions/configs where outputs already exist (resumable).
"""
import subprocess
import sys
import os
import numpy as np
from pathlib import Path

BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Processed"))
BINNED_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Binned_new(osem)"))
RECON_CONFIG_PATH = Path(__file__).parent / "recon_config.py"
RUN_RECON_PATH = Path(__file__).parent / "run_recon.py"

SKIP_DIRS = {"outside_fov", "pdf_outputs"}

# Scanner geometry (must match recon_config.py)
RADIUS = 117.711 * 2       # inner radius in mm
AXIAL_LENGTH = 296          # mm

# Three clinically relevant detector discretisation levels
BINNING_CONFIGS = {
    "very_fine_500x50":{"n_det_per_ring": 500, "n_rings": 50},
    "fine_400x40":     {"n_det_per_ring": 400, "n_rings": 40},
    "moderate_300x30": {"n_det_per_ring": 300, "n_rings": 30},
    "standard_200x20": {"n_det_per_ring": 200, "n_rings": 20},
    "coarse_100x10":   {"n_det_per_ring": 100, "n_rings": 10},
}


def get_lor_csv(pos_dir, kind):
    """Find the LOR CSV for a given position and kind (gt or pred)."""
    pos_name = pos_dir.name
    if kind == "gt":
        csv_path = pos_dir / f"gt_processed_{pos_name}.csv"
    else:
        csv_path = pos_dir / f"prediction_processed_{pos_name}.csv"
    return csv_path if csv_path.exists() else None


def patch_config(original_text, path_to_lors, output_dir, n_rings, n_det_per_ring):
    """Replace config values for this job."""
    lines = original_text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        if stripped.startswith("PATH_TO_LORS") and "=" in stripped:
            new_lines.append(f"PATH_TO_LORS = '{path_to_lors}'")
        elif stripped.startswith("OUTPUT_DIR") and "=" in stripped:
            new_lines.append(f"OUTPUT_DIR = '{output_dir}'")
        elif stripped.startswith("N_AXIAL_RINGS") and "=" in stripped:
            new_lines.append(f"N_AXIAL_RINGS = {n_rings}")
        elif stripped.startswith("N_DETECTORS_PER_RING") and "=" in stripped:
            new_lines.append(f"N_DETECTORS_PER_RING = {n_det_per_ring}")
        elif stripped.startswith("RECON_ALGORITHM") and "=" in stripped:
            new_lines.append('RECON_ALGORITHM = "OSEM"')
        elif stripped.startswith("SENSITIVITY_CORRECTION") and "=" in stripped:
            new_lines.append("SENSITIVITY_CORRECTION = True")
        elif stripped.startswith("SCATTER_CORRECTION") and "=" in stripped:
            new_lines.append("SCATTER_CORRECTION = False")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def phase1_snap(overwrite=False):
    """Snap all LORs to each detector grid."""
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers.data_io import load_lors
    from helpers.plotting import snap_lors_to_detectors

    snapped_count = 0
    for pos_dir in sorted(BASE_DIR.iterdir()):
        if not pos_dir.is_dir() or pos_dir.name in SKIP_DIRS:
            continue

        for kind in ("gt",):
            csv_path = get_lor_csv(pos_dir, kind)
            if csv_path is None:
                continue

            binned_pos_dir = BINNED_DIR / pos_dir.name
            binned_pos_dir.mkdir(parents=True, exist_ok=True)

            # Check if all binning levels already have snapped LORs
            if not overwrite:
                all_exist = all(
                    (binned_pos_dir / f"{kind}_binned_{label}.csv").exists()
                    for label in BINNING_CONFIGS
                )
                if all_exist:
                    continue

            # Load LORs once per (position, kind)
            lor_data = load_lors(csv_path)

            for label, cfg in BINNING_CONFIGS.items():
                out_csv = binned_pos_dir / f"{kind}_binned_{label}.csv"
                if not overwrite and out_csv.exists():
                    continue

                snapped = snap_lors_to_detectors(
                    lor_data=lor_data,
                    n_rings=cfg["n_rings"],
                    n_det_per_ring=cfg["n_det_per_ring"],
                    radius=RADIUS,
                    axial_length=AXIAL_LENGTH,
                    max_dist_mm=2 * np.pi * RADIUS / cfg["n_det_per_ring"],
                )
                np.savetxt(out_csv, snapped, delimiter=",",
                           header="x1,y1,z1,x2,y2,z2", comments="")
                snapped_count += 1
                print(f"  Snapped {pos_dir.name}/{kind} → {label} ({snapped.shape[0]} LORs)")

    print(f"\nPhase 1 complete: {snapped_count} snapped LOR files created.\n")


def phase2_reconstruct(overwrite=False):
    """Run OSEM reconstruction for each binned LOR set."""
    original_config = RECON_CONFIG_PATH.read_text()

    # Collect jobs
    jobs = []
    for pos_dir in sorted(BASE_DIR.iterdir()):
        if not pos_dir.is_dir() or pos_dir.name in SKIP_DIRS:
            continue

        for kind in ("gt",):
            for label, cfg in BINNING_CONFIGS.items():
                out_dir = BINNED_DIR / pos_dir.name / f"{kind}_binned_{label}"
                csv_path = BINNED_DIR / pos_dir.name / f"{kind}_binned_{label}.csv"
                recon_path = out_dir / "reconstruction_image.npy"

                if not csv_path.exists():
                    continue
                if not overwrite and recon_path.exists():
                    print(f"  SKIP {pos_dir.name}/{kind}_binned_{label} (already reconstructed)")
                    continue

                jobs.append((pos_dir.name, kind, label, cfg, str(csv_path), str(out_dir)))

    if not jobs:
        print("Nothing to reconstruct — all binned outputs already exist.")
        return

    print(f"\n{len(jobs)} reconstruction(s) to run:\n")
    for pos_name, kind, label, _, _, _ in jobs:
        print(f"  {pos_name}/{kind}_binned_{label}")
    print()

    for i, (pos_name, kind, label, cfg, csv_path, output_dir) in enumerate(jobs, 1):
        print("=" * 60)
        print(f"[{i}/{len(jobs)}] Reconstructing {pos_name}/{kind}_binned_{label}")
        print(f"  LOR file:   {csv_path}")
        print(f"  Output dir: {output_dir}")
        print(f"  Geometry:   {cfg['n_det_per_ring']} det/ring × {cfg['n_rings']} rings")
        print("=" * 60)

        patched = patch_config(
            original_config, csv_path, output_dir,
            cfg["n_rings"], cfg["n_det_per_ring"],
        )
        RECON_CONFIG_PATH.write_text(patched)

        try:
            subprocess.run(
                [sys.executable, str(RUN_RECON_PATH)],
                cwd=str(RUN_RECON_PATH.parent),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n  ERROR: Reconstruction failed for {pos_name}/{kind}_binned_{label} "
                  f"(exit code {e.returncode})")
            RECON_CONFIG_PATH.write_text(original_config)
            sys.exit(1)

        print(f"\n  Done: {pos_name}/{kind}_binned_{label}\n")

    RECON_CONFIG_PATH.write_text(original_config)
    print("\nAll binned reconstructions complete. Original recon_config.py restored.")


def main():
    overwrite = False
    reply = input("Overwrite existing outputs? [y/N]: ").strip().lower()
    if reply in ("y", "yes"):
        overwrite = True
        print("Will overwrite existing snapped LORs and reconstructions.\n")
    else:
        print("Will skip existing outputs.\n")

    print("=" * 60)
    print("Phase 1: Snapping GT LOR endpoints to detector grids")
    print("=" * 60)
    phase1_snap(overwrite=overwrite)

    print("=" * 60)
    print("Phase 2: Running OSEM reconstructions")
    print("=" * 60)
    phase2_reconstruct(overwrite=overwrite)


if __name__ == "__main__":
    main()
