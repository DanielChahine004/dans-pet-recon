"""Batch reconstruction for point source positions.

Iterates through Point_Sources_Processed subfolders. For each position,
if the gt/ or pred/ subfolder is empty, runs run_recon.py with the
appropriate LOR CSV and output directory.
"""
import subprocess
import sys
import os
from pathlib import Path

# BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Processed"))
# BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_r235_to_257mm"))
# BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_polar_max50deg"))
BASE_DIR = Path(os.path.expanduser("~/Opengate/ragy_files/Point_Sources_Processed_OOB_filtered_recons"))
RECON_CONFIG_PATH = Path(__file__).parent / "recon_config.py"
RUN_RECON_PATH = Path(__file__).parent / "run_recon.py"

SKIP_DIRS = {"outside_fov", "pdf_outputs"}


def has_reconstruction(d: Path) -> bool:
    """Check if the output directory contains a completed reconstruction."""
    return (d / "reconstruction_image.npy").exists()


def get_lor_csv(pos_dir: Path, kind: str) -> Path | None:
    """Find the LOR CSV for a given position and kind (gt or pred).

    Matches any CSV starting with the expected prefix so that filtered
    variants (e.g. _polar_max50deg, _r235_to_257mm) are found automatically.
    """
    pos_name = pos_dir.name
    prefix = "gt_processed_" if kind == "gt" else "prediction_processed_"
    matches = sorted(pos_dir.glob(f"{prefix}{pos_name}*.csv"))
    return matches[0] if matches else None


def patch_config(original_text: str, path_to_lors: str, output_dir: str) -> str:
    """Replace PATH_TO_LORS and OUTPUT_DIR in the config file text."""
    lines = original_text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip comment-only lines
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        # Replace active PATH_TO_LORS assignment
        if stripped.startswith("PATH_TO_LORS") and "=" in stripped:
            new_lines.append(f"PATH_TO_LORS = '{path_to_lors}'")
        elif stripped.startswith("OUTPUT_DIR") and "=" in stripped:
            new_lines.append(f"OUTPUT_DIR = '{output_dir}'")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def main():
    # Read original config
    original_config = RECON_CONFIG_PATH.read_text()

    # Collect all (position, kind) pairs that need reconstruction
    jobs = []
    for pos_dir in sorted(BASE_DIR.iterdir()):
        if not pos_dir.is_dir() or pos_dir.name in SKIP_DIRS:
            continue

        for kind in ("gt", "pred"):
            sub_dir = pos_dir / kind
            if has_reconstruction(sub_dir):
                print(f"  SKIP {pos_dir.name}/{kind} (already has outputs)")
                continue

            csv_path = get_lor_csv(pos_dir, kind)
            if csv_path is None:
                print(f"  SKIP {pos_dir.name}/{kind} (no CSV found)")
                continue

            output_dir = str(sub_dir)
            jobs.append((pos_dir.name, kind, str(csv_path), output_dir))

    if not jobs:
        print("Nothing to reconstruct — all folders already have outputs.")
        return

    print(f"\n{len(jobs)} reconstruction(s) to run:\n")
    for pos_name, kind, csv_path, output_dir in jobs:
        print(f"  {pos_name}/{kind}")
    print()

    # Run each reconstruction
    for i, (pos_name, kind, csv_path, output_dir) in enumerate(jobs, 1):
        print("=" * 60)
        print(f"[{i}/{len(jobs)}] Reconstructing {pos_name}/{kind}")
        print(f"  LOR CSV:    {csv_path}")
        print(f"  Output dir: {output_dir}")
        print("=" * 60)

        # Patch config with this job's paths
        patched = patch_config(original_config, csv_path, output_dir)
        RECON_CONFIG_PATH.write_text(patched)

        try:
            result = subprocess.run(
                [sys.executable, str(RUN_RECON_PATH)],
                cwd=str(RUN_RECON_PATH.parent),
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"\n  ERROR: Reconstruction failed for {pos_name}/{kind} (exit code {e.returncode})")
            # Restore original config before exiting
            RECON_CONFIG_PATH.write_text(original_config)
            sys.exit(1)

        print(f"\n  Done: {pos_name}/{kind}\n")

    # Restore original config
    RECON_CONFIG_PATH.write_text(original_config)
    print("\nAll reconstructions complete. Original recon_config.py restored.")


if __name__ == "__main__":
    main()
