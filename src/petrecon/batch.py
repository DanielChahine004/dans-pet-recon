"""Batch reconstruction over a folder of datasets (formerly batch_recon.py).

Expects BASE_DIR to contain one subfolder per position, each holding LOR CSVs
named gt_processed_<pos>*.csv / prediction_processed_<pos>*.csv. Results go to
<pos>/gt and <pos>/pred. Positions that already contain a completed
reconstruction are skipped (resumable).

Each job runs in a fresh subprocess (`python -m petrecon run ...`) so a CUDA
failure in one job cannot poison the rest of the batch.
"""
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {"outside_fov", "pdf_outputs"}
PREFIXES = {"gt": "gt_processed_", "pred": "prediction_processed_"}


def has_reconstruction(d: Path) -> bool:
    """Check if the output directory contains a completed reconstruction."""
    return (d / "reconstruction_image.npy").exists()


def get_lor_csv(pos_dir: Path, kind: str) -> Path | None:
    """Find the LOR CSV for a given position and kind (gt or pred).

    Matches any CSV starting with the expected prefix so that filtered
    variants (e.g. _polar_max50deg, _r235_to_257mm) are found automatically.
    """
    matches = sorted(pos_dir.glob(f"{PREFIXES[kind]}{pos_dir.name}*.csv"))
    return matches[0] if matches else None


def run_batch(base_dir, kinds=("gt", "pred"), extra_run_args=None):
    """Reconstruct every position under base_dir that has no outputs yet.

    Args:
        base_dir: Folder of position subfolders.
        kinds: Which dataset kinds to process ("gt", "pred").
        extra_run_args: Extra CLI arguments forwarded to every `petrecon run`
            invocation (e.g. ["--config", "recon.toml"]).
    """
    base_dir = Path(base_dir).expanduser()
    if not base_dir.is_dir():
        sys.exit(f"Base directory does not exist: {base_dir}")
    extra_run_args = list(extra_run_args or [])

    # Collect all (position, kind) pairs that need reconstruction
    jobs = []
    for pos_dir in sorted(base_dir.iterdir()):
        if not pos_dir.is_dir() or pos_dir.name in SKIP_DIRS:
            continue

        for kind in kinds:
            sub_dir = pos_dir / kind
            if has_reconstruction(sub_dir):
                print(f"  SKIP {pos_dir.name}/{kind} (already has outputs)")
                continue

            csv_path = get_lor_csv(pos_dir, kind)
            if csv_path is None:
                print(f"  SKIP {pos_dir.name}/{kind} (no CSV found)")
                continue

            jobs.append((pos_dir.name, kind, str(csv_path), str(sub_dir)))

    if not jobs:
        print("Nothing to reconstruct — all folders already have outputs.")
        return

    print(f"\n{len(jobs)} reconstruction(s) to run:\n")
    for pos_name, kind, _, _ in jobs:
        print(f"  {pos_name}/{kind}")
    print()

    # Run each reconstruction in its own process
    for i, (pos_name, kind, csv_path, output_dir) in enumerate(jobs, 1):
        print("=" * 60)
        print(f"[{i}/{len(jobs)}] Reconstructing {pos_name}/{kind}")
        print(f"  LOR CSV:    {csv_path}")
        print(f"  Output dir: {output_dir}")
        print("=" * 60)

        cmd = [sys.executable, "-m", "petrecon", "run",
               "--lors", csv_path, "--output", output_dir, *extra_run_args]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"\n  ERROR: Reconstruction failed for {pos_name}/{kind} (exit code {e.returncode})")

        print(f"\n  Done: {pos_name}/{kind}\n")

    print("\nAll reconstructions complete.")
