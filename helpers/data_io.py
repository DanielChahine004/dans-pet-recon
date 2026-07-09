"""Data loading utilities for PET reconstruction."""
from pathlib import Path
import numpy as np


def load_lors(path, default_path=None):
    """Load LOR data from CSV or NPZ file and filter invalid LORs.

    Args:
        path: Path to LOR file, or None to use default
        default_path: Default path if path is None

    Returns:
        lor_data: (N, 6) array of LOR endpoints [x1,y1,z1,x2,y2,z2]
    """
    if path is None:
        if default_path is None:
            default_path = Path(__file__).parent.parent.parent / "Post_process" / "Outputs" / "coincidence_pairs.npz"
        path = default_path

    path = Path(path)

    if path.suffix == '.csv':
        lor_data = np.loadtxt(path, delimiter=',', encoding='utf-8-sig', skiprows=1)
        assert lor_data.shape[1] == 6, f"CSV must have 6 columns, got {lor_data.shape[1]}"
    else:
        lor_data = np.load(path)
        try:
            # Check if it's already a single array
            lor_data.shape
        except AttributeError:
            # Handle dictionary-like npz objects
            xyz1 = lor_data['xyz1']
            xyz2 = lor_data['xyz2']
            lor_data = np.hstack([xyz1, xyz2])

    assert lor_data.shape[1] == 6, f"Data must have 6 columns, got {lor_data.shape[1]}"

    # 1. Filter out LORs with (0,0,0) endpoints
    xyz1_not_zero = ~np.all(lor_data[:, :3] == 0, axis=1)
    xyz2_not_zero = ~np.all(lor_data[:, 3:] == 0, axis=1)
    
    # 2. Filter out LORs where xyz1 and xyz2 are exactly the same (zero length)
    # This checks if [x1, y1, z1] == [x2, y2, z2] for every row
    not_identical = ~np.all(lor_data[:, :3] == lor_data[:, 3:], axis=1)

    # Combine all masks
    valid_mask = xyz1_not_zero & xyz2_not_zero & not_identical
    print(f"Loaded {lor_data.shape[0]} LORs from {path}. After filtering, {valid_mask.sum()} valid LORs remain.")
    lor_data = lor_data[valid_mask]

    return lor_data


def load_mu_map(path, image_grid_size):
    """Load mu-map for attenuation correction.

    Args:
        path: Path to mu-map .npy file, or None for zeros
        image_grid_size: Expected shape tuple (nx, ny, nz)

    Returns:
        mu_map: 3D array of attenuation coefficients (1/mm)
    """
    if path is not None:
        mu_map = np.load(path)
        assert mu_map.shape == image_grid_size, \
            f"mu_map shape {mu_map.shape} does not match IMAGE_GRID_SIZE {image_grid_size}"
        return mu_map.astype(np.float32)
    else:
        print("WARNING: No mu-map provided. Using zeros (no attenuation/scatter correction).")
        return np.zeros(image_grid_size, dtype=np.float32)


def load_gt_activity(path):
    """Load ground truth activity image for comparison."""
    if path is None:
        return np.array(0)
    return np.load(path)
