import math
# ==============================================================================
# PATHS
# ==============================================================================
PATH_TO_LORS = '/home/h/Opengate/ragy_files/IEC/iec_prediction_filtered.csv'
# MU_MAP_PATH = '/home/h/Opengate/Simulation/Outputs/mu_map.npy'       # Path to mu-map .npy file (required if ATTENUATION_CORRECTION=True)
MU_MAP_PATH = '/home/h/Opengate/Reconstructions/water_attenuation_map.npy'
# MU_MAP_PATH = None
OUTPUT_DIR = '/home/h/Opengate/ragy_files/IEC/pred'

# ==============================================================================
# RECONSTRUCTION ALGORITHM
# ==============================================================================
# Options: "Unfiltered_backprojection", "OSEM", "None"
RECON_ALGORITHM = "OSEM"

# OSEM parameters (only used if RECON_ALGORITHM == "OSEM")
OSEM_ITERATIONS = 4
OSEM_SUBSETS = 3

# ==============================================================================
# SCANNER GEOMETRY
# ==============================================================================
N_AXIAL_RINGS = 40
N_DETECTORS_PER_RING = 400
AXIAL_LENGTH = 296       # mm
INNER_RADIUS = 117.711 *2 # mm
OUTER_RADIUS = 139.148 *2  # mm

# ==============================================================================
# IMAGE GRID
# ==============================================================================
FOV_RADIUS = INNER_RADIUS * 0.95
IMAGE_GRID_SIZE = (256, 256, 256)
# IMAGE_GRID_SIZE = (200, 200, 200)
# IMAGE_GRID_SIZE = (128, 128, 128)
VOXEL_SIZE_IN_MM = 1 # was 2
# VOXEL_SIZE_IN_MM = (2 * FOV_RADIUS) / IMAGE_GRID_SIZE[0]  # isotropic voxels

# FOV filtering - only keep LORs that pass through the reconstruction FOV
FOV_AXIAL_HALF_LENGTH = (IMAGE_GRID_SIZE[2] * VOXEL_SIZE_IN_MM) // 2

# ==============================================================================
# CORRECTIONS
# ==============================================================================
# Sensitivity correction
SENSITIVITY_CORRECTION = True
SENSITIVITY_CHUNK_SIZE = 5_000_000  # LORs per CPU chunk during scanner LOR streaming (~120 MB at 5M)
SENSITIVITY_BATCH_SIZE = 5_000_000  # LORs per GPU kernel launch — tune to GPU VRAM (5M ≈ 184 MB on GPU for 256³ grid)
SMOOTH_SENSITIVITY = True
GAUSSIAN_SIGMA = 5
GAUSSIAN_RADIUS = 5

# Attenuation correction (requires MU_MAP_PATH)
ATTENUATION_CORRECTION = True

# Scatter correction - SSS (requires ATTENUATION_CORRECTION and MU_MAP_PATH)
SCATTER_CORRECTION = False

# ==============================================================================
# SCATTER CORRECTION (SSS) PARAMETERS
# ==============================================================================
SSS_COARSE_FACTOR = 8         # scatter point grid downsampling factor
SSS_N_ANGULAR_DETECTORS = 32  # coarse detector angular bins
SSS_N_AXIAL_DETECTORS = 15    # coarse detector axial bins
SSS_START_ITERATION = 2       # begin scatter correction after this many OSEM iterations
SSS_UPDATE_INTERVAL = 2       # re-estimate scatter every N iterations
SSS_SCALE_FACTOR = 1.0        # manual scatter scale factor

# ==============================================================================
# LOR FILTERING
# ==============================================================================
# Filter out LORs whose endpoints fall outside the physical scanner geometry.
# An endpoint is valid if: INNER_RADIUS <= r_xy <= OUTER_RADIUS and |z| <= AXIAL_LENGTH/2
FILTER_OOB_LORS = False