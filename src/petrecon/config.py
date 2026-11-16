"""Reconstruction configuration.

Defaults live on the ReconConfig dataclass. Values can be overridden by a
TOML config file (petrecon run --config recon.toml) and again by CLI flags.
Precedence: dataclass defaults < TOML file < CLI flags.
"""
from dataclasses import dataclass, fields
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class ReconConfig:
    # ---- Paths ----
    path_to_lors: str | None = None   # (N,6) CSV or NPZ of LOR endpoints [x1,y1,z1,x2,y2,z2] in mm
    output_dir: str | None = None
    mu_map_path: str | None = None    # .npy mu-map volume (enables attenuation correction)

    # ---- Algorithm ----
    recon_algorithm: str = "OSEM"     # "OSEM" | "Unfiltered_backprojection" | "None"
    osem_iterations: int = 4
    osem_subsets: int = 3

    # ---- Scanner geometry ----
    n_axial_rings: int = 40
    n_detectors_per_ring: int = 400
    axial_length: float = 296.0       # mm
    inner_radius: float = 235.422     # mm
    outer_radius: float = 278.296     # mm

    # ---- Image grid ----
    image_grid_size: tuple[int, int, int] = (256, 256, 256)
    voxel_size_in_mm: float = 1.0
    fov_radius: float | None = None            # default: 0.95 * inner_radius
    fov_axial_half_length: float | None = None  # default: grid_z * voxel_size // 2

    # ---- Corrections ----
    sensitivity_correction: bool = True
    sensitivity_chunk_size: int = 5_000_000  # unused: the scanner LORs are now generated on the GPU (kept for old configs)
    sensitivity_batch_size: int = 5_000_000  # unused: see sensitivity_chunk_size
    # "grid": point detectors at the ring/detector positions (classic; aliases on the scanner axis, where some
    # voxels get ~zero sensitivity, so it needs smooth_sensitivity). "continuous": each detector-pair endpoint
    # uniform within its detector cell, which matches continuous event positions and has no on-axis hole.
    sensitivity_mode: str = "grid"
    sensitivity_samples: int = 1             # independent draws of every detector pair ("continuous" only)
    smooth_sensitivity: bool = True
    gaussian_sigma: float = 5.0              # in voxels (so its width in mm changes with voxel_size_in_mm)
    gaussian_radius: int = 5                 # truncation radius in voxels (5 with sigma 5 = cut at 1 sigma)
    gaussian_sigma_mm: float | None = None   # if set, overrides gaussian_sigma: blur sigma in mm
    gaussian_radius_mm: float | None = None  # truncation radius in mm (default: gaussian_sigma_mm, i.e. 1 sigma)
    attenuation_correction: bool | None = None  # None = auto: on iff mu_map_path is set
    scatter_correction: bool = False

    # ---- Single scatter simulation (SSS) ----
    sss_coarse_factor: int = 8          # scatter point grid downsampling factor
    sss_n_angular_detectors: int = 32   # coarse detector angular bins
    sss_n_axial_detectors: int = 15     # coarse detector axial bins
    sss_start_iteration: int = 2        # begin scatter correction after this many OSEM iterations
    sss_update_interval: int = 2        # re-estimate scatter every N iterations
    sss_scale_factor: float = 1.0       # manual scatter scale factor

    # ---- LOR filtering ----
    # Drop LORs whose endpoints fall outside the physical scanner geometry:
    # valid iff INNER_RADIUS <= r_xy <= OUTER_RADIUS and |z| <= AXIAL_LENGTH/2
    filter_oob_lors: bool = False

    def __post_init__(self):
        self.image_grid_size = tuple(int(v) for v in self.image_grid_size)
        if self.attenuation_correction is None:
            self.attenuation_correction = self.mu_map_path is not None
        if self.fov_radius is None:
            self.fov_radius = self.inner_radius * 0.95
        if self.fov_axial_half_length is None:
            self.fov_axial_half_length = (self.image_grid_size[2] * self.voxel_size_in_mm) // 2

    # ------------------------------------------------------------------
    def validate(self):
        """Check the configuration is runnable; raises ValueError otherwise."""
        errors = []
        if self.path_to_lors is None:
            errors.append("path_to_lors is required (--lors)")
        elif not Path(self.path_to_lors).exists():
            errors.append(f"LOR file does not exist: {self.path_to_lors}")
        if self.output_dir is None:
            errors.append("output_dir is required (--output)")
        if self.recon_algorithm not in ("Unfiltered_backprojection", "OSEM", "None"):
            errors.append(f"Invalid recon_algorithm: {self.recon_algorithm!r}")
        if self.scatter_correction:
            if self.mu_map_path is None:
                errors.append("scatter_correction requires mu_map_path (--mu-map)")
            if not self.attenuation_correction:
                errors.append("scatter_correction requires attenuation_correction")
        if self.attenuation_correction:
            if self.mu_map_path is None:
                errors.append("attenuation_correction requires mu_map_path (--mu-map)")
            elif not Path(self.mu_map_path).exists():
                errors.append(f"mu_map_path does not exist: {self.mu_map_path}")
        if self.sensitivity_mode not in ("grid", "continuous"):
            errors.append(f"sensitivity_mode must be 'grid' or 'continuous', got {self.sensitivity_mode!r}")
        if self.sensitivity_samples < 1:
            errors.append("sensitivity_samples must be >= 1")
        if self.gaussian_sigma_mm is not None and self.gaussian_sigma_mm <= 0:
            errors.append("gaussian_sigma_mm must be > 0")
        if len(self.image_grid_size) != 3:
            errors.append(f"image_grid_size must have 3 dimensions, got {self.image_grid_size}")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------
    def sensitivity_blur_voxels(self) -> tuple[float, int]:
        """(sigma, truncation radius) of the sensitivity smoothing, in voxels."""
        if self.gaussian_sigma_mm is None:
            return float(self.gaussian_sigma), int(self.gaussian_radius)
        radius_mm = self.gaussian_radius_mm if self.gaussian_radius_mm is not None else self.gaussian_sigma_mm
        return self.gaussian_sigma_mm / self.voxel_size_in_mm, max(1, int(round(radius_mm / self.voxel_size_in_mm)))

    # ------------------------------------------------------------------
    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_toml(cls, path, overrides: dict | None = None) -> "ReconConfig":
        """Load config from a flat TOML file, optionally applying overrides on top."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        data = {k.lower(): v for k, v in data.items()}
        unknown = set(data) - cls.field_names()
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        if overrides:
            data.update(overrides)
        return cls(**data)

    # ------------------------------------------------------------------
    def to_legacy_dict(self) -> dict:
        """Dict with the historical UPPERCASE keys, as saved to recon_config.pkl.

        Downstream viewers/analysis scripts read this pickle, so the key set
        must stay stable.
        """
        is_osem = self.recon_algorithm == "OSEM"
        # the effective blur in voxels (viewers re-apply it with these two keys)
        blur_sigma, blur_radius = self.sensitivity_blur_voxels()
        return {
            "OUTPUT_DIR": self.output_dir,
            "PATH_TO_LORS": self.path_to_lors,
            "RECON_ALGORITHM": self.recon_algorithm,
            "SENSITIVITY_CORRECTION": self.sensitivity_correction,
            "SMOOTH_SENSITIVITY": self.smooth_sensitivity,
            "GAUSSIAN_SIGMA": blur_sigma,
            "GAUSSIAN_RADIUS": blur_radius,
            "GAUSSIAN_SIGMA_MM": self.gaussian_sigma_mm,
            "SENSITIVITY_MODE": self.sensitivity_mode,
            "SENSITIVITY_SAMPLES": self.sensitivity_samples,
            "IMAGE_GRID_SIZE": self.image_grid_size,
            "VOXEL_SIZE_IN_MM": self.voxel_size_in_mm,
            "N_AXIAL_RINGS": self.n_axial_rings,
            "N_DETECTORS_PER_RING": self.n_detectors_per_ring,
            "AXIAL_LENGTH": self.axial_length,
            "INNER_RADIUS": self.inner_radius,
            "OUTER_RADIUS": self.outer_radius,
            "OSEM_ITERATIONS": self.osem_iterations if is_osem else None,
            "OSEM_SUBSETS": self.osem_subsets if is_osem else None,
            "FOV_RADIUS": self.fov_radius,
            "FOV_AXIAL_HALF_LENGTH": self.fov_axial_half_length,
            "ATTENUATION_CORRECTION": self.attenuation_correction,
            "MU_MAP_PATH": self.mu_map_path,
            "SCATTER_CORRECTION": self.scatter_correction,
            "SSS_COARSE_FACTOR": self.sss_coarse_factor,
            "SSS_N_ANGULAR_DETECTORS": self.sss_n_angular_detectors,
            "SSS_N_AXIAL_DETECTORS": self.sss_n_axial_detectors,
            "SSS_START_ITERATION": self.sss_start_iteration,
            "SSS_UPDATE_INTERVAL": self.sss_update_interval,
            "SSS_SCALE_FACTOR": self.sss_scale_factor,
            "FILTER_OOB_LORS": self.filter_oob_lors,
        }
