"""Minimal reproducers for omegatomo 2.2.0 list-mode reconstructions_main issues.

Usage: python omega_repro_listmode.py <1|2|3>
  1 = coordinate listmode + compute_sensitivity_image=True  (transaxial collapse)
  2 = coordinate listmode + compute_sensitivity_image=False (all-zero output)
  3 = index-based listmode + compute_sensitivity_image=True (flat noise + NaNs)

Data: point source at (96,96,96) mm; correct peak on this 128^3/2mm grid is
~(112,112,110) — confirmed by LOR convergence and by two independent
reconstructions (petrecon; OMEGA's own projector via the CuPy custom path).
See omega_upstream_issues.md.
"""
import sys
import numpy as np
from omegatomo.projector import proj
from omegatomo.reconstruction import reconstructions_main

CASE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
lors = np.loadtxt("/home/h/Opengate/ragy_files/Point_Sources_Processed/96_96_96/"
                  "gt_processed_96_96_96.csv", delimiter=",", skiprows=1,
                  dtype=np.float32)[:5000]

options = proj.projectorClass()
options.blocks_per_ring = 40; options.cryst_per_block = 10
options.linear_multip = 4; options.cryst_per_block_axial = 10
options.diameter = 2 * 235.422
options.cr_p = np.pi * 2 * 235.422 / 400
options.cr_pz = 296.0 / 40
options.det_per_ring = 400; options.det_w_pseudo = 400
options.rings = 40; options.detectors = 16000
options.FOVa_x = options.FOVa_y = options.axial_fov = 256.0
options.Nx = options.Ny = options.Nz = 128
options.OSEM = True
options.Niter = 2; options.subsets = 3; options.subsetType = 1
options.projector_type = 1
options.x0 = np.ones((128, 128, 128), dtype=np.float32)
options.useCUDA = True
options.verbose = 1

if CASE in (1, 2):
    options.x = np.asfortranarray(lors.T)
    options.SinM = np.ones(len(lors), dtype=np.float32)
    options.compute_sensitivity_image = CASE == 1
else:
    def to_idx(p):
        tr = np.round(np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi / 400)).astype(np.int64) % 400
        ax = np.clip(np.round((p[:, 2] + 148.0 - 3.7) / 7.4), 0, 39)
        return tr.astype(np.uint16), ax.astype(np.uint16)
    t1, a1 = to_idx(lors[:, :3]); t2, a2 = to_idx(lors[:, 3:])
    keep = ~((t1 == t2) & (a1 == a2))
    options.useIndexBasedReconstruction = True
    options.trIndex = np.asfortranarray(np.stack([t1[keep], t2[keep]]))
    options.axIndex = np.asfortranarray(np.stack([a1[keep], a2[keep]]))
    options.SinM = np.ones(int(keep.sum()), dtype=np.float32)
    options.compute_sensitivity_image = True

pz, fp = reconstructions_main(options)
img = np.squeeze(np.asarray(pz))
if img.ndim == 4:
    img = img[..., -1]
finite = img[np.isfinite(img)]
print(f"case {CASE}: max {np.nanmax(img):.4g} | finite max {finite.max():.4g} | "
      f"NaNs {int(np.isnan(img).sum())} | argmax {np.unravel_index(np.nanargmax(np.nan_to_num(img)), img.shape)} "
      f"| expected peak ~(112, 112, 110)")
