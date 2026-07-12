"""Programmatic CAD model of the binned reconstruction's equivalent scanner.

Builds the crystal grid the sinogram binning quantizes onto — DETS_PER_RING
segments around the ring x RINGS axial rows, each crystal a radial box from
the inner to the outer detector radius, with visible gaps between segments —
as one GLB mesh for the danvas Model3D panel. Geometry comes from scanner.py
(the single source of truth), so the model is exactly the bin structure.

Built with trimesh (vectorized; a CAD kernel like build123d would model
16,000 solids through OCC for no visual gain). World units: meters (glTF).

    python scanner_mesh.py            -> writes scanner_crystals.glb here
    build_crystal_grid()              -> trimesh.Trimesh (mm, unscaled)
    crystal_grid_glb(path)            -> cached GLB in meters, returns path
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import (RINGS, DETS_PER_RING, INNER_RADIUS_MM, OUTER_RADIUS_MM,
                     AXIAL_MM, PITCH_Z_MM)


def build_crystal_grid(dets=DETS_PER_RING, rings=RINGS):
    """One mesh containing every crystal box (mm units) for a virtual
    scanner with `dets` transaxial segments x `rings` axial rows on the
    physical cylinder. Gaps scale with pitch (capped at 1 mm)."""
    import trimesh

    pitch_t = 2 * np.pi * INNER_RADIUS_MM / dets
    pitch_z = AXIAL_MM / rings
    width = pitch_t - min(1.0, 0.27 * pitch_t)   # tangential
    height = pitch_z - min(1.0, 0.14 * pitch_z)  # axial
    depth = OUTER_RADIUS_MM - INNER_RADIUS_MM    # radial (DOI extent)

    # template box centered on +x at the crystal's radial midline
    box = trimesh.creation.box(extents=(depth, width, height))
    v0 = box.vertices + np.array([INNER_RADIUS_MM + depth / 2, 0.0, 0.0])
    f0 = np.asarray(box.faces)

    theta = np.arange(dets) * 2 * np.pi / dets
    c, s = np.cos(theta), np.sin(theta)
    rot = np.zeros((dets, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1] = c, -s
    rot[:, 1, 0], rot[:, 1, 1] = s, c
    rot[:, 2, 2] = 1.0
    ring_z = -AXIAL_MM / 2 + pitch_z / 2 + np.arange(rings) * pitch_z

    # (dets, V, 3) rotated template, tiled over rings with z offsets
    v_det = np.einsum("dij,vj->dvi", rot, v0)
    verts = np.tile(v_det[None], (rings, 1, 1, 1))
    verts[..., 2] += ring_z[:, None, None]
    verts = verts.reshape(-1, 3)

    n_boxes = rings * dets
    faces = (f0[None] + (np.arange(n_boxes) * len(v0))[:, None, None]).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def crystal_grid_glb(path, **kw):
    """Write (once) and return the GLB path, scaled to meters."""
    path = Path(path)
    if not path.exists():
        mesh = build_crystal_grid(**kw)
        mesh.apply_scale(1e-3)                   # mm -> m (glTF convention)
        path.write_bytes(mesh.export(file_type="glb"))
    return str(path)


if __name__ == "__main__":
    out = crystal_grid_glb(Path(__file__).parent / "scanner_crystals.glb")
    print("wrote", out)
