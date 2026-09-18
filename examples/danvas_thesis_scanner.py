"""Thesis scanner + point-source layout on a danvas Model3D panel.

Programmatic CAD model of the continuous annular LaF3:Ce/PS scanner from
the PRISM thesis (Table 4.1, §4.2.1) together with the 25 point-source
positions of Table 5.1, all in one Model3D panel:

  * scintillator   — one monolithic annular prism: 104-facet inner surface
                     (apothem 235.422 mm), 124-facet outer surface (apothem
                     278.296 mm), 296 mm long. Translucent so the sources
                     inside the bore stay visible.
  * inner / outer SiPM tiles — 14 x 14 x 1.85 mm silicon tiles, one per facet
                     per ring (104 x 21 = 2184 inner, 124 x 21 = 2604 outer),
                     seated on the polyhedral faces.
  * sources        — 25 spheres (the 5 mm water sphere around each 18F point
                     source) at x = y in {0,32,64,96,120} mm crossed with
                     z in {0,32,64,96,120} mm, plus a pickable points layer.
  * fov            — wireframe of the 256 mm reconstruction cube for scale.

The thesis quotes the radii as the *minimum* radius of the polygon, i.e. the
apothem; facet width then works out to 14.22 mm (inner) / 14.10 mm (outer),
which is what fits a 14 mm tile. Axial tile pitch is 296 / 21 mm so the
photodetector stack spans exactly the scintillator length (the quoted 0.2 mm
gap with 14 mm tiles would give 298 mm).

  python examples/danvas_thesis_scanner.py   ->  http://127.0.0.1:8002
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
import danvas

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import INNER_RADIUS_MM, OUTER_RADIUS_MM, AXIAL_MM  # 235.422 / 278.296 / 296

# ---- Geometry (Table 4.1) --------------------------------------------------------
INNER_FACETS = 104          # 3.46 deg pitch
OUTER_FACETS = 124          # 3.10 deg pitch
RINGS = 21
TILE_MM = 14.0              # SiPM tile edge
TILE_THICK_MM = 1.85        # silicon thickness
PITCH_Z_MM = AXIAL_MM / RINGS
FOV_MM = 256.0              # reconstruction cube (Table 5.2)
SOURCE_RADIUS_MM = 5.0      # water sphere around each point source (§5.4.1)
SOURCE_OFFSETS = (0, 32, 64, 96, 120)   # Table 5.1: x = y = t, z = a
MM = 1e-3                   # Model3D world units are meters

OUT = Path(__file__).resolve().parent.parent / "example_output_thesis"
OUT.mkdir(exist_ok=True)


def _polygon(n, apothem, phase=0.0):
    """Vertices (n, 2) of a regular n-gon with inscribed radius `apothem`."""
    r = apothem / np.cos(np.pi / n)
    th = 2 * np.pi * np.arange(n) / n + phase
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def build_scintillator():
    """Annular prism: outer 124-gon minus inner 104-gon, extruded 296 mm.
    Side walls are quads per facet; the end caps bridge the two polygons
    with a merge-by-angle triangle strip (pure numpy, no shapely)."""
    pi_ = _polygon(INNER_FACETS, INNER_RADIUS_MM)
    po = _polygon(OUTER_FACETS, OUTER_RADIUS_MM)
    zs = (-AXIAL_MM / 2, AXIAL_MM / 2)
    V = np.vstack([np.column_stack([pi_, np.full(INNER_FACETS, z)]) for z in zs]
                  + [np.column_stack([po, np.full(OUTER_FACETS, z)]) for z in zs])
    i0, i1 = 0, INNER_FACETS                       # inner ring bottom / top
    o0, o1 = 2 * INNER_FACETS, 2 * INNER_FACETS + OUTER_FACETS

    def wall(base0, base1, n, flip):
        k = np.arange(n); k1 = (k + 1) % n
        a, b, c, d = base0 + k, base0 + k1, base1 + k1, base1 + k
        t = np.column_stack([a, b, c, a, c, d]).reshape(-1, 3)
        return t[:, ::-1] if flip else t

    def cap(ib, ob, flip):
        # walk both polygons in angle order, always advancing the lagging one
        ai = np.angle(pi_[:, 0] + 1j * pi_[:, 1]) % (2 * np.pi)
        ao = np.angle(po[:, 0] + 1j * po[:, 1]) % (2 * np.pi)
        tris, i, o = [], 0, 0
        while i < INNER_FACETS or o < OUTER_FACETS:
            ni, no = (i + 1) % INNER_FACETS, (o + 1) % OUTER_FACETS
            adv_i = (o >= OUTER_FACETS) or (i < INNER_FACETS and
                     ai[ni] + (2 * np.pi if ni == 0 else 0)
                     <= ao[no] + (2 * np.pi if no == 0 else 0))
            if adv_i:
                tris.append((ib + i, ob + o % OUTER_FACETS, ib + ni)); i += 1
            else:
                tris.append((ib + i % INNER_FACETS, ob + o, ob + no)); o += 1
        t = np.array(tris)
        return t[:, ::-1] if flip else t

    F = np.vstack([wall(o0, o1, OUTER_FACETS, False),   # outer wall, normals out
                   wall(i0, i1, INNER_FACETS, True),    # inner wall, normals into bore
                   cap(i0, o0, True),                   # bottom cap, normal -z
                   cap(i1, o1, False)])                 # top cap, normal +z
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def build_tiles(n_facets, apothem, outward):
    """One box per facet per ring, seated on the polygon face. `outward`
    puts the tile outside the surface (outer readout), otherwise inside."""
    r0 = apothem if outward else apothem - TILE_THICK_MM
    box = trimesh.creation.box(extents=(TILE_THICK_MM, TILE_MM, TILE_MM))
    v0 = box.vertices + np.array([r0 + TILE_THICK_MM / 2, 0.0, 0.0])
    f0 = np.asarray(box.faces)

    th = 2 * np.pi * np.arange(n_facets) / n_facets + np.pi / n_facets  # face normals
    c, s = np.cos(th), np.sin(th)
    rot = np.zeros((n_facets, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1] = c, -s
    rot[:, 1, 0], rot[:, 1, 1] = s, c
    rot[:, 2, 2] = 1.0
    ring_z = -AXIAL_MM / 2 + PITCH_Z_MM / 2 + np.arange(RINGS) * PITCH_Z_MM

    v = np.einsum("dij,vj->dvi", rot, v0)                      # (facets, 8, 3)
    verts = np.tile(v[None], (RINGS, 1, 1, 1))
    verts[..., 2] += ring_z[:, None, None]
    verts = verts.reshape(-1, 3)
    n = RINGS * n_facets
    faces = (f0[None] + (np.arange(n) * len(v0))[:, None, None]).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def source_positions():
    t, a = np.meshgrid(SOURCE_OFFSETS, SOURCE_OFFSETS, indexing="ij")
    return np.column_stack([t.ravel(), t.ravel(), a.ravel()]).astype(np.float64)


def build_sources(pts):
    """Merged icospheres, one per source, at the water-sphere radius."""
    ball = trimesh.creation.icosphere(subdivisions=2, radius=SOURCE_RADIUS_MM)
    parts = []
    for p in pts:
        m = ball.copy()
        m.apply_translation(p)
        parts.append(m)
    return trimesh.util.concatenate(parts)


FOV_EDGE_MM = 2.0            # beam thickness of the cube edges


def build_fov_frame():
    """The 256 mm cube as 12 solid beams (GL lines are 1 px and hard to
    see against the scintillator; a thin box per edge stays crisp)."""
    h, e = FOV_MM / 2, FOV_EDGE_MM
    parts = []
    for axis in range(3):
        ext = [e, e, e]
        ext[axis] = FOV_MM + e
        for sa in (-h, h):
            for sb in (-h, h):
                pos = [sa, sb]
                pos.insert(axis, 0.0)
                m = trimesh.creation.box(extents=ext)
                m.apply_translation(pos)
                parts.append(m)
    return trimesh.util.concatenate(parts)


RULE_MM = 300.0              # 30 cm rule, laid parallel to the scanner axis
RULE_W_MM, RULE_T_MM = 30.0, 4.0
RULE_OFFSET_MM = OUTER_RADIUS_MM + 40.0   # gap from the outer SiPM surface


def _rule_frame(pts, orient):
    """Rule-local (u along the rule, v across, w up out of the face) ->
    world. The rule is hung on a virtual wall at y = -RULE_OFFSET with its
    face pointing -y (towards the default camera), through the scanner's
    mid-plane:
      transaxial: u -> x, lying across the ring, level with z = 0
      axial:      u -> z, standing along the scanner axis
    """
    u, v, w = pts[:, 0], pts[:, 1], pts[:, 2]
    y = -RULE_OFFSET_MM - w
    if orient == "transaxial":
        return np.column_stack([u, y, v])
    return np.column_stack([v, y, u])


def build_rule(orient):
    """Rule body: a 300 mm flat bar centred on the scanner's mid-plane
    (axially it overhangs the 296 mm scanner by 2 mm at each end)."""
    m = trimesh.creation.box(extents=(RULE_MM, RULE_W_MM, RULE_T_MM))
    m.apply_translation([0, 0, -RULE_T_MM / 2])         # top face at w = 0
    m.vertices = _rule_frame(m.vertices, orient)
    return m


def build_rule_ticks(orient):
    """Raised graduations on the rule face: 1 cm short, 5 cm medium,
    10 cm full width; 31 ticks from 0 to 30 cm."""
    parts = []
    for k in range(31):
        length = RULE_W_MM * (1.0 if k % 10 == 0 else 0.6 if k % 5 == 0 else 0.35)
        t = trimesh.creation.box(extents=(1.2, length, 0.8))
        t.apply_translation([-RULE_MM / 2 + k * 10.0, -RULE_W_MM / 2 + length / 2, 0.4])
        parts.append(t)
    m = trimesh.util.concatenate(parts)
    m.vertices = _rule_frame(m.vertices, orient)
    return m


def glb(name, builder, *args):
    """Build once, cache as GLB in meters."""
    path = OUT / f"{name}.glb"
    if not path.exists():
        mesh = builder(*args)
        mesh.apply_scale(MM)
        path.write_bytes(mesh.export(file_type="glb"))
    return str(path)


# ---- Canvas ----------------------------------------------------------------------
canvas = danvas.Canvas()
viewer = canvas.model3d("scanner", label="Thesis scanner — LaF3:Ce/PS annulus + point sources",
                        x=40, y=40, w=1100, h=820)

pts = source_positions()
SCINT_COLOR = (120, 170, 235, 70)
INNER_COLOR = (200, 120, 60, 200)
OUTER_COLOR = (90, 90, 110, 200)
SOURCE_COLOR = (255, 60, 60, 255)
FOV_COLOR = (255, 200, 40, 255)
RULE_COLOR = (235, 225, 190, 255)
TICK_COLOR = (30, 30, 30, 255)


def show_layers():
    viewer.layer("scintillator").update(glb("scintillator", build_scintillator),
                                        mesh_color=SCINT_COLOR)
    if tiles_sel.value == "on":
        viewer.layer("inner_sipms").update(
            glb("inner_sipms", build_tiles, INNER_FACETS, INNER_RADIUS_MM, False),
            mesh_color=INNER_COLOR)
        viewer.layer("outer_sipms").update(
            glb("outer_sipms", build_tiles, OUTER_FACETS, OUTER_RADIUS_MM, True),
            mesh_color=OUTER_COLOR)
    else:
        viewer.clear("inner_sipms")
        viewer.clear("outer_sipms")
    viewer.layer("sources").update(glb("sources", build_sources, pts), mesh_color=SOURCE_COLOR)
    viewer.layer("source_points").points(pts * MM, color_by=pts[:, 2])
    if fov_sel.value == "on":
        viewer.layer("fov").update(glb("fov_frame", build_fov_frame), mesh_color=FOV_COLOR)
    else:
        viewer.clear("fov")
    if rule_sel.value == "on":
        o = rule_orient.value
        viewer.layer("rule").update(glb(f"rule_{o}", build_rule, o), mesh_color=RULE_COLOR)
        viewer.layer("rule_ticks").update(glb(f"rule_ticks_{o}", build_rule_ticks, o),
                                          mesh_color=TICK_COLOR)
    else:
        viewer.clear("rule")
        viewer.clear("rule_ticks")


col = canvas.column(x=1180, y=40, w=380, gap=14)
with col:
    canvas.markdown("#### Layers", name="sec_layers", h="auto")
    tiles_sel = canvas.toggle(["on", "off"], name="tiles", default="on", label="SiPM tiles")
    fov_sel = canvas.toggle(["on", "off"], name="fov", default="on",
                            label="256 mm reconstruction cube")
    rule_sel = canvas.toggle(["on", "off"], name="rule", default="on",
                             label="30 cm rule (1 cm ticks, long at 5 / 10 cm)")
    rule_orient = canvas.toggle(["transaxial", "axial"], name="rule_orient",
                                default="transaxial", label="Rule orientation")
    picked = canvas.label("picked", "click a source point to read its coordinates", h="auto")
    canvas.markdown(f"""
#### Scanner (Table 4.1)
- LaF3:Ce/PS continuous monolithic annulus, 296 mm long
- Inner surface: {INNER_FACETS} facets, apothem {INNER_RADIUS_MM} mm
  (facet width {2 * INNER_RADIUS_MM * np.tan(np.pi / INNER_FACETS):.2f} mm)
- Outer surface: {OUTER_FACETS} facets, apothem {OUTER_RADIUS_MM} mm
  (facet width {2 * OUTER_RADIUS_MM * np.tan(np.pi / OUTER_FACETS):.2f} mm)
- Radial thickness {OUTER_RADIUS_MM - INNER_RADIUS_MM:.3f} mm, bore Ø {2 * INNER_RADIUS_MM:.1f} mm
- SiPM tiles 14 × 14 × 1.85 mm: {INNER_FACETS * RINGS} inner, {OUTER_FACETS * RINGS} outer
  ({RINGS} rings, {PITCH_Z_MM:.3f} mm axial pitch)

#### Point sources (Table 5.1)
- 25 positions: x = y ∈ {{0, 32, 64, 96, 120}} mm × z ∈ {{0, 32, 64, 96, 120}} mm
- Drawn at the 5 mm water-sphere radius; points layer colored by z
- All lie on the x = y diagonal of the positive octant, the furthest at
  r = {120 * np.sqrt(2):.0f} mm transaxially and z = 120 mm axially
""", name="notes", h="auto")


@tiles_sel.on_change
def _(v):
    show_layers()


@fov_sel.on_change
def _(v):
    show_layers()


@rule_sel.on_change
def _(v):
    show_layers()


@rule_orient.on_change
def _(v):
    show_layers()


@viewer.on_pick
def _(msg):
    try:
        i = int(msg.get("point") if msg.get("point") is not None else -1)
        if msg.get("layer") == "source_points" and 0 <= i < len(pts):
            x, y, z = pts[i]
            picked.update(f"source ({x:.0f}, {y:.0f}, {z:.0f}) mm — "
                          f"r = {np.hypot(x, y):.1f} mm, |z| = {abs(z):.0f} mm")
        else:
            picked.update(f"picked: {msg}")
    except Exception as exc:
        picked.update(f"pick: {exc}")


show_layers()
viewer.view(eye=(0.55, -0.75, 0.45), look=(0, 0, 0), up=(0, 0, 1))

if __name__ == "__main__":
    canvas.serve(port=8002)
