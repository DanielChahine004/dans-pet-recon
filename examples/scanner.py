"""Single source of truth for the scanner geometry used by every engine.

petrecon's ReconConfig defaults, the OMEGA runner, the openSSS bridge and the
comparison dashboard must all describe the same cylinder — import these
constants instead of re-typing them (four copies had already crept in).
Values mirror petrecon.config.ReconConfig defaults; a unit test pins them.
"""
RINGS = 40
DETS_PER_RING = 400
INNER_RADIUS_MM = 235.422
OUTER_RADIUS_MM = 278.296
AXIAL_MM = 296.0
PITCH_Z_MM = AXIAL_MM / RINGS

# openSSS sector/module/crystal factorization of the uniform cylinder
SECTORS_TRANS, MODULES_TRANS, CRYSTALS_TRANS = 40, 1, 10   # = 400/ring
SECTORS_AXIAL, MODULES_AXIAL, CRYSTALS_AXIAL = 1, 4, 10    # = 40 rings
