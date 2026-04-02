"""
prepare_utils — Shared utilities for InterMask physics evaluation pipeline.

Consolidates code previously duplicated across prepare/, prepare2/, ..., prepare6/.
All prepare folders should import from here instead of redefining constants,
retargeting functions, XML generation, or PD control utilities.

Modules:
    constants   — SMPL↔Newton mappings, dimension constants, coordinate transforms
    xml_gen     — Per-subject MJCF XML generation (unified from prepare2 + prepare4)
    retarget    — Rotation-based and IK-based retargeting, FK
    pd          — PD control gains, torque computation, model building
    provenance  — Output metadata tagging for traceability
"""
