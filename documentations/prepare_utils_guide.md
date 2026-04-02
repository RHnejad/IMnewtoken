# prepare_utils — Shared Utilities Guide

## Overview

`prepare_utils/` consolidates code previously duplicated across prepare/, prepare2/, ..., prepare6/. All prepare folders now import shared constants, XML generation, and provenance tracking from here.

## Modules

### constants.py
Single source of truth for all SMPL-Newton mappings and simulation defaults.

```python
from prepare_utils.constants import (
    SMPL_TO_NEWTON,      # {smpl_joint_idx: newton_body_idx}  (22 entries)
    NEWTON_TO_SMPL,      # inverse mapping
    R_ROT,               # (3,3) SMPL-X body-local -> Newton body-local rotation
    BODY_TO_SMPLX,       # {"Pelvis": 0, "L_Hip": 1, ...}
    BODY_NAMES,          # 24 body names in Newton order
    DOF_NAMES,           # 75 DOF names
    SMPL_22_PARENTS,     # kinematic tree parent indices
    # Dimensions
    COORDS_PER_PERSON,   # 76 (7 freejoint + 69 hinges)
    DOFS_PER_PERSON,     # 75 (6 freejoint + 69 hinges)
    BODIES_PER_PERSON,   # 24
    N_SMPL_JOINTS,       # 22
    # Simulation defaults
    DEFAULT_FPS,         # 30
    DEFAULT_SIM_FREQ,    # 480
    ARMATURE_HINGE,      # 0.5
    ARMATURE_ROOT,       # 5.0
)
```

Previously defined independently in: prepare2/pd_utils.py, prepare2/retarget.py, prepare4/retarget.py, prepare4/gen_xml.py, prepare5/phc_config.py.

### gen_xml.py
Per-subject MJCF XML generation. Supports three foot geometries and two joint styles.

```python
from prepare_utils.gen_xml import generate_xml, get_or_create_xml

# Default style (our original conservative limits/gains)
xml = get_or_create_xml(betas=betas, foot_geom="box")

# PHC style (PHC-identical joint limits, gains, capsule torso geoms)
xml = get_or_create_xml(betas=betas, joint_style="phc")

# Sphere feet variant
xml = get_or_create_xml(betas=betas, foot_geom="sphere")
```

**Parameters:**
- `betas`: (10,) numpy array of SMPL-X shape parameters
- `foot_geom`: `"box"` | `"sphere"` | `"capsule"`
- `joint_style`: `"default"` | `"phc"`
- `bone_source`: `"smplx"` (from betas) | `"npy"` (from InterHuman positions)

**Cache locations:**
| Style | Directory |
|-------|-----------|
| default | `prepare_utils/xml_cache/` |
| phc (heuristic fallback) | `prepare_utils/xml_cache_phc/` |
| phc (SMPL_Robot, exact) | `prepare_utils/xml_cache_phc_robot/` |
| npy bone source | `data/xml_npy/` |

### smpl_robot_bridge.py
Bridge to PHC's original `SMPL_Robot` class from the `smpl_sim` package (cloned at `prepare_utils/smpl_sim_repo/`).

When `joint_style="phc"` is used and SMPL_Robot dependencies are available, `get_or_create_xml` automatically uses this bridge for exact PHC-identical XML output with mesh-derived capsule sizing.

```python
from prepare_utils.smpl_robot_bridge import is_available, get_or_create_phc_xml

if is_available():
    xml = get_or_create_phc_xml(betas, gender=0)  # 0=neutral, 1=male, 2=female
```

**Dependencies** (in mimickit conda env):
- torch, numpy, scipy, lxml, mujoco, smplx, joblib, torchgeometry, easydict, numpy-stl, opencv-python-headless

**Fallback**: If SMPL_Robot is unavailable (e.g., missing packages), `get_or_create_xml(joint_style="phc")` falls back to `_apply_phc_style()` — a heuristic that applies PHC joint/actuator/density properties to our template-based XML. The heuristic matches PHC for joints, gains, and densities but approximates capsule fromto values instead of computing them from mesh convex hulls.

### provenance.py
Output metadata tagging for traceability.

```python
from prepare_utils.provenance import save_with_provenance, check_provenance

# Save with metadata sidecar
save_with_provenance("data/torques/clip_1000.npy", torques,
                     source="prepare4/dynamics.py",
                     params={"fps": 30, "method": "pd"})

# Validate origin
ok, warnings = check_provenance("data/torques/clip_1000.npy",
                                expected_params={"fps": 30})
```

## PHC Joint Style — What Changes

| Property | `default` | `phc` |
|----------|-----------|-------|
| Joint ranges | Conservative (e.g., knee +-30deg) | +-180deg everywhere (+-720deg for arms) |
| Stiffness | 200-1000 (body-varied) | 500-1000 (PHC body-group tiers) |
| Damping | 20-100 | 50-100 |
| Motor gear | 25-300 (body-varied) | 500 (uniform) |
| Torso/Spine/Chest/Neck geom | sphere | capsule |
| Pelvis density | 2000 | 4630 |
| Limb densities | 1000-2000 | PHC-specific (1000-4630) |

**When to use PHC style:**
- When retargeted motions exceed default joint limits (generates constraint forces → false torque spikes)
- When comparing against PHC-based methods (ImDy, PP-Motion)
- When you want collision geometry closer to the SMPL mesh surface

**When to keep default:**
- When you need backward compatibility with existing cached results
- When joint limit enforcement is desired for physically implausible motion detection

## CLI Usage

All prepare4 scripts now accept `--joint-style`:

```bash
# Retarget with PHC-style skeleton
python prepare4/retarget.py --clip 1000 --joint-style phc

# Visualize GT vs generated with PHC XML
python prepare4/view_gt_vs_gen.py --clip 1000 --joint-style phc

# Run dynamics analysis
python prepare4/dynamics.py --clip 1000 --torques pd --joint-style phc

# Full analysis pipeline
python prepare4/run_full_analysis.py --clips 1000 hit --joint-style phc
```

## Import Hierarchy

```
prepare_utils/constants.py        <- single source of truth
prepare_utils/gen_xml.py          <- canonical XML generator
prepare_utils/smpl_robot_bridge.py <- PHC SMPL_Robot bridge
prepare_utils/smpl_sim_repo/      <- cloned from github.com/ZhengyiLuo/SMPLSim
prepare_utils/provenance.py       <- output metadata

    prepare/retarget_newton.py    -> imports constants
    prepare2/pd_utils.py          -> imports constants
    prepare4/gen_xml.py           -> re-exports from prepare_utils/gen_xml.py
    prepare4/retarget.py          -> imports gen_xml, constants
    prepare4/dynamics.py          -> imports gen_xml
    prepare4/view_gt_vs_gen.py    -> imports gen_xml
    prepare5/phc_config.py        -> imports constants
    prepare6/*.py                 -> imports via prepare4/gen_xml.py (re-export)
    newton_vqvae/*.py             -> imports constants, gen_xml
```

## Backward Compatibility

`prepare4/gen_xml.py` is now a thin re-export shim. All existing imports like `from prepare4.gen_xml import get_or_create_xml` continue to work unchanged. The `joint_style` parameter defaults to `"default"`, so existing code produces identical output.

## XML Cache Management

Cached XMLs are identified by SHA-256 hash of (bone_source, foot_geom, joint_style, betas). To regenerate:

```bash
# Clear specific cache
rm -rf prepare_utils/xml_cache_phc_robot/

# Clear all caches
rm -rf prepare_utils/xml_cache/ prepare_utils/xml_cache_phc/ prepare_utils/xml_cache_phc_robot/
```
