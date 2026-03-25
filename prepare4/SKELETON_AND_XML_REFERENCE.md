# Skeleton, Body Model & XML Reference

## Overview

This document records verified findings about the body models, skeleton
representations, and MJCF XML generation pipeline used across InterGen,
InterMask, and our Newton physics simulation framework.

All claims below have been verified against source code and data files.
Where a statement was found incorrect it has been corrected (see §6).

---

## 1. SMPL-X Body Model

The InterHuman dataset (used by InterGen and InterMask) stores motion as
SMPL-X parameters in `.pkl` files:

```
pkl['person1'] = {
    'trans':       (T, 3)      root translation
    'root_orient': (T, 3)      root axis-angle
    'pose_body':   (T, 63)     21 body joints × 3  (axis-angle)
    'betas':       (10,)       shape parameters
    'gender':      'neutral'
}
```

- Body model file: `data/body_model/smplx/SMPLX_NEUTRAL.npz`
- J_regressor shape: (55, 10475) — maps vertices → 55 joint positions
- Rest-pose FK with betas: `BodyModel(betas=β) → Jtr (55, 3)`
- Mocap frame rate: 120 fps (raw), typically downsampled to 30 fps

### SMPL-X 22-Joint Kinematic Tree

Verified from `kintree_table` in SMPLX_NEUTRAL.npz:

```
Joint  Name          Parent  Parent Name
  0    pelvis          -1    (root)
  1    L_hip            0    pelvis
  2    R_hip            0    pelvis
  3    spine1           0    pelvis
  4    L_knee           1    L_hip
  5    R_knee           2    R_hip
  6    spine2           3    spine1
  7    L_ankle          4    L_knee
  8    R_ankle          5    R_knee
  9    spine3           6    spine2      (also called "Chest")
 10    L_foot           7    L_ankle     (also called "L_Toe")
 11    R_foot           8    R_ankle     (also called "R_Toe")
 12    neck             9    spine3
 13    L_collar         9    spine3      ← parent is spine3, NOT neck
 14    R_collar         9    spine3      ← parent is spine3, NOT neck
 15    head            12    neck
 16    L_shoulder      13    L_collar
 17    R_shoulder      14    R_collar
 18    L_elbow         16    L_shoulder
 19    R_elbow         17    R_shoulder
 20    L_wrist         18    L_elbow
 21    R_wrist         19    R_elbow
```

Parent chain (Python array):
```python
SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
```

**Important**: Joints 12 (neck), 13 (L_collar), 14 (R_collar) are all
children of joint 9 (spine3/Chest). The MJCF XML hierarchy matches this
exactly.

---

## 2. InterHuman NPY Representation

The InterGen/InterMask training pipeline uses pre-processed `.npy` files:

```
data/InterHuman/motions_processed/person{1,2}/<seq_id>.npy
  Shape: (T, 492)
  Layout: [positions(62×3=186) | rotations(51×6=306)]
```

### How NPY is converted to 262-dim training input

`load_motion()` in `data/utils.py` extracts the first 22 joints:
```python
motion1 = motion[:, :22*3]            # 66 dims — 22 joint positions
motion2 = motion[:, 62*3:62*3+21*6]   # 126 dims — 21 joint rotations (6D)
motion = concat([motion1, motion2])    # 192 dims per person
```

`process_motion_np()` then produces the 262-dim representation:
```python
positions   = motion[:, :66]   → transform + normalize → 66 dims
velocities  = positions[1:] - positions[:-1]            → 66 dims
rotations   = motion[:, 66:]                            → 126 dims (6D)
foot_contact= velocity/height threshold on joints 7,10,8,11 → 4 dims
─────────────────────────────────────────────────────────────────────
Total: 66 + 66 + 126 + 4 = 262 dims per person
```

### NPY Bone Lengths

- **Per-subject**: bone lengths vary across sequences (up to 5cm for
  long bones like ankle/knee). These reflect the original per-subject
  SMPL-X betas used during MoCap processing.
- **Constant within sequence**: bone lengths have std=0.000000 across
  frames — the skeleton is rigid within each sequence.
- **Coordinate system**: NPY positions are in a coordinate frame where
  Z is approximately up. `trans_matrix` in `data/utils.py` rotates them
  to Y-up for the 262-dim training representation.

### NPY Joint Positions vs SMPL-X FK

NPY positions do NOT come from the same `J_regressor` as `SMPLX_NEUTRAL.npz`:

| Metric                       | Value   |
|------------------------------|---------|
| NPY bone len vs SMPLX FK     | 3.3cm RMSE (with matching betas) |
| Largest per-bone difference   | head: −9.0cm, neck: +5.1cm       |
| Limb differences              | L_knee: −3.4cm, ankles: −3.2cm   |

This means if you generate rotations and apply SMPLX FK, the resulting
joint positions will be systematically different from the NPY distribution
that the model was trained on.

---

## 3. InterCLIP Evaluator & Metric Comparability

Evaluation metrics (FID, R-Precision, MM Dist, Diversity, MModality) are
computed using InterCLIP embeddings — a contrastive model trained on the
262-dim NPY representation.

The evaluator in `models/evaluator/evaluator_models.py`:
1. Takes concatenated person1+person2 motion (262×2 = 524 dim)
2. Strips foot contacts: `x = x.reshape(B,T,2,-1)[..., :-4].reshape(B,T,-1)`
3. Embeds via Linear + Transformer → 512-dim embedding

**Comparability rule**: metrics are comparable across methods if and only
if the same evaluator and the same 262-dim representation are used. Body
model differences do not affect standard metrics.

**FK concern**: if you generate rotations and apply FK with a different
body model to produce positions, the evaluator will see a distribution
shifted by the systematic bone-length error (~3cm RMSE). This can unfairly
affect metrics.

---

## 4. MJCF XML Generation

### 4.1 Template

The template XML is `prepare/assets/smpl.xml`. It contains 24 bodies:
Pelvis (root) + 23 joints, with 3-DOF hinge joints per body and
capsule/box collision geometry.

### 4.2 Two Bone Sources

We support two sources for per-subject body dimensions:

#### `bone_source="smplx"` (SMPL-X betas FK)

- Requires: `betas` array (10,) from PKL file
- Method: `BodyModel(betas=β) → Jtr (55,3)` rest-pose positions
- Then: `offset = R_ROT @ (child_joint - parent_joint)` → XML body `pos`
- Pro: uses the exact SMPL-X body model
- Con: resulting skeleton differs from NPY by ~3cm RMSE

#### `bone_source="npy"` (NPY position-derived)

- Requires: `npy_path` to InterHuman NPY file
- Method: extracts 22-joint positions from frame 0, computes Euclidean
  distance between each XML-hierarchy joint pair
- Then: scales the template T-pose offset DIRECTION by the ratio
  `npy_distance / template_distance`
- Pro: matches the representation space used in training/evaluation
- Con: body model provenance is unknown (different J_regressor)

### 4.3 Foot Geometry Variants

Three foot contact geometries are supported:

| Variant   | Ankle geoms        | Toe geoms        |
|-----------|-------------------|------------------|
| `box`     | 1 box             | 1 box            |
| `sphere`  | 3 spheres (heel + 2 balls) | 1 sphere |
| `capsule` | 1 capsule         | 1 capsule        |

### 4.4 Coordinate Transform

SMPL-X is Y-up. The XML / Newton is Z-up. The mapping is:

```
R_ROT = [[0, 0, 1],    SMPL-X Z (forward) → Newton X
         [1, 0, 0],    SMPL-X X (right)   → Newton Y
         [0, 1, 0]]    SMPL-X Y (up)      → Newton Z
```

### 4.5 File Organization

| Bone source | Foot geom | Cache directory            | Filename pattern          |
|-------------|-----------|----------------------------|---------------------------|
| smplx       | box       | `prepare4/xml_cache/`      | `smpl_box_<hash>.xml`     |
| smplx       | sphere    | `prepare4/xml_cache/`      | `smpl_sphere_<hash>.xml`  |
| smplx       | capsule   | `prepare4/xml_cache/`      | `smpl_capsule_<hash>.xml` |
| npy         | box       | `data/xml_npy/`            | `npy_box_<hash>.xml`      |
| npy         | sphere    | `data/xml_npy/`            | `npy_sphere_<hash>.xml`   |
| npy         | capsule   | `data/xml_npy/`            | `npy_capsule_<hash>.xml`  |
| npy (batch) | any       | `data/xml_npy/` (or custom)| `<seq>_person<1|2>.xml`   |

---

## 5. Usage

### Generate single XML

```python
from prepare4.gen_xml import generate_xml, get_or_create_xml

# From SMPL-X betas
xml = generate_xml(betas=betas, bone_source="smplx", foot_geom="box",
                   output_path="out.xml")

# From NPY positions
xml = generate_xml(npy_path="data/.../person1/42.npy",
                   bone_source="npy", foot_geom="sphere",
                   output_path="out.xml")

# Cached (hash-based dedup)
xml = get_or_create_xml(betas=betas, foot_geom="box")
xml = get_or_create_xml(npy_path="...", bone_source="npy", foot_geom="capsule")
```

### Batch generate from dataset

```bash
python prepare4/gen_xml.py --batch --data-root data/InterHuman \
       --output-dir data/xml_npy --split test
```

### View all variants in Newton GUI

```bash
python prepare4/view_xml_variants.py --clip 1000 --person 1
```

---

## 6. Corrections Record

| Date       | What was wrong                                                    | Correction                                                           |
|------------|-------------------------------------------------------------------|----------------------------------------------------------------------|
| 2026-03-04 | `SMPL_22_PARENTS` had L_collar(13)/R_collar(14) parented to neck(12) | Fixed: both are children of spine3(9), matching `kintree_table`   |
| 2026-03-04 | Docstring claimed XML hierarchy differs from SMPL kinematic chain    | Corrected: XML hierarchy matches SMPL-X `kintree_table` exactly  |
