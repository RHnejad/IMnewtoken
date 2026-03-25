# prepare/ — Motion Extraction & Retargeting Pipeline (Generic Skeleton)

Extract joint positions from InterHuman/Inter-X datasets, retarget onto Newton's SMPL skeleton, and visualize results.

## Overview

This pipeline operates in phases:

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `extract_joint_positions.py` | Extract 22-joint positions from raw dataset |
| 2 | `retarget_newton.py` | IK-based retargeting onto Newton skeleton (~1.7cm MPJPE) |
| 2b | `retarget_rotation.py` | Direct rotation transfer (exact rotations, proportional position error) |
| 3 | `visualize_*.py` | Visualize results in matplotlib or Newton viewer |

> **Note:** This pipeline uses a **generic** `smpl.xml` skeleton (single set of bone lengths for all subjects). For per-subject skeletons matching each person's body shape, see [`prepare2/`](../prepare2/README.md).

## Requirements

- **Conda env:** `mimickit` (Python 3.10)
- **GPU:** CUDA 12.9, Newton/Warp
- **Data:**
  - `data/InterHuman/` — InterHuman dataset (motions_processed/)
  - `data/Inter-X_Dataset/` — Inter-X dataset (H5 files)
  - `data/body_model/smplx/SMPLX_NEUTRAL.npz` — SMPL-X body model

## Coordinate Systems

| Frame | Convention | Usage |
|-------|-----------|-------|
| SMPL-X | X-left, Y-up, Z-forward | Input data |
| InterHuman world | Z-up | Raw dataset frame |
| Inter-X world | Y-up | Raw dataset frame (auto-converted to Z-up) |
| Newton XML (`R_ROT`) | `[[0,0,1],[1,0,0],[0,1,0]]` | Body offsets, rotation conjugation |
| Newton FK output | Z-up world | Direct (same as InterHuman) |

**Position mapping:** Direct (no rotation). InterHuman data is Z-up, Newton `up_axis=Z`. Root orient encodes the SMPL-X Y-up → Z-up rotation.

**Root orientation:** `R_newton = R(root_orient) @ R_ROT^T`.

**Body joints:** `R_hinge = R_ROT @ R_smplx @ R_ROT^T` → extrinsic Euler XYZ.

## Newton joint_q Layout (76 values)

```
[0:3]   = Pelvis position: tx, ty, tz
[3:7]   = Pelvis orientation quaternion: qx, qy, qz, qw  (xyzw format)
[7:10]  = L_Hip:    extrinsic Euler XYZ (3 hinges)
[10:13] = L_Knee:   extrinsic Euler XYZ
[13:16] = L_Ankle:  extrinsic Euler XYZ
[16:19] = L_Toe:    extrinsic Euler XYZ
[19:22] = R_Hip:    extrinsic Euler XYZ
[22:25] = R_Knee:   extrinsic Euler XYZ
...     (depth-first order, 3 hinges per body, 24 bodies total)
```

Newton bodies 18 (L_Hand) and 23 (R_Hand) have no SMPL-X equivalent — their angles stay at 0.

## Scripts

### Phase 1: Position Extraction

```bash
# InterHuman — reshapes pre-computed 492-dim vectors
python prepare/extract_joint_positions.py --dataset interhuman

# Inter-X — runs SMPL-X FK on raw H5 parameters
python prepare/extract_joint_positions.py --dataset interx
```

**Output:** `data/extracted_positions/{dataset}/{clip_id}_person{0,1}.npy` — shape `(T, 22, 3)`

### Phase 2: Retargeting

#### IK-based (position accuracy, ~1.7cm MPJPE)

```bash
# Single clip
python prepare/retarget_newton.py \
    --input_dir data/extracted_positions/interhuman \
    --output_dir data/retargeted/interhuman \
    --clip 1000_person0

# Batch all clips
python prepare/retarget_newton.py \
    --input_dir data/extracted_positions/interhuman \
    --output_dir data/retargeted/interhuman
```

**Output:** `data/retargeted/{dataset}/{clip_id}.npy` — shape `(T, 22, 3)` positions

#### Rotation-based (exact joint rotations)

```bash
# Single clip
python prepare/retarget_rotation.py --dataset interhuman --clip 1000

# Batch + save joint_q arrays
python prepare/retarget_rotation.py --dataset interhuman --save_joint_q
```

**Output:** `data/retargeted_rotation/{dataset}/` — positions + optional joint_q

### Phase 3: Visualization

```bash
# Matplotlib 3D skeleton (default: extracted positions)
python prepare/visualize_positions.py --clip 1000
python prepare/visualize_positions.py --clip 1000 --person 0 --save  # save mp4

# Point at retargeted data instead
python prepare/visualize_positions.py --clip 1000 \
    --data-dir data/retargeted/interhuman

# Newton OpenGL viewer (IK → skeleton playback)
python prepare/visualize_newton.py --clip 1000 --person 0 --fps 30
python prepare/visualize_newton.py --clip 1000 --person 0 \
    --data-dir data/retargeted/interhuman

# Newton viewer (full SMPL-X mesh, loads directly from dataset)
python prepare/visualize_mesh_newton.py --clip 1000 --fps 30
python prepare/visualize_mesh_newton.py --dataset interx --clip G001T000A000R000
```

All visualization scripts accept `--data-dir` to point at any directory containing `(T, 22, 3)` .npy files (extracted, retargeted, or prepare2 output).

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `bone_analysis.py` | Analyze bone length variance across InterHuman clips |

## Assets

- `assets/smpl.xml` — MJCF skeleton with generic SMPL bone lengths (24 bodies, 3 hinges each)
- `assets/humanoid.xml` — Alternative humanoid model (reference only)

## SMPL-X ↔ Newton Joint Mapping

```python
SMPL_TO_NEWTON = {
    0:  0,  # Pelvis → Pelvis
    1:  1,  # L_Hip → L_Hip
    2:  5,  # R_Hip → R_Hip
    3:  9,  # Spine1 → Torso
    4:  2,  # L_Knee → L_Knee
    5:  6,  # R_Knee → R_Knee
    6: 10,  # Spine2 → Spine
    7:  3,  # L_Ankle → L_Ankle
    8:  7,  # R_Ankle → R_Ankle
    9: 11,  # Spine3 → Chest
   10:  4,  # L_Foot → L_Toe
   11:  8,  # R_Foot → R_Toe
   12: 12,  # Neck → Neck
   13: 14,  # L_Collar → L_Thorax
   14: 19,  # R_Collar → R_Thorax
   15: 13,  # Head → Head
   16: 15,  # L_Shoulder → L_Shoulder
   17: 20,  # R_Shoulder → R_Shoulder
   18: 16,  # L_Elbow → L_Elbow
   19: 21,  # R_Elbow → R_Elbow
   20: 17,  # L_Wrist → L_Wrist
   21: 22,  # R_Wrist → R_Wrist
}
```
