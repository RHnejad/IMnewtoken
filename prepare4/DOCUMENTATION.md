# InterMask Physics Analysis Pipeline — Documentation

## Table of Contents
1. [Overview](#overview)
2. [What Was Done](#what-was-done)
3. [The rot6d Problem & IK Fix](#the-rot6d-problem--ik-fix)
4. [Torque Pipeline: How It Works](#torque-pipeline-how-it-works)
5. [Data Pipeline & Verified Facts](#data-pipeline--verified-facts)
6. [Viewer Guide: view_gt_vs_gen.py](#viewer-guide-view_gt_vs_genpy)
7. [Remaining / TODO](#remaining--todo)
8. [Assumptions & Caveats](#assumptions--caveats)

---

## Overview

The goal of this pipeline is to:
1. **Visually verify** that generated InterMask motions are loaded correctly.
2. **Compute joint torques** on GT and Generated motions using Newton Physics SDK.
3. **Compare** torque distributions between GT and Generated to assess physical plausibility.

---

## What Was Done

### Phase 1: Data Verification & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| **Wrong FPS (critical)** | All scripts used `FPS=20`, but InterMask data is at 30fps and GT PKL is at 59.94fps (~60fps). | Changed FPS to 30 everywhere. GT PKL downsampled `[::2]` (60→30fps). |
| **GT positions overlapping** | `process_motion_np()` centers each person independently to origin, destroying relative positioning. | Load raw NPY positions in Z-up world frame instead. |
| **Gen SMPL bodies garbled** | InterMask's 262-dim rot6d (cols 132-258) are NOT SMPL-X joint rotations. | Replaced with IK from `positions_zup` (see [below](#the-rot6d-problem--ik-fix)). |
| **`log_points` color crash** | Newton `log_points` requires `wp.array` for colors. | Use `wp.full(n, wp.vec3(*color), ...)` for color arrays. |
| **NumPy 2.x compat** | `numpy.core.umath_tests` removed in NumPy 2.x. | Replaced with `np.einsum`. |
| **Matplotlib compat** | `ax.lines = []` is read-only in newer matplotlib. | Replaced with `ax.cla()`. |

### Phase 2: Torque Pipeline Fixes (current session)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| **Generated torques used wrong rotations** | `process_clip()` called `rotation_retarget()` on InterMask's bad rot6d-derived rotations for generated data. | Changed to `ik_retarget(positions_zup)` — IK from the actual 22-joint positions. |
| **IK angle windup → massive torques (~15,000 Nm)** | Newton IK solver produces hinge angles with multi-revolution values (e.g., -9.7 rad ≈ -556°). Combined with MJCF joint limits (e.g., ±30°), the MuJoCo zero-torque step generated enormous constraint forces. | Two fixes: (1) Normalize hinge angles by shifting each DOF's trajectory by constant multiples of 2π so the mean angle is closest to 0. (2) Disable joint limits in `_setup_model_for_id()` for inverse dynamics — so `qdd_free` reflects only gravity + Coriolis, not artificial limit enforcement. |
| **Old wrong torque results persisted** | Previous runs used FPS=20 and/or bad rotations. ~1.6GB of stale results in `data/compute_torques/`, `data/skyhook_metrics/`, old `data/torque_stats/`. | Deleted all old results. |

### Files in prepare4/

| File | Purpose |
|------|---------|
| `view_gt_vs_gen.py` | Newton GUI viewer: GT vs Generated motions side-by-side. |
| `gen_intermask_mp4.py` | Generate InterMask-style matplotlib MP4s for visual comparison. |
| `retarget.py` | IK retarget (position→Newton joint_q) and rotation retarget (GT SMPL→Newton). |
| `batch_torque_distribution.py` | Batch torque computation across sampled clips. |
| `dynamics.py` | Frame-level inverse dynamics (SavGol derivatives + zero-torque step). |
| `gen_xml.py` | MJCF skeleton generation from SMPL-X betas. |
| `compare_torque_distributions.py` | GT vs Generated torque comparison (plots + stats). |
| `DOCUMENTATION.md` | This file. |
| `SKELETON_AND_XML_REFERENCE.md` | Newton skeleton / MJCF reference. |

### Modified External Files

| File | Change |
|------|--------|
| `generate_and_save.py` | Added `positions_zup` to output PKL (the actual 22-joint 3D positions from InterMask). |

---

## The rot6d Problem & IK Fix

### Problem
InterMask's 262-dim output format encodes:
- Columns 0–3: root velocity (XZ) + height (Y)
- Columns 4–67: 22 joints × 3D positions (relative)
- Columns 67–131: 22 joints × 3D velocities
- **Columns 132–257: 21 joints × 6D rotation representations**
- Columns 258–261: foot contacts (4 values)

The 6D rotations are in InterMask's **own convention** — NOT standard SMPL-X joint rotations. Naively converting `rot6d → rotation_matrix → axis-angle` and using as SMPL-X `pose_body` produces **garbled, exploding bodies**.

### Fix: IK from Positions
Instead of using the broken rot6d, we solve for SMPL-X parameters using **Inverse Kinematics** from the 22-joint world positions (`positions_zup`).

**Method:** `prepare4/retarget.py::ik_retarget()`
- Uses Newton's `IKSolver` with `IKObjectivePosition` (position-based IK)
- Levenberg-Marquardt optimizer with autodiff Jacobian
- **Sequential warm-starting**: 50 iterations for frame 0, 5 iterations for subsequent frames (each initialized from the previous frame's solution)
- Result: smooth, temporally coherent SMPL-X body motions

**Quality:**
- Person 1 MPJPE: **29.5mm** (very good)
- Person 2 MPJPE: **43.7mm** (good)


### Why Not Fix the rot6d Mapping?
InterMask's 6D rotations are computed during training from a specific kinematic chain ordering and convention. Recovering the correct mapping would require:
1. Tracing through InterMask's data preprocessing (`process_motion_np`)
2. Understanding their specific rotation decomposition
3. Matching their kinematic chain to SMPL-X's

The IK approach bypasses all of this — positions are unambiguous.

---

## Torque Pipeline: How It Works

### Overview

The torque pipeline computes **inverse dynamics** torques for both GT and generated motions. The key difference is how joint angles are obtained:

```
GT motion:    PKL (SMPL rotations) ──→ rotation_retarget() ──→ joint_q (T, 76)
Generated:    PKL (positions_zup)  ──→ ik_retarget()       ──→ joint_q (T, 76)
                                                                    │
                                                          compute_derivatives()
                                                           (SavGol, window=11)
                                                                    │
                                                            qd (T, 75), qdd (T, 75)
                                                                    │
                                                          inverse_dynamics()
                                                   τ = M(q)(q̈_desired − q̈_free)
                                                                    │
                                                            torques (T, 75)
```

### Key Components

1. **`rotation_retarget()`** (for GT): Direct SMPL-X axis-angle → Newton hinge Euler angles. GT PKL has correct SMPL rotations. Downsampled `[::2]` from 60fps→30fps.

2. **`ik_retarget()`** (for generated): Newton IK solver from 22-joint 3D positions. Sequential mode: frame-by-frame solving, warm-starting from previous frame. After IK, each DOF's trajectory is shifted by a constant multiple of 2π to normalize angles near zero (prevents windup).

3. **`compute_derivatives()`**: Savitzky-Golay filter (window=11, polyorder=5) computes smoothed 1st and 2nd derivatives. Root quaternion is converted to rotation vector, unwrapped, then differentiated.

4. **`inverse_dynamics()`**: For each frame:
   - Compute mass matrix M(q)
   - Run zero-torque MuJoCo step to get q̈_free (gravity + Coriolis only — joint limits disabled)
   - τ = M(q)(q̈_desired − q̈_free)

### DOF Layout

Joint torques are (T, 75):
- DOFs 0-2: Root translation forces (N) — virtual, not physically meaningful
- DOFs 3-5: Root rotation torques (Nm) — virtual, not physically meaningful
- DOFs 6-74: Hinge joint torques (Nm) — 23 bodies × 3 hinge DOFs each

For analysis, **only hinge torques (DOFs 6:75)** should be used. Root DOFs are virtual forces that would be needed to track the prescribed trajectory and are not real joint torques.

### Verified Results (200 GT clips)

| Body Group | GT abs_mean (Nm) | Interpretation |
|------------|------------------|----------------|
| L/R Leg | ~10 Nm | Walking/standing forces |
| Spine/Torso | ~15 Nm | Core stabilization |
| L/R Arm | ~3-4 Nm | Arm movements |

These are physically realistic for human motion.

### Running the Pipeline

```bash
# GT batch (fast, ~40 min for 200 clips)
conda run -n mimickit python prepare4/batch_torque_distribution.py \
    --source gt --n-clips 200 --output-dir data/torque_stats --seed 42

# Generated batch (slow, ~12 hours for 200 clips due to sequential IK)
conda run -n mimickit python prepare4/batch_torque_distribution.py \
    --source generated --data-dir data/generated/interhuman \
    --n-clips 200 --output-dir data/torque_stats_generated --seed 42

# Compare GT vs Generated (after both complete)
conda run -n mimickit python prepare4/compare_torque_distributions.py
```

---

## Data Pipeline & Verified Facts

### Coordinate Systems

| System | Convention | Used By |
|--------|-----------|---------|
| Z-up (world) | X=right, Y=forward, Z=up | Newton, raw positions, `positions_zup` |
| Y-up (InterMask) | X=right, Y=up, Z=backward | InterMask's 262-dim format, matplotlib vis |

**Transforms:**
```python
trans_matrix = [[1,0,0], [0,0,1], [0,-1,0]]    # Z-up → Y-up
INV_TRANS    = [[1,0,0], [0,0,-1], [0,1,0]]     # Y-up → Z-up
```

Both are pure rotations (det = +1.0). No scaling.

### process_motion_np — Verified Pure Rigid Transform
`data/utils.py::process_motion_np()` applies:
1. `trans_matrix` rotation (Z-up → Y-up)
2. Floor correction (lift lowest foot to ground)
3. XZ centering (move to origin)
4. Face-Z+ rotation (orient character to face forward)

**Verified facts:**
- **Bone length error**: < 0.0000004m (< 0.4 microns)
- **Acceleration difference**: < 0.001 m/s²
- **Torques**: INVARIANT to coordinate frame (rigid transforms don't change dynamics)

This means computing torques on `positions_zup` is **identical** to computing on the 262-dim processed positions that InterMask reports metrics on.

### Frame Rates

| Source | FPS | Notes |
|--------|-----|-------|
| GT PKL files | 59.94 (~60) | `mocap_framerate` field |
| InterMask NPY (processed) | 30 | Downsampled from 60fps |
| Generated output | 30 | Matches training data |

GT PKL must be downsampled `[::2]` to align with InterMask/generated data.

### Data Directories

| Directory | Contents | Source |
|-----------|----------|--------|
| `data/generated/interhuman/` | 1098 PKLs | `generate_and_save.py` (transformer text-conditioned) |
| `data/reconstructed_dataset/interhuman/` | VQ reconstruction | `save_generated_as_dataset.py --use_trans False` |
| `data/generated_dataset/interhuman/` | ⚠️ NOT YET CREATED | `save_generated_as_dataset.py --use_trans True` |
| `data/InterHuman/motions_processed/` | GT processed NPY | InterHuman dataset |
| `data/InterHuman/annots/` | Text descriptions | InterHuman dataset |

### Generated PKL Format (from `generate_and_save.py`)
```python
{
    'motion1': np.array (T, 262),  # person 1, InterMask 262-dim
    'motion2': np.array (T, 262),  # person 2, InterMask 262-dim
    'text': str,                   # text prompt
    'lengths': [T, T],             # frame counts
    'positions_zup': [             # ← ADDED: actual 3D positions
        np.array (T, 22, 3),      # person 1 in Z-up world frame
        np.array (T, 22, 3),      # person 2 in Z-up world frame
    ]
}
```

---

## Viewer Guide: view_gt_vs_gen.py

### What Each Panel/Group Shows

The viewer displays up to 5 groups of bodies stacked along the Y-axis (separated by `--y-offset`, default 2.0m). Each group shows a different representation of the same motion:

```
Y = -2.0   ┌─────────────────────────────────┐
            │  GT STICK-FIGURES               │  ← Raw GT positions from NPY
            │  Blue (person 0)                │     22-joint stick figures
            │  Purple (person 1)              │     Source: InterHuman/motions_processed/
            └─────────────────────────────────┘

Y =  0.0   ┌─────────────────────────────────┐
            │  GT SMPL BODIES                 │  ← GT SMPL-X rotations retargeted
            │  Skin-colored 3D body meshes    │     to Newton skeleton = correct body
            │  (the ground-truth reference)   │     Source: InterHuman/motions/*.pkl
            └─────────────────────────────────┘

Y = +2.0   ┌─────────────────────────────────┐
            │  GENERATED SMPL BODIES (via IK) │  ← InterMask positions → IK solve
            │  Skin-colored 3D body meshes    │     = SMPL body from generated output
            │  (~30mm MPJPE from IK)          │     Source: data/generated/interhuman/*.pkl
            └─────────────────────────────────┘     (positions_zup → ik_retarget)

Y = +4.0   ┌─────────────────────────────────┐
            │  GEN STICK-FIGURES              │  ← Raw generated positions
            │  Green (person 0)               │     22-joint stick figures
            │  Orange (person 1)              │     Source: positions_zup from generated PKL
            └─────────────────────────────────┘     (THE actual InterMask output)

Y = +6.0   ┌─────────────────────────────────┐  (only with --rec-data)
            │  REC STICK-FIGURES              │  ← VQ-VAE reconstructed positions
            │  Yellow (person 0)              │     (near-GT quality, not generation)
            │  Cyan (person 1)               │     Source: reconstructed_dataset/
            └─────────────────────────────────┘
```

**How to read it:**
- Compare the Y=0 (GT body) with Y=+2 (Gen body) to see if the generated motion looks natural as a human body.
- Compare the GT bodies (Y=0) with the GT stick-figures (Y=-2) — they should match perfectly (different representation of same data).
- The Gen stick-figures (Y=+4) are the ACTUAL InterMask output — just 22 3D points. The Gen bodies (Y=+2) are those same positions fitted to a SMPL body via IK (introduces ~30mm error).
- The Rec group (Y=+6, optional) shows VQ-VAE reconstruction quality — should be very close to GT.

### CLI Arguments

```bash
# Basic usage (interactive viewer)
conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py --clip 3678

# Save MP4 with side camera view
conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py \
    --clip 3678 --save-mp4 prepare4/vis_compare/3678_newton.mp4

# Include reconstructed_dataset comparison
conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py \
    --clip 3678 --rec-data

# Custom camera
conda run -n mimickit --no-capture-output python prepare4/view_gt_vs_gen.py \
    --clip 3678 --cam-preset quarter --cam-dist 15 --cam-height 5
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--clip` | required | Clip ID (e.g., `3678`) |
| `--person` | both | Show only person 0 or 1 |
| `--fps` | 30 | Playback FPS |
| `--gt-only` | off | Show only GT |
| `--gen-only` | off | Show only Generated |
| `--y-offset` | 2.0 | Y spacing between groups (meters) |
| `--no-fix-ground` | off | Disable auto floor correction |
| `--no-smpl-body` | off | Skip SMPL body rendering |
| `--rec-data` | off | Show reconstructed_dataset positions |
| `--save-mp4 PATH` | off | Record MP4 (auto-enables headless) |
| `--mp4-width` | 1280 | MP4 frame width |
| `--mp4-height` | 720 | MP4 frame height |
| `--cam-preset` | side | `front`, `side`, `top`, `quarter` |
| `--cam-yaw` | preset | Override camera yaw (degrees) |
| `--cam-pitch` | preset | Override camera pitch (degrees) |
| `--cam-dist` | preset | Override camera distance |
| `--cam-height` | preset | Override camera Z height |

### Camera Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| `side` | Camera at +X, looking across Y-axis | Seeing all groups spread out horizontally |
| `front` | Camera at -Y, looking forward | Close-up of one group |
| `top` | Overhead elevated view | Overall layout check |
| `quarter` | Diagonal elevated | Good balance of depth and spread |

### MP4 Text Overlay
When recording MP4 (`--save-mp4`), each frame gets an overlay showing:
- Clip ID and frame counter
- Motion text description (if available)
- Color-coded legend of each group with its Y position

---

## Remaining / TODO

### In Progress

1. **Generated batch torque computation** — Running as background process (PID tracked via `ps aux | grep batch_torque`). 200 clips, ~3.5 min/clip = ~12 hours total. Saves intermediate results every 50 clips to `data/torque_stats_generated/`. When complete, outputs `torque_distribution.npz`, `torque_stats.json`, and plots.

### Not Yet Done

2. **Run GT vs Generated comparison** — After the generated batch completes:
   ```bash
   conda run -n mimickit python prepare4/compare_torque_distributions.py
   ```
   This loads both `data/torque_stats/torque_distribution.npz` (GT) and `data/torque_stats_generated/torque_distribution.npz` (Generated), computes side-by-side statistics, and generates comparison plots to `data/torque_comparison/`.

3. **Text-conditioned dataset generation** — For InterMask's official evaluation pipeline:
   ```bash
   conda run -n intermask python save_generated_as_dataset.py \
       --dataset_name interhuman --name trans_default \
       --use_trans True --save_processed \
       --output_dir data/generated_dataset/interhuman
   ```
   Creates a full InterHuman-format dataset from transformer-generated motions (not just VQ reconstruction).

### Completed

- ✅ GT batch torques (200 clips, 53,024 frames → `data/torque_stats/`)
- ✅ Fixed generated torque pipeline to use positions via IK (not bad rot6d)
- ✅ Fixed IK angle normalization (remove multi-revolution windup)
- ✅ Disabled joint limits in inverse dynamics (prevents constraint artifacts)
- ✅ Cleaned all old wrong torque results (~1.6GB deleted)
- ✅ Cleaned prepare4/ of test/debug scripts
- ✅ FPS corrected from 20→30 everywhere
- ✅ Added `positions_zup` to generated PKLs
- ✅ Re-generated 1098 clips with `positions_zup`

---

## Assumptions & Caveats

### CRITICAL — Read Before Interpreting Results

1. **Torques are computed from IK-solved joint angles, not directly from positions.**
   Generated positions (`positions_zup`) → IK solve → Newton joint_q → SavGol derivatives → inverse dynamics → torques. The IK step introduces ~30mm positional error (MPJPE). This means the joint angles are an approximation of what would produce the generated positions. The torques reflect these approximate angles, not the raw positions exactly. GT torques use the original SMPL rotations directly (no IK).

2. **IK angle normalization shifts angles by multiples of 2π.**
   After IK, each DOF's trajectory is shifted by `round(mean/2π) × 2π` to keep angles near zero. This doesn't change positions (FK is rotationally periodic) but affects the specific angle convention. The normalization preserves temporal smoothness because it shifts the entire trajectory by a constant.

3. **Joint limits are disabled during inverse dynamics.**
   The MJCF model has joint limits (e.g., hip range ±30-90°). These are DISABLED in `_setup_model_for_id()` for the zero-torque step so that `qdd_free` reflects only gravity + Coriolis forces. Without this, IK angles that exceed MJCF limits (even after normalization) would cause spurious constraint forces. This applies to both GT and generated data identically.

4. **FPS must be 30.** Any script computing derivatives is extremely sensitive to FPS. Using 20 instead of 30 changes accelerations by ~(30/20)² = 2.25× and torques proportionally. All old calculations with FPS=20 were **wrong** and have been deleted.

### Important

5. **GT PKL downsample**: GT PKL files are at ~60fps. Always downsample `[::2]` when comparing with InterMask/generated data at 30fps. This is done automatically in `process_clip()`.

6. **process_motion_np centers independently**: If you use `process_motion_np()` on two people separately, they will both be centered to the origin and lose their relative positioning. For interaction analysis, always use raw positions in world frame.

7. **positions_zup was added post-hoc**: The original `generate_and_save.py` did not save world positions. The `positions_zup` field was added by us. The 1098 clips in `data/generated/interhuman/` have this field. If you regenerate, make sure the modified `generate_and_save.py` is used.

8. **Reconstructed ≠ Generated**: `data/reconstructed_dataset/` = VQ-VAE reconstructions (near-GT quality, not text-conditioned). `data/generated/` = transformer text-conditioned outputs (actual model inference output). For evaluation, use transformer-generated data.

9. **Generated torque batch is very slow**: Sequential IK solving takes ~3.5 min per clip (236 frames × 2 persons × frame-by-frame IK). This is inherent to the sequential warm-starting approach needed for temporal consistency. The batch operation supports `--resume` for recovery from interruptions.

10. **Body mass assumed 75kg**: All inverse dynamics use `total_mass=75.0` kg with De Leva 1996 segment mass fractions. Actual person masses are not available from InterHuman data. This affects absolute torque magnitudes but not relative GT-vs-Generated comparisons (both use the same mass).

11. **Root DOFs (0:5) are virtual forces**: The first 6 DOFs in the torque output (3 translational + 3 rotational) are the forces/torques needed to track the prescribed root trajectory. They are NOT physical joint torques. Only hinge DOFs 6:75 represent anatomical joint torques.

### Minor

12. **Newton IK warm-starting** can drift if positions are very noisy. Sequential mode initializes each frame from the previous frame's solution (50 iters frame 0, 5 iters subsequent).

13. **Display required for MP4**: Newton's `--save-mp4` requires a display (pyglet/OpenGL). Must run on the physical machine or with X11 forwarding.

14. **InterMask's eval metrics use reconstructed_dataset by default**: `save_generated_as_dataset.py --use_trans False` creates VQ reconstructions. Use `--use_trans True` for actual text-conditioned generation.
