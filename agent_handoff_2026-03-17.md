# Agent Handoff (2026-03-17)

## Core Summary

The physics evaluation pipeline was refactored to **replace inverse dynamics with PD forward simulation**. Torques are now computed by running a Newton/MuJoCo forward sim with ground contacts and PD tracking, then extracting the applied feedback torques. This eliminates the "skyhook" (root residual) problem — torques are physically grounded.

---

## What Was Implemented

### 1. PD Forward Torque Computation

- **New function**: `pd_forward_torques(joint_q, betas, dt, device, verbose)` in `prepare4/run_full_analysis.py`
  - Builds Newton model with ground plane
  - Uses `pd_torque_kernel` from `prepare2.pd_utils` at every 480 Hz substep
  - Writes PD torques to `control.joint_f`, accumulates via `accumulate_torque_kernel`, averages per frame
  - Returns `(T, 75)` torques and `(T, 76)` simulated joint coordinates

### 2. Refactored `compute_torques_for_person()`

- **Before**: Called `inverse_dynamics()` (analytical ID, no ground contacts → skyhook residuals)
- **After**: Calls `pd_forward_torques()` (forward sim with ground plane → physically valid torques)

### 3. Updated `batch_torque_distribution.py`

- `process_clip()` now uses `pd_forward_torques` instead of `inverse_dynamics`
- Output format unchanged: `(N, 75)` torques for `torque_distribution.npz`

### 4. Output Pipeline

- `data.npz` still contains `gt_torques_p1`, `gt_torques_p2`, `gen_torques_p1`, `gen_torques_p2` with shape `(N, 75)`
- `compare_torque_distributions.py` and downstream scripts work without modification

### 5. Docstring / Label Updates

- Replaced "skyhook" / "root residual" terminology with "root PD forces" where appropriate
- `root_residuals.png` now titled "Root PD Forces" (same data, clearer semantics)

---

## Key Files

| File | Role |
|------|------|
| `prepare4/run_full_analysis.py` | Main pipeline; `pd_forward_torques()`, `compute_torques_for_person()` |
| `prepare4/batch_torque_distribution.py` | Batch torque stats; uses `pd_forward_torques` via `run_full_analysis` |
| `prepare2/pd_utils.py` | `pd_torque_kernel`, `accumulate_torque_kernel` (GPU PD logic) |
| `prepare4/dynamics.py` | PD gains (`ROOT_POS_KP`, `PD_GAINS`, etc.), `_setup_model_for_id` |
| `prepare4/view_gt_vs_gen.py` | Newton viewer; `presimulate_torque()` for torque-driven videos (still uses built-in PD) |

---

## How to Run

```bash
# Full analysis (6 default clips, with torque video)
conda run -n mimickit --no-capture-output python prepare4/run_full_analysis.py \
  --torque-video --output-dir output/newton_pd_forward --device cuda:0

# Single clip
conda run -n mimickit --no-capture-output python prepare4/run_full_analysis.py \
  --clips "1129 hit" --torque-video --output-dir output/newton_pd_forward

# Batch torque distribution (for compare_torque_distributions.py)
conda run -n mimickit python prepare4/batch_torque_distribution.py \
  --n-clips 200 --output-dir data/torque_stats --device cuda:0
```

---

## Output Structure

```
output/newton_pd_forward/
├── clip_1129_hit/
│   ├── newton_video.mp4
│   ├── newton_video_torque.mp4
│   ├── torque_comparison.png
│   ├── forces.png
│   ├── root_residuals.png      # now "Root PD Forces"
│   ├── interaction_forces.png
│   ├── foot_sole_acceleration.png
│   ├── skeleton_keyframes.png
│   ├── data.npz
│   └── summary.txt
├── summary.txt
└── eval_alignment_audit.txt
```

---

## Technical Notes

- **PD gains**: Same as `presimulate_torque` — `ROOT_POS_KP=2000`, `ROOT_ROT_KP=1000`, body-specific `PD_GAINS` from `dynamics.py`
- **Sim frequency**: 480 Hz physics, 30 fps control (16 substeps/frame)
- **Torque extraction**: PD torques computed at each substep, accumulated, then averaged per frame
- **`view_gt_vs_gen.py` torque videos**: Still use built-in `JointTargetMode.POSITION` + optional ID feedforward; separate from analysis torque computation

---

## Pending / Follow-up

- `compare_torque_distributions.py` expects `torque_distribution.npz` from `batch_torque_distribution.py`; run batch with PD forward torques to regenerate
- Torque-driven videos in `view_gt_vs_gen.py` still use ID+PD or pure PD; analysis torques are now PD-only
- Consider adding a CLI flag to optionally use ID torques for comparison (e.g. `--torque-method id|pd`)

---

## Verification

- Pipeline ran successfully on all 6 default clips
- `data.npz` verified: `gt_torques_p1` shape `(102, 75)`, `|hinge| mean ≈ 11 Nm`, `|root| mean ≈ 243 N` for clip 1129
