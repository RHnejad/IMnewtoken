# Physics-Based Evaluation of Two-Person Interaction Motions
### Supervisor Meeting — 2026-03-25

---

## 1. Context & Motivation

**InterMask / InterGen** generate text-conditioned two-person motion sequences (SMPL-X format).
Existing metrics (FID, MPJPE, diversity) are **purely geometric** — they say nothing about
whether a motion is *physically plausible*: can a real human actually execute it?

**Goal:** Add a physics-based plausibility metric that can:
- Compare GT vs generated motions
- Detect physically impossible interactions (e.g., characters passing through each other, anti-gravity jumps)
- Provide a quantitative "Dynamic Gap" score between real and generated datasets

---

## 2. What Has Been Built

### 2.1 SMPL-X → Newton Physics Retargeting (prepare2)

- Converts SMPL-X motion parameters to a physics simulator (Newton/MuJoCo-XML)
- Per-subject body shapes: SMPL-X betas → subject-specific skeleton proportions
- **Accuracy achieved: 0.00 cm MPJPE** (joint positions match exactly after retargeting)

### 2.2 Inverse Dynamics Torque Computation (prepare2/prepare4)

- From the retargeted joint angles, compute the joint torques needed to reproduce the motion
- Uses B-spline differentiation (no noise from finite differences)
- Produces per-frame torques `τ ∈ ℝ^75` for each person

### 2.3 Residual Wrench Metric (`prepare2/residual_wrench_eval.py`)

The **key insight**: the first 6 DOFs of inverse dynamics are *virtual root forces* —
a "skyhook" — that must be injected to maintain the character's trajectory.

For physically valid motion (real mocap, realistic simulation): these should be **near zero**.
For generated motions with physics violations: they can be **very large**.

**Metrics computed:**
| Metric | Description |
|--------|-------------|
| `F_sky_mean` | Mean root translation force needed (Newtons) |
| `τ_sky_mean` | Mean root rotation torque needed (Nm) |
| `P_active` | Mean joint actuation power (W) — excludes root |
| `ΔF_sky` | `F_sky_gen − F_sky_gt` per clip pair = "Dynamic Gap" |

Threshold: skyhook force > 3× body weight (~2208 N) = physically impossible frame.

### 2.4 PD Forward Simulation (prepare4)

Alternative to inverse dynamics:
- Run physics sim with PD controllers tracking the reference motion
- Extract PD torques actually applied at each substep
- Eliminates pure ID artefacts by grounding torques in contact forces
- 480 Hz physics, 30 fps control (16 substeps/frame)

### 2.5 Differentiable Trajectory Optimization (`prepare5/optimize_tracking.py`)

For higher-quality physics tracking:
- Parameterize PD target residuals `Δq` (learnable)
- Roll out physics for a window of W frames using Newton's differentiable SolverFeatherstone
- Backprop through physics → Adam update on `Δq`
- Goal: minimize `||sim_pos(t) − ref_pos(t)||² + λ||Δq||²`

---

## 3. Current Results

### 3.1 Timing benchmark (clip 1129, 30fps, person 1)

| Config | Fwd/window | Bwd/window | Total/window | Est. 20 epochs |
|--------|------------|------------|--------------|----------------|
| 16 substeps (old) | — | — | ~4 min | **~1400 min** |
| **4 substeps (new)** | 2.8s | 16.9s | **19.8s** | **~138 min** |

Loss = 0.061, grad_norm = 0.57 — gradients are flowing correctly.

### 3.2 PHC-style tracker on clip 1129 (prepare5)

| Source | MPJPE |
|--------|-------|
| GT motion | ~150 mm |
| Generated motion | ~193 mm |

Both are currently high — indicates the tracker has not yet converged (20 epochs insufficient).
The gap (GT vs Gen) is the signal of interest, not the absolute value.

---

## 4. Problems & Open Issues

### Problem 1: Differentiable optimization is slow

**Cause**: Full differentiable physics sim requires `O(W × substeps)` memory for backprop tape.
- 16 substeps → 81 states per window → backward pass dominates (6× slower than forward)
- Even at 4 substeps: **19.8s/window, 138 min/epoch×20** — prohibitive for the full dataset

**Current workaround**: Reduce substeps (4 instead of 16), reduce window size (W=5 frames).
**Risk**: Lower physics fidelity, contacts may not be captured accurately.

**Possible solutions**:
- Truncated BPTT (don't backprop through the full window)
- Horizon schedule (start small, grow window over training)
- Pre-train with kinematic loss only, then refine with physics
- Use IK-initialized warm start to reduce needed iterations

### Problem 2: Sim-to-sim transfer / tracking accuracy

The PHC tracker achieves ~150 mm MPJPE even on GT mocap — which is too high.
This suggests the physics model (joint limits, gains, mass distribution) is not well-calibrated
for the SMPL-X body model.

**Root cause candidates:**
- SMPL-X has pose-dependent blend shapes not captured in the rigid-body approximation
- Foot contact model (sphere feet) introduces sliding artefacts
- PD gains not tuned for the 75-DOF SMPL model
- Root drift when `root_mode = free` (no root anchor)

**Current state**: investigating gain tuning (PHC_BODY_GAINS in prepare5/phc_config.py).

### Problem 3: Residual wrench on GT is non-zero

Even real mocap (GT) requires non-trivial skyhook forces (~100–300 N range).
**This is expected** (mocap has noise, foot sliding, imperfect retargeting), but makes
absolute thresholding unreliable.

**Solution**: Use the **relative metric** `ΔF_sky = F_sky_gen − F_sky_gt` per clip pair,
normalised to body weight. This removes systematic bias from the retargeting pipeline.

### Problem 4: Dataset pipeline latency

Full batch evaluation (200+ clips, both persons, GT + generated) takes hours even with multi-GPU.
A quick proxy is needed for development iterations.

**Current approach**: 50-clip stratified subset, multi-GPU (both 3090s), early stopping.

---

## 5. Proposed Evaluation Protocol (Final Metric)

```
For each test clip c:
  1. Retarget GT and generated to Newton (0 cm MPJPE error)
  2. Run inverse dynamics → get joint torques τ(t)
  3. Extract root residuals F_sky(t) ∈ ℝ^3
  4. ΔF_sky(c) = mean_t ||F_sky_gen(t)|| − mean_t ||F_sky_gt(t)||

Report:
  • mean_c ΔF_sky  (Dynamic Gap, Newtons)
  • % clips where ΔF_sky > 0 (generated worse than GT)
  • Distribution plot (histogram GT vs gen)
```

This is **fast** (no optimization needed) and **interpretable** (physical units).

---

## 6. Discussion Points for Meeting

1. **Is the residual wrench metric sufficient, or do we need actual simulation?**
   - Wrench: fast, no optimization needed, directly measures Newton's law violation
   - Simulation: stronger claim ("can a physics controller reproduce this?"), but slow

2. **Should we compare against a physics-based baseline** (e.g., PHC-retargeted GT)?

3. **Scope of the paper contribution**:
   - New metric only?
   - New metric + physics-guided post-processing to improve generated motions?
   - Evaluation on multiple models (InterMask, InterGen, ReInteract…)?

4. **Compute budget**: Full 138min/run × N clips makes systematic ablation hard.
   Is there access to more compute (cluster, more GPUs)?

---

## 7. Next Steps

| Priority | Task | Status |
|----------|------|--------|
| HIGH | Run residual_wrench_eval.py on 50 test clips | ready to run |
| HIGH | Tune PHC tracker gains → get GT MPJPE < 50mm | in progress |
| MED | Implement truncated BPTT for faster optimization | planned |
| MED | Batch residual wrench comparison (GT vs generated) | pipeline ready |
| LOW | Optimize across full test set | blocked by speed |

---

*All code in `prepare2/`, `prepare4/`, `prepare5/` in the InterMask repo.*
*Physics simulator: [Newton](https://github.com/newton-physics/newton) (Warp-based, differentiable)*
*Body model: SMPL-X (55 joints, 10 betas), retargeted to 24-body rigid articulation*
