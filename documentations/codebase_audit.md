# InterMask Codebase Audit — Updated 2026-04-01

Status legend: FIXED, UNFIXED, NO LONGER RELEVANT, NEW

---

## prepare/ — Motion Extraction & Generic Retargeting

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | CRITICAL | Identity quaternion `[1,0,0,0]` (wxyz) — Newton expects `[0,0,0,1]` (xyzw) | **FIXED** | retarget_newton.py:120 |
| 2 | LOW | Hardcoded data paths | UNFIXED | Low priority |
| 3 | LOW | Silent skip on missing dirs | UNFIXED | Low priority |
| 4 | LOW | `int \| None` requires Python 3.10+ | UNFIXED | Low priority |
| 5 | LOW | Quaternion convention comments inconsistent | **FIXED** | Verified via Newton docs: xyzw confirmed |

---

## prepare2/ — Full Physics Simulation & Inverse Dynamics

| # | Severity | Bug | Status | Notes |
|---|----------|-----|--------|-------|
| 1 | CRITICAL | Smoothing applied AFTER differentiation | **FIXED** | FD path now smooths joint_q before finite differences |
| 2 | MINOR | Redundant `model.collide()` | ACCEPTABLE | Harmless |
| 3 | MINOR | Unnecessary state pre-zeroing | **FIXED** (previously) | |
| 4 | MODERATE | SolverFeatherstone contacts insufficient for two-person | NO LONGER RELEVANT | prepare4 uses SolverMuJoCo for paired sim |
| 5 | MODERATE | No PD stabilization in optimize_interaction | NO LONGER RELEVANT | optimize_interaction is a failed approach (Featherstone-based) |
| 6 | MODERATE | PD computed once/frame, applied to 24 substeps | UNFIXED | Only affects prepare2/batch_sim_solo.py; prepare4 pipeline uses per-substep PD |
| 7 | MINOR | GPU→CPU transfer every substep | **FIXED** (previously) | |
| 8 | MINOR | Gradient clipping allocations | ACCEPTABLE | |
| 9 | MINOR | Delta array converted every render frame | ACCEPTABLE | |
| 10 | CRITICAL | FPS/downsample mismatch | **FIXED** | Torque files now save `_meta.json` with fps/downsample; optimize_interaction validates on load |
| 11 | CRITICAL | Armature inconsistency (inverse dynamics vs optimization) | **FIXED** | optimize_interaction.py imports ARMATURE_HINGE/ROOT from pd_utils |
| 12 | MODERATE | Large dt in zero-torque step | UNFIXED | Only affects `--diff-method fd` path; default spline method avoids this |
| 13 | MINOR | Worker symlink race condition | UNFIXED | Low priority, only affects multi-GPU batch |

---

## prepare3/ — RL-Based Motion Tracking (PPO)

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | MEDIUM | Reward lacks physical plausibility check | NO LONGER RELEVANT | prepare3 superseded by prepare5 (PHC tracker) and prepare6 (RL tracker) |
| 2-6 | LOW-MEDIUM | Various config/logic issues | NO LONGER RELEVANT | prepare3 is an early prototype, not used in current pipeline |

---

## prepare4/ — Paired-vs-Solo Torque Analysis (MOST IMPORTANT)

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | CRITICAL | Identity quaternion wrong format in IK init | **FIXED** | Changed to `[0,0,0,1]` (xyzw). IK solver likely still converged but from better initial guess now |
| 2 | CRITICAL | IK angle wrapping (global mean shift fails bimodal) | **FIXED** | Uses `np.unwrap()` + centering. Eliminates spurious torque spikes |
| 3 | CRITICAL | Multi-person DOF split unvalidated | **FIXED** | Added assertions: `n_dof == 2*DOFS_PER_PERSON`, `n_coords == 2*COORDS_PER_PERSON` |
| 4 | CRITICAL | N3LV metrics depend on #3 | **FIXED** | #3 confirmed correct (150 DOF, 152 coords); assertions guard against future changes |
| 5 | MEDIUM | Resume serialization loses numpy arrays | UNFIXED | Only affects interrupted batch runs |
| 6 | MEDIUM | XML from NPY doesn't validate skeleton plausibility | UNFIXED | Low priority |
| 7 | MEDIUM | Torque delta vs BPS semantics unclear | UNFIXED | Documentation needed |
| 8 | MEDIUM | ID uses single zero-torque step | UNFIXED | Could improve accuracy of inverse dynamics |
| 9 | MEDIUM | rotation_retarget claims 0.000 MPJPE with no validation | UNFIXED | Should add inline assert |
| 10 | LOW | Hardcoded file patterns in batch eval | UNFIXED | |
| 11 | LOW | SavGol window clamp for short trajectories | UNFIXED | |

### NEW issues found

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| N1 | **NEW** | Joint limits in default XML too tight (knee ±30°, elbow ±30°) — creates constraint forces that inflate torque estimates | **MITIGATED** | prepare4/dynamics.py already widens to ±1e6 for ID. PHC style (`--joint-style phc`) uses ±180°. But default retargeting path still has tight limits |
| N2 | **NEW** | `--joint-style phc` now available on all prepare4 scripts | **DONE** | retarget.py, view_gt_vs_gen.py, dynamics.py, run_full_analysis.py |

---

## prepare5/ — PHC-Style PD Tracker

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | HIGH | Termination flag set once, never cleared | UNFIXED | Still a real bug but low impact on aggregate metrics |
| 2 | MEDIUM | Reward logs position-only | UNFIXED | Documentation issue |
| 3 | MEDIUM | Settle phase forces skyhook unconditionally | UNFIXED | |
| 4 | MEDIUM | Dual PD mode fragile (no guard) | UNFIXED | |
| 5 | MEDIUM | OLD_BODY_GAINS still defined | **PARTIALLY FIXED** | Constants now imported from prepare_utils; but OLD_BODY_GAINS still exists for backward compat |
| 6 | LOW | FK recomputed every frame | UNFIXED | Performance, not correctness |
| 7 | LOW | No GPU memory cleanup | UNFIXED | |
| 8 | LOW | SETTLE_FRAMES hardcoded | UNFIXED | |
| 9 | INFO | optimize_tracking.py dead code | UNFIXED | Should archive |

### NEW issues found/fixed

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| N3 | **NEW/FIXED** | `_setup_model_properties()` was missing `joint_target_ke/kd` zeroing — XML stiffness could leak | **FIXED** | Added `model.joint_target_ke.fill_(0.0)` and `model.joint_target_kd.fill_(0.0)` |
| N4 | **NEW/DONE** | `--joint-style phc` support added to phc_tracker.py, visualize_newton_tracking.py, run_phc_tracker.py | **DONE** | |

---

## prepare6/ — PPO RL Per-Clip Tracker

### Design Flaw (unchanged)
**The smoothness confound invalidates this approach as a physics plausibility metric.** Generated motions track BETTER than GT. This is fundamental — no code fix resolves it.

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | CRITICAL | Observations NOT normalized | UNFIXED | Training instability |
| 2 | HIGH | Advantage normalization wrong (global vs per-batch) | UNFIXED | |
| 3 | HIGH | GAE bootstrap on done uses old value | UNFIXED | |
| 4 | HIGH | Auto-reset loses episode boundaries | UNFIXED | |
| 5 | HIGH | Gimbal lock in yaw normalization | UNFIXED | |
| 6 | HIGH | Multi-env stride unvalidated | UNFIXED | |
| 7 | MEDIUM | RSI_PROB=0.9 too aggressive | UNFIXED | |
| 8 | MEDIUM | Action scaling exceeds joint limits | UNFIXED | |
| 9 | MEDIUM | Reference velocities noisy (finite diff at 30fps) | UNFIXED | |
| 10 | LOW | Early stopping double-smoothed | UNFIXED | |

**Recommendation**: prepare6 has 10 unfixed bugs + the smoothness confound. Given that prepare4 (paired-vs-solo torque) is the winning metric and prepare5 (PHC tracker) is the more principled tracker, prepare6 should be considered **deprecated** unless the smoothness confound is resolved.

---

## newton_vqvae/ — Physics-Informed VQ-VAE

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | MEDIUM | Imports R_ROT from old prepare2/gen_smpl_xml | **FIXED** | Now imports from prepare_utils.constants |
| 2 | MEDIUM | Batch transpose `.T` on (B,3,3) | UNFIXED | |
| 3 | MEDIUM | No gradient checkpointing | UNFIXED | |
| 4 | LOW | Loss weights hardcoded | UNFIXED | |
| 5 | INFO | Never integration-tested | UNFIXED | |

---

## eval_pipeline/

| # | Severity | Issue | Status | Notes |
|---|----------|-------|--------|-------|
| 1 | MEDIUM | No framerate validation | **PARTIALLY FIXED** | Torque files now include `_meta.json` with fps. But retarget step still doesn't validate |

---

## Cross-Cutting Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | Quaternion convention (xyzw vs wxyz) | **FIXED** | All identity quats corrected; confirmed via Newton docs |
| 2 | Retargeting code duplicated | **PARTIALLY FIXED** | Constants consolidated to prepare_utils; retarget functions still separate (intentional divergence between folders) |
| 3 | PD utilities duplicated | **PARTIALLY FIXED** | Constants shared; gain definitions still per-folder (intentional: PHC gains differ from prepare2 gains) |
| 4 | Data paths scattered | UNFIXED | |
| 5 | Dead code (optimize_tracking, optimize_interaction) | UNFIXED | Should archive |
| 6 | No integration tests | UNFIXED | |

### NEW cross-cutting improvements

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| N5 | **NEW/DONE** | `prepare_utils/` package created as single source of truth | **DONE** | constants.py, gen_xml.py, provenance.py, smpl_robot_bridge.py |
| N6 | **NEW/DONE** | XML generation consolidated to `prepare_utils/gen_xml.py` | **DONE** | prepare4/gen_xml.py is now re-export shim; xml_cache migrated |
| N7 | **NEW/DONE** | PHC-style XML generation via `SMPL_Robot` | **DONE** | `joint_style="phc"` uses real PHC SMPL_Robot for mesh-derived capsules |
| N8 | **NEW/DONE** | SMPL model files symlinked from ImDy | **DONE** | `data/body_model/smpl/` → prepare5/ImDy/models/smpl/ |
| N9 | **NEW/DONE** | All stiffness/damping paths audited | **DONE** | All simulation paths correctly disable XML defaults; phc_tracker gap fixed |
| N10 | **NEW/DONE** | MP4 comparison script | **DONE** | `scripts/generate_phc_comparison_mp4s.sh` generates default vs PHC videos |

---

## Updated Priority Summary

### FIXED (7 critical + 4 other)
1. ~~Quaternion format bug~~ (prepare/, prepare4/)
2. ~~Paired simulation state split~~ (prepare4/)
3. ~~IK angle wrapping~~ (prepare4/)
4. ~~Smoothing order~~ (prepare2/)
5. ~~FPS/armature mismatch~~ (prepare2/)
6. ~~R_ROT import from old module~~ (newton_vqvae/)
7. ~~XML stiffness leak in phc_tracker~~ (prepare5/)

### Still needs fixing (by priority)

**HIGH — affects correctness:**
1. prepare5 #1: Termination flag race condition in phc_tracker
2. newton_vqvae #2: Batch transpose `.T` on (B,3,3) tensor

**MEDIUM — affects quality:**
3. prepare4 #8: ID uses single zero-torque step (should use multiple)
4. prepare4 #9: rotation_retarget needs inline MPJPE validation
5. prepare2 #6: PD stale for 23/24 substeps in batch_sim_solo
6. prepare2 #12: Large dt in FD zero-torque step

**LOW — cleanup:**
7. Archive dead code (prepare5/optimize_tracking.py, prepare2/optimize_interaction.py)
8. Data path manifest
9. Integration tests

### NO LONGER RELEVANT (3 items)
- prepare2 #4, #5: SolverFeatherstone and optimize_interaction issues — superseded by prepare4's MuJoCo-based pipeline
- prepare3 all: Superseded by prepare5/prepare6

### DEPRECATED
- prepare6 entire folder: 10 unfixed bugs + fundamental smoothness confound. Use prepare4 (torque analysis) or prepare5 (PHC tracker) instead.
