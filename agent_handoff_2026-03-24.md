# Agent Handoff — 2026-03-24

## What Was Being Worked On

Two parallel workstreams for evaluating **physical plausibility** of InterMask-generated two-person interaction motions, inspired by PP-Motion (ACM MM 2025).

---

## Workstream 1: PHC-style Physics Tracker (`prepare5/`)

### Status: FUNCTIONAL, optimization WIP

A physics-based motion tracker that simulates a humanoid following a reference motion using PD control, producing the **closest physically feasible version**. The tracking error (MPJPE) serves as a physical plausibility metric.

### What's implemented and working:
- **`prepare5/phc_tracker.py`** — Main tracker class (`PHCTracker`), solo + paired simulation
- **`prepare5/run_phc_tracker.py`** — CLI entry point for running the tracker
- **`prepare5/phc_config.py`** — PHC-matched PD gains (Hip kp=800, Torso kp=1000, Root pos kp=5000)
- **`prepare5/phc_reward.py`** — PHC imitation reward function
- **`prepare5/visualize_newton_tracking.py`** — Newton GL viewer (reference vs simulated side-by-side)
- **`prepare5/visualize_tracking.py`** — Matplotlib skeleton visualization + MP4 export

### Current results (zero-residual PD, no learned corrections):
| Clip | MPJPE |
|------|-------|
| 1129 GT | 150 mm |
| 1000 GT | ~75 mm |
| 1129 Gen | ~105 mm |

PHC with RL achieves ~20mm. The gap is because we use zero-residual PD.

### What was being worked on last:
**Differentiable trajectory optimization** (`prepare5/optimize_tracking.py`, `prepare5/run_optimize.py`) to close the gap:
- Optimizes per-frame PD target residuals (Δq) through Newton's differentiable Featherstone solver
- `target_q = ref_q + Δq`, backprop through physics → gradient on Δq → Adam step
- Uses windowed optimization (default window=5 frames, 4 substeps = 120Hz)
- **Status (2026-03-24)**: 20-epoch run completed but **optimization doesn't help**:
  - Training loss decreased 8.6% (0.067 → 0.061) over 20 epochs (~2 hours)
  - **Critical issue: sim-to-sim transfer failure**. Δq tuned for Featherstone solver (differentiable, 120Hz) produces 890mm MPJPE when evaluated with MuJoCo solver (stable contacts, 480Hz) — far worse than baseline 150mm
  - Featherstone solver standalone also diverges (NaN) — its contact model is insufficient without the tape context
  - Higher LR (0.01) causes NaN immediately
  - The Δq grew to |0.076| mean — too large, indicating the optimization is compensating for solver artifacts rather than learning useful corrections
- **Root cause**: Newton's only differentiable solver (Featherstone) handles contacts differently than the stable solver (MuJoCo), creating an unbridgeable sim-to-sim gap
- **Conclusion**: Trajectory optimization via Featherstone backprop is not viable for this use case. Would need either: (a) differentiable MuJoCo solver, (b) RL-based approach (PHC-style), or (c) zeroth-order optimization (CMA-ES/evolution)

### How to run:
```bash
conda activate mimickit

# Basic PD tracker
python prepare5/run_phc_tracker.py --clip-id 1129 --source gt

# Differentiable optimization
python prepare5/run_optimize.py --clip-id 1129 --source gt --epochs 50

# Visualize results
python prepare5/visualize_newton_tracking.py --clip-id 1129 --source gt --run
```

### Key design decisions:
- Joint limits DISABLED for generated data (IK angles exceed MJCF limits → constraint forces fight PD)
- Explicit PD via `control.joint_f` works better than Newton's built-in PD
- Uses `SolverFeatherstone` for optimization (differentiable), `SolverMuJoCo` for final eval (stable)

---

## Workstream 2: Paired-vs-Solo Torque Analysis (`prepare4/`)

### Status: COMPLETE — batch evaluation done on 200 clips each for GT and Generated

All planned files exist and the pipeline has been run:
- `prepare4/paired_simulation.py` — Paired + solo PD simulation engine
- `prepare4/interaction_metrics.py` — 6 metrics: TD, RFD, N3LV, SII, BPS, CTC
- `prepare4/batch_paired_evaluation.py` — Batch processing with resume
- `prepare4/run_paired_analysis.py` — CLI orchestration

### Key Results (200 clips each):

| Metric | GT | Generated | Ratio |
|--------|-----|-----------|-------|
| Torque Delta (Nm) | 4.6 | 145.5 | **32x** |
| Root Force Delta (N) | 48.9 | 434.1 | **9x** |
| BPS (paired) | 1.3% | 51.3% | **39x** |
| Paired hinge torque (Nm) | 11.0 | 145.0 | **13x** |
| Solo hinge torque (Nm) | 8.3 | 7.7 | ~same |
| N3LV | 0.75 | 0.79 | ~same |

**Conclusion**: Generated motions require extreme torques in paired simulation (physically implausible), while solo torques are similar. The torque delta is 32x higher for generated motions.

### Output files:
- `data/paired_eval_gt/paired_eval_results.json`
- `data/paired_eval_generated/paired_eval_results.json`

---

## Existing Infrastructure (from earlier sessions)

### prepare4/ — Single-person torque analysis (COMPLETE)
- `prepare4/run_full_analysis.py` — Main pipeline: `pd_forward_torques()`, `compute_torques_for_person()`
- `prepare4/batch_torque_distribution.py` — Batch torque stats using PD forward sim
- `prepare4/dynamics.py` — PD gains, mass model, biomechanical constants
- `prepare4/retarget.py` — Motion retargeting (rotation + IK)
- `prepare4/gen_xml.py` — MJCF XML generation for SMPL bodies
- `prepare4/view_gt_vs_gen.py` — Newton viewer with torque-driven playback

### prepare2/ — Low-level simulation utilities (COMPLETE)
- `prepare2/pd_utils.py` — `build_model`, `pd_torque_kernel`, `compute_all_pd_torques_np`, etc.

---

## Environment

- **Conda env**: `mimickit` (has Newton, Warp, PyTorch, etc.)
- **GPU**: CUDA device, use `--device cuda:0`
- **Simulator**: NVIDIA Newton (Warp-based, MuJoCo solver backend)
- **Data**: InterHuman dataset at `data/interhuman/`, generated data at `data/test/`
- **SSH**: Work is done via SSH, use `conda run -n mimickit --no-capture-output` prefix

---

## What to Do Next

### Option A: Batch PHC tracker comparison (most promising)
The zero-residual PHC tracker already differentiates GT from generated:
- Clip 1129: GT=150mm vs Gen=193mm (+29%)
- Clip 500: GT=143mm vs Gen=236mm (+64%)
Run batch comparison on 200+ clips with `run_phc_tracker.py` for a PP-Motion-style plausibility metric.

### Option B: Visualization and paper figures
The paired-vs-solo results are very strong (32x torque delta ratio). Create publication-quality plots:
- `prepare4/plot_paired_torques.py` (if not already done)
- Bar charts, violin plots of GT vs Generated metrics

### Option C: Alternative optimization approaches
If better tracking is needed:
- **CMA-ES**: Zeroth-order optimization avoids sim-to-sim transfer issues
- **RL (PHC-style)**: Train a policy network, but expensive
- **Differentiable MuJoCo**: Wait for Newton to add differentiable MuJoCo support

---

## Important Notes

- Previous handoff files: `agent_handoff_2026-03-17.md` (PD forward refactor), `agent_handoff_session_2026-03-13.md` (initial setup)
- Memory files in `~/.claude/projects/-media-rh-codes-sim-InterMask/memory/` have project context
- The plan file at `~/.claude/plans/imperative-jumping-hedgehog.md` has the full paired analysis design
