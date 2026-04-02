# prepare6/ — PPO RL-Based Physics Motion Tracker

## Overview

`prepare6/` implements a **per-clip PPO reinforcement learning tracker** for evaluating the physical plausibility of two-person interaction motions. It replaces/complements the open-loop PD tracker in `prepare5/` with a learned policy that actively corrects tracking errors during physics simulation.

**Core idea:** A physically plausible motion should be easy to track in physics simulation. Ground truth (GT) motions track better (lower MPJPE) than generated motions — the MPJPE gap is the physics plausibility metric.

## Motivation

| Approach | Prepare | MPJPE (GT clip 1129) | Issue |
|----------|---------|----------------------|-------|
| Differentiable optimisation | `prepare5/optimize_tracking.py` | ~890 mm | Featherstone ≠ MuJoCo solver mismatch |
| Open-loop PD (kinematic) | `prepare5/phc_tracker.py` | 70–886 mm | Cannot recover from drift; no contact adaptation |
| **PPO RL (this work)** | `prepare6/` | **47.9 mm** | Learns corrective actions via reward signal |

NVIDIA's ProtoMotions and SONIC (GTC 2026) use the same architecture — RL (PPO) on top of the Newton physics engine — achieving ~20 mm MPJPE on single-person tracking. Our implementation adapts this for per-clip training on InterHuman two-person interaction data.

## Architecture

```
Reference Motion (T×76)
         │
         ▼
┌─────────────────────────┐
│  RolloutEnv (Newton)    │  N_ENVS=64–256 parallel characters
│  ├─ builder.replicate() │  via Newton's multi-env API
│  ├─ MuJoCo-Warp solver  │  ground contacts, gravity, friction
│  └─ Explicit PD control │  root skyhook + hinge torques
└────────────┬────────────┘
             │ obs (446-dim), reward, done
             ▼
┌─────────────────────────┐
│  PPO Agent (PyTorch)    │
│  ├─ PolicyNet [512,256,128] → 69-dim residual actions
│  ├─ ValueNet  [512,256,128] → scalar value
│  └─ GAE + clipped PPO   │  ProtoMotions-matched hyperparams
└─────────────────────────┘
```

### Observation Space (446 dimensions)

| Range | Content | Dims |
|-------|---------|------|
| 0–65 | Sim body positions (root-relative, yaw-normalised) | 22×3 |
| 66–153 | Sim body rotations (root-frame quaternions) | 22×4 |
| 154–219 | Sim body linear velocities (root frame) | 22×3 |
| 220–285 | Sim body angular velocities (root frame) | 22×3 |
| 286 | Root height | 1 |
| 287–289 | Gravity vector in root frame | 3 |
| 290–355 | Reference body positions (root-relative) | 22×3 |
| 356–443 | Reference body rotations (root-relative) | 22×4 |
| 444–445 | Phase encoding (sin/cos) | 2 |

### Action Space (69 dimensions)

Residual joint angle offsets for 23 hinge bodies × 3 DOFs each:
```
target_angle = ref_angle + tanh(action) × 0.5 × π
```

Root DOFs (6D) are handled by a fixed PD controller, not the learned policy.

### Reward Function

Reuses `prepare5/phc_reward.py` — PHC-style imitation reward:
```
r = 0.5·exp(-100·pos_err²) + 0.3·exp(-10·rot_err²) + 0.1·exp(-0.1·vel_err²) + 0.1·exp(-0.1·ang_vel_err²)
```

### Training

- Per-clip: one small MLP trained per clip (~200k–500k timesteps)
- Reference State Initialisation (RSI): random start frame with 90% probability
- Early stopping: when mean reward exceeds 0.85 for 20 consecutive updates
- Termination: root drift > 25 cm or height < 30 cm

## Key Design Decisions (Verified from ProtoMotions Source)

1. **Multi-env construction**: `builder.replicate(robot, N)` — NOT per-env `add_mjcf()` calls. Verified from [ProtoMotions simulator/newton/simulator.py](https://github.com/NVlabs/ProtoMotions).

2. **Passive force zeroing**: `model.mujoco.dof_passive_stiffness.zero_()` after `finalize()` to prevent double-counting with explicit PD control.

3. **PPO hyperparameters**: Matched to ProtoMotions defaults — actor LR=2e-5, critic LR=1e-4, clip_eps=0.2, GAE lambda=0.95, actor_logstd=-2.9 (fixed).

4. **Explicit PD over built-in PD**: All torques via `control.joint_f` for full control visibility (used by ProtoMotions for `ControlType.PROPORTIONAL`).

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `rl_config.py` | ~55 | Hyperparameters (extends `prepare5/phc_config.py`) |
| `obs_builder.py` | ~145 | 446-dim observation vector construction |
| `rollout_env.py` | ~600 | Vectorised Newton env with `builder.replicate()` |
| `ppo_agent.py` | ~260 | Pure PyTorch PPO (no external RL library) |
| `rl_tracker.py` | ~140 | Per-clip train-and-evaluate orchestrator |
| `run_rl_tracker.py` | ~190 | CLI entry point (single-clip) |
| `batch_rl_evaluation.py` | ~210 | Batch evaluation on test set |

## Usage

```bash
# Single clip (smoke test, ~2 min)
python prepare6/run_rl_tracker.py --clip-id 1129 --source gt \
    --n-envs 4 --total-timesteps 10000

# Single clip (full training, ~6 min)
python prepare6/run_rl_tracker.py --clip-id 1129 --source gt \
    --n-envs 64 --total-timesteps 500000

# Compare RL vs PHC baseline
python prepare6/run_rl_tracker.py --clip-id 1129 --source gt --compare-phc

# Batch evaluation (test set)
python prepare6/batch_rl_evaluation.py --n-clips 20 --source both

# Full test set (long run, resume-able)
python prepare6/batch_rl_evaluation.py --n-clips 0 --source both --resume
```

## Results

### Single Clip (1129 GT, 500k timesteps, 64 envs)

| Method | MPJPE | Improvement |
|--------|-------|-------------|
| PHC open-loop PD | 886.4 mm | — |
| **RL (PPO)** | **47.9 mm** | **18× better** |

Training curve: reward 0.36 → 0.71 over 245 updates (~36 min on RTX 3090).

### Physics Plausibility Metric

The metric is the **MPJPE gap** between GT and generated motions when tracked through the RL tracker. If generated motions are less physically plausible, they will have higher tracking error:

```
ΔPhysics = MPJPE_generated − MPJPE_gt
```

A positive gap indicates the metric correctly identifies physically implausible motions.

## Dependencies

- Same as `prepare5/`: Newton (Warp), PyTorch, numpy, scipy
- No external RL libraries required
- Reuses: `prepare5/phc_config.py`, `prepare5/phc_reward.py`, `prepare4/gen_xml.py`, `prepare4/dynamics.py`
