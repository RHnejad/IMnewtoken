# prepare3 — RL-Based Motion Tracking with Newton Physics

Physics-based character motion tracking using reinforcement learning (PPO)
with **Newton 0.2.0** and **SolverMuJoCo**. This is an alternative to
`prepare2`'s differentiable simulation pipeline, addressing key architectural
limitations identified in the previous approach.

## Architecture Overview

```
                                    ┌──────────────────┐
                                    │   Reference       │
                                    │   Motion (.pkl)   │
                                    └────────┬─────────┘
                                             │
                ┌────────────────────────────┼────────────────────────────┐
                │                            ▼                            │
                │   ┌──────────────────────────────────────────────────┐  │
                │   │              NewtonMimicEnv                      │  │
                │   │                                                  │  │
                │   │  ┌─────────┐   ┌──────────────┐  ┌───────────┐  │  │
                │   │  │ XML     │──▶│ Newton Model │──▶│SolverMuJoCo│  │  │
                │   │  │ Builder │   │ (24 bodies)  │  │ (LCP)     │  │  │
                │   │  └─────────┘   └──────────────┘  └───────────┘  │  │
                │   │                                                  │  │
                │   │  obs: heading-invariant     action: 69 hinge DOFs│  │
                │   │  reward: DeepMimic-style    root forces: ZERO    │  │
                │   └──────────────────────────────────────────────────┘  │
                │                       ▲     │                           │
                │                  obs  │     │  action                   │
                │                       │     ▼                           │
                │              ┌──────────────────────┐                  │
                │              │  PPO Agent            │                  │
                │              │  (ActorCritic MLP)    │                  │
                │              │  512 → 256 → 128      │                  │
                │              └──────────────────────┘                  │
                │                                                        │
                │                  train.py                               │
                └────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Criticism (from prepare2)         | prepare3 Solution                           |
|-----------------------------------|---------------------------------------------|
| World-frame overfitting           | Heading-invariant obs (no global X/Y)       |
| Skyhook root forces               | Root DOFs 0..5 ≡ 0 (no virtual forces)      |
| Contact differentiability gap     | SolverMuJoCo (LCP contacts, no penalty)     |
| Finite difference noise           | RL: no velocity differentiation needed      |
| Alternating optimization issues   | End-to-end PPO (single optimization)        |

## Pipeline

### Phase 1: XML Generation (`xml_builder.py`)

Generates per-subject SMPL MJCF XML files from body shape parameters (betas),
with SHA-256 hash-based caching to avoid redundant regeneration in RL resets.

```bash
# Programmatic usage (called by env automatically)
from prepare3.xml_builder import get_or_create_xml
xml_path = get_or_create_xml(betas_array)
```

### Phase 2: Motion Conversion (`convert_to_mimickit.py`)

Converts `prepare2` retargeted trajectories (Newton `joint_q` arrays) to
MimicKit-compatible `.pkl` motion format.

```bash
# Convert a single clip
python prepare3/convert_to_mimickit.py \
    --clip 1000 \
    --data-dir data/retargeted_v2/interhuman \
    --output-dir data/mimickit_motions/interhuman

# Convert entire dataset
python prepare3/convert_to_mimickit.py \
    --dataset interhuman \
    --output-dir data/mimickit_motions/interhuman
```

**Format conversion:**
```
Newton joint_q (76,): [root_pos(3), root_quat_xyzw(4), hinge_angles(69)]
    ↓
MimicKit frame (75,): [root_pos(3), root_expmap(3), hinge_angles(69)]
```

### Phase 3: RL Environment (`newton_mimic_env.py`)

Gymnasium-style environment wrapping Newton physics:

- **Solver**: `SolverMuJoCo` (LCP-based contacts, NOT `SolverFeatherstone`)
- **requires_grad**: `False` everywhere (no `wp.Tape`, pure forward sim)
- **Action space**: `(69,)` hinge DOF position targets (PD mode) or torques
- **Root DOFs**: Always zero — character must self-balance
- **Observations**: Heading-invariant proprioceptive features:
  - Root height (1)
  - Local root orientation via gravity/up vectors (6)
  - Local root velocity and angular velocity (3+3)
  - Joint positions and velocities (69+69)
  - Phase encoding sin/cos (2)
  - Target joint angles (69, optional)
  - **Total: 222 dims**
- **Reward**: DeepMimic-style tracking (pose + vel + root_pose + root_vel + key_pos)
- **Termination**: Fall detection (height), orientation deviation, NaN/Inf

### Phase 4: Training (`train.py`)

Standalone PPO with GAE-Lambda advantages, clipped surrogate objective,
observation normalization, TensorBoard logging, and checkpoint management.

```bash
python prepare3/train.py \
    --motion data/mimickit_motions/interhuman/1000_person0.pkl \
    --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \
    --output-dir output/prepare3/1000_p0 \
    --max-steps 10000000 \
    --device cuda:0
```

**Key hyperparameters:**

| Parameter        | Default | Description                    |
|------------------|---------|--------------------------------|
| `--lr`           | 3e-4    | Learning rate                  |
| `--gamma`        | 0.99    | Discount factor                |
| `--lam`          | 0.95    | GAE lambda                     |
| `--clip-eps`     | 0.2     | PPO clipping                   |
| `--hidden-dims`  | 512 256 128 | MLP layer sizes           |
| `--sim-freq`     | 240     | Physics Hz                     |
| `--control-freq` | 30      | Policy Hz                      |
| `--control-mode` | pd      | PD position targets vs torques |

### Phase 5: Evaluation (`evaluate_policy.py`)

Run deterministic policy rollouts and compute tracking metrics:

```bash
python prepare3/evaluate_policy.py \
    --checkpoint output/prepare3/1000_p0/best_policy.pt \
    --motion data/mimickit_motions/interhuman/1000_person0.pkl \
    --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \
    --num-episodes 50 \
    --save-trajectory
```

**Metrics reported:**
- **MPJPE** (Mean Per-Joint Position Error, meters)
- Root position / orientation error
- Joint angle error (radians)
- CoM drift (meters)
- Survival rate (fraction of reference motion completed)
- Episode reward statistics

## File Structure

```
prepare3/
├── __init__.py              # Package init
├── xml_builder.py           # SMPL XML generation + SHA-256 cache
├── convert_to_mimickit.py   # Newton → MimicKit motion conversion
├── newton_mimic_env.py      # Newton RL environment (Gymnasium-style)
├── train.py                 # PPO training loop
├── evaluate_policy.py       # Policy evaluation + MPJPE logging
├── test_prepare3.py         # Unit tests
├── README.md                # This file
└── xml_cache/               # Cached per-subject XML files
```

## Dependencies

- **Newton** ≥ 0.2.0
- **Warp** ≥ 1.12.0
- **PyTorch** ≥ 2.0
- **NumPy**, **SciPy**
- **TensorBoard** (optional, for logging)

## Comparison with prepare2

| Feature          | prepare2 (Differentiable)       | prepare3 (RL)                 |
|------------------|---------------------------------|-------------------------------|
| Solver           | SolverFeatherstone              | SolverMuJoCo                  |
| Gradients        | wp.Tape backprop                | None (forward only)           |
| Contacts         | Penalty-based                   | LCP (hard constraints)        |
| Root control     | Virtual forces (skyhook)        | Zero (self-balance)           |
| Observations     | World-frame → ego-relative      | Heading-invariant from start  |
| Optimization     | Alternating / joint             | End-to-end PPO                |
| Training time    | Minutes (few gradient steps)    | Hours (millions of env steps) |
| Generalization   | Per-clip optimization           | Policy can generalize         |
