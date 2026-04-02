# Physics Plausibility Metric — Methodology

## The Core Idea

**If a motion is physically plausible, a simulated humanoid should be able to reproduce it accurately in a physics engine. If it's not physically plausible (e.g., floating, interpenetrating, impossible accelerations), the physics engine will deviate from the reference — producing higher tracking error.**

We measure this tracking error as **MPJPE (Mean Per-Joint Position Error)** in millimetres. The gap between GT and generated MPJPE is the physics plausibility metric:

```
ΔPhysics = MPJPE_generated − MPJPE_gt
```

A positive gap means generated motions are less physically plausible than GT.

---

## What We Built

### The Physics Tracker (`prepare6/`)

A **per-clip PPO reinforcement learning agent** that learns to make a simulated humanoid follow a reference motion as closely as possible, subject to real physics constraints (gravity, ground contacts, joint torque limits, rigid body dynamics).

### Why RL, Not Simple PD Control?

We tried three approaches in chronological order:

| Approach | Folder | Result | Problem |
|----------|--------|--------|---------|
| **Inverse dynamics** | `prepare4/` | Skyhook forces needed | No ground contacts → unphysical torques |
| **Differentiable optimisation** | `prepare5/optimize_tracking.py` | 890 mm MPJPE | Featherstone solver ≠ MuJoCo solver (sim-to-sim mismatch) |
| **Open-loop PD control** | `prepare5/phc_tracker.py` | 70–886 mm MPJPE | Cannot recover from drift; no contact adaptation |
| **PPO RL (this work)** | `prepare6/` | **47–62 mm MPJPE** | Current best; matches NVIDIA's approach |

The RL approach is validated by NVIDIA's ProtoMotions and SONIC (GTC 2026), which use the same architecture (PPO + Newton physics engine) and achieve ~20 mm MPJPE on single-person tracking.

---

## How It Works Step-by-Step

### Step 1: Reference Motion Input

Input: a reference motion as **Newton joint coordinates** `(T, 76)` — 7 root coordinates (3 position + 4 quaternion) + 69 hinge angles (23 joints × 3 DOFs).

This comes from either:
- **GT (ground truth):** InterHuman dataset SMPL-X parameters, retargeted to Newton skeleton via rotation mapping (`prepare4/retarget.py`)
- **Generated:** InterMask output, retargeted via IK from predicted joint positions

### Step 2: Physics Simulation Environment

We build a **Newton physics simulation** with:
- **Humanoid model:** MJCF XML skeleton generated from SMPL-X body shape (`betas`) via `prepare4/gen_xml.py`
- **Physics engine:** Newton (NVIDIA Warp) with MuJoCo-Warp solver
- **Ground plane:** flat surface with friction
- **Gravity:** -9.81 m/s² in Z
- **Simulation rate:** 480 Hz (16 substeps per 30 fps frame)

64 parallel copies of the humanoid are created in a single Newton model using `builder.replicate()` for efficient batch simulation.

### Step 3: PPO Training (Per-Clip)

A small neural network (MLP: 512→256→128) is trained to control the humanoid's joints to follow the reference motion.

**Observation (446 dimensions):**
The policy sees the humanoid's current state and the reference target:
- Current body positions/rotations/velocities (root-relative, yaw-normalised)
- Root height and gravity direction
- Next reference frame's body positions/rotations
- Phase (how far through the clip)

**Action (69 dimensions):**
The policy outputs **residual joint angle corrections** added to the reference joint angles:
```
target_angle = reference_angle + tanh(action) × 0.5 × π
```
A PD controller then computes torques to reach these target angles:
```
τ = kp × (target − current_angle) − kd × angular_velocity
```

The root (pelvis) uses a fixed PD controller tracking the reference root position/orientation — this prevents the character from walking away but doesn't prevent falls or contact violations.

**Reward function** (from PHC — Perpetual Humanoid Controller):
```
reward = 0.5 × exp(−100 × position_error²)
       + 0.3 × exp(−10  × rotation_error²)
       + 0.1 × exp(−0.1 × velocity_error²)
       + 0.1 × exp(−0.1 × angular_velocity_error²)
```
Each term compares the simulated body state to the reference across all 22 SMPL joints. The reward is in [0, 1] — 1.0 means perfect tracking.

**Training details:**
- Algorithm: PPO (Proximal Policy Optimisation) with GAE
- 64 parallel environments, 32 rollout steps per update
- ~200k–500k total timesteps (~6–36 minutes on RTX 3090)
- Reference State Initialisation (RSI): reset to random frames for better coverage
- Termination: root drift > 25 cm or character falls below 30 cm height

### Step 4: Evaluation

After training, the policy is run **deterministically** (mean action, no sampling) from frame 0 through the entire clip. At each frame, we extract the simulated body positions and compare against the reference:

```
MPJPE = mean over all frames and joints of ||sim_position − ref_position||
```

### Step 5: Compare GT vs Generated

We run the entire pipeline on both GT and generated versions of the same clip:
- **GT clip → Train RL tracker → MPJPE_gt**
- **Generated clip → Train RL tracker → MPJPE_gen**

The **physics plausibility gap** = MPJPE_gen − MPJPE_gt.

---

## What Physics Properties Does This Capture?

The tracking MPJPE implicitly measures violations of:

1. **Newton's laws:** Impossible accelerations require impossible forces → the PD controller can't produce them → tracking error increases.

2. **Ground contact:** Floating/sliding motions don't have proper ground reaction forces → the simulated character falls or drifts → high MPJPE.

3. **Joint torque limits:** Extreme accelerations require torques beyond the PD controller's capacity (clamped at 1000 Nm) → the character can't follow → tracking error.

4. **Conservation of momentum:** Generated motions may have discontinuous velocities or impossible direction changes → the inertial physics engine smooths these out → the simulated trajectory deviates.

5. **Body dynamics:** Mass distribution, moments of inertia, segment lengths all affect how the simulated body responds to the same joint commands → physically implausible motions create larger discrepancies.

---

## Tools and Libraries Used

| Tool | Version | Purpose |
|------|---------|---------|
| **Newton** | via Warp 1.12 | GPU-accelerated physics engine (rigid body dynamics) |
| **MuJoCo-Warp solver** | (built into Newton) | Contact resolution, constraint solving |
| **NVIDIA Warp** | 1.12.0 | GPU compute kernels for PD control |
| **PyTorch** | — | PPO policy and value networks |
| **SMPL-X** | — | Human body model (shape params → skeleton) |
| **scipy** | — | Quaternion operations (Rotation class) |
| **numpy** | — | Array operations |

**No external RL libraries** — PPO is implemented in ~250 lines of pure PyTorch (`prepare6/ppo_agent.py`).

---

## Code Structure

```
prepare6/
├── rl_config.py              # All hyperparameters
├── obs_builder.py             # 446-dim observation construction
├── rollout_env.py             # Vectorised Newton environment
├── ppo_agent.py               # PPO agent (PolicyNet + ValueNet + update)
├── rl_tracker.py              # Per-clip train-and-evaluate orchestrator
├── run_rl_tracker.py          # CLI for single clips
├── batch_rl_evaluation.py     # Batch evaluation on test set
├── README.md                  # Architecture overview
└── METHODOLOGY.md             # This document
```

**Dependencies on other prepare folders:**
- `prepare4/gen_xml.py` — generates MJCF XML from SMPL-X betas
- `prepare4/dynamics.py` — sets segment masses
- `prepare4/retarget.py` — SMPL-X → Newton joint coordinate retargeting
- `prepare4/run_full_analysis.py` — data loading (load_gt_persons, load_gen_persons)
- `prepare5/phc_config.py` — PD gains, body definitions, simulation constants
- `prepare5/phc_reward.py` — PHC imitation reward function

---

## Results So Far

### Single Clip (1129 GT)

| Method | MPJPE |
|--------|-------|
| Open-loop PD (`prepare5`) | 886.4 mm |
| **RL PPO (`prepare6`)** | **47.9 mm** |

### Test Set (20 clips, GT vs Generated)

Full batch evaluation on 20 randomly sampled test clips:

| Statistic | GT MPJPE | Gen MPJPE |
|-----------|----------|-----------|
| Mean | 72.1 mm | 55.4 mm |
| Median | 58.3 mm | 40.2 mm |
| Std | 40.9 mm | 37.2 mm |

**Overall gap: −16.7 mm** (generated tracks *better* than GT on average).

#### Per-Clip Results (selected)

| Clip | GT MPJPE | Gen MPJPE | Gap | Notes |
|------|----------|-----------|-----|-------|
| 161 | 61.4 mm | 80.6 mm | +19.2 mm | Gen has contact issues |
| 880 | 50.3 mm | 67.0 mm | +16.7 mm | Gen has contact issues |
| 1197 | 73.0 mm | 124.9 mm | +51.9 mm | Long clip, gen drifts |
| 4903 | 47.0 mm | 72.3 mm | +25.3 mm | Gen has implausible motion |
| 1147 | 58.1 mm | 27.3 mm | −30.8 mm | Gen is smoother |
| 3955 | 125.5 mm | 40.9 mm | −84.6 mm | GT has fast motion |
| 4324 | 122.6 mm | 27.4 mm | −95.2 mm | Short gen clip |

Positive gap (gen tracks worse): **8 / 20 clips (40%)**

### Key Finding: Smoothness Confound

**The raw MPJPE gap does NOT reliably indicate physics plausibility.** The reason:

1. **Generated motions are smoother:** Neural network outputs (InterMask) are inherently smooth — no measurement noise, no sharp contact events, no subtle balance corrections. Smoother motions are easier for PD control to track.

2. **GT motions have more complexity:** Real human motions include rapid direction changes, contact-rich interactions, subtle balance adjustments. These are harder to track even though they are physically valid.

3. **The metric confounds smoothness with plausibility:** A floating person moving slowly in a straight line would track perfectly (low MPJPE) but is physically impossible.

### Implications for Future Work

The per-clip RL tracker successfully measures **tracking difficulty**, but tracking difficulty ≠ physical plausibility. To build a valid metric, we need one of:

1. **Normalise by motion complexity:** Divide MPJPE by a complexity measure (e.g., acceleration magnitude, contact frequency) so smooth-but-implausible motions don't get a free pass.

2. **Contact-conditioned evaluation:** Only compare MPJPE during contact events (pushes, catches, collisions) where physics violations are most apparent.

3. **Root residual force (ΔF_sky):** Measure the skyhook force needed to keep the character on-track — this directly quantifies "how much external help does the physics engine need?" regardless of smoothness. This was the original metric idea from `prepare2/` and may be the correct approach after all.

4. **Paired vs solo gap:** Run the same motion solo and paired. The paired-solo torque difference isolates interaction forces from self-balance, avoiding the smoothness confound. This is what `prepare4/batch_paired_evaluation.py` already implements analytically.

---

## References

- **PHC (Perpetual Humanoid Controller):** Luo et al., 2023. Our reward function and PD gains are matched to PHC.
- **ProtoMotions (NVIDIA):** GPU RL framework for humanoid control. Our Newton multi-env construction follows their API.
- **SONIC (NVIDIA):** Foundation model for humanoid whole-body control. Validates RL > differentiable for motion tracking.
- **DeepMimic:** Peng et al., 2018. The original reward formulation for physics-based motion imitation.
- **PP-Motion:** Physics-based plausibility metric using PHC tracking error (conceptually similar to our approach).
