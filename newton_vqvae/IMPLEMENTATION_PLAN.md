# Newton Physics-Informed VQ-VAE: Implementation Plan & Status

## Overview

This module extends InterMask's RVQ-VAE architecture to learn motion tokens that
are **physically valid**, not just kinematically plausible. It does this by running
decoded motions through a differentiable physics simulator (Newton + Featherstone solver)
and backpropagating physics losses into the VQ-VAE encoder/decoder.

**Combined Loss:**
$$L_{\text{VQ-Newton}} = L_{\text{kinematic}} + w(e) \cdot \left(\alpha L_{\text{FK-MPJPE}} + \beta L_{\text{Torque}} + \gamma L_{\text{Skyhook}} + \delta L_{\text{SoftFlow}} + \varepsilon L_{\text{ZMP}} + \zeta L_{\text{ContactBudget}}\right)$$

Where $w(e)$ is a curriculum schedule: 0 during warmup, linear ramp, then 1.0.

---

## Architecture

```
Input motion (B, T, 262)
    ↓
[Encoder] — GCN (LightSAGEConv) + 1D ResNet downsampling
    ↓
[Residual VQ] — 6 quantizers, 1024 codes × 512 dims, EMA codebook
    ↓
[Decoder] — 1D ResNet upsampling + GCN
    ↓
x_hat (B, T, 262)     ← kinematic losses
    ↓
[denormalize + 6D→euler conversion]
    ↓
joint_q (B, T, 76)    ← Newton joint coordinates
    ↓
[PD Controller + Featherstone Solver]  ← 480Hz, 16 substeps per 30fps frame
    ↓
sim_positions, torques, root_forces    ← physics losses
```

### Gradient Flow

```
physics losses → wp.Tape.backward() → PD targets (torch) → decoder → encoder
```

The key bridge is `wp.from_torch()` / `wp.to_torch()` which allows
zero-copy gradient flow between PyTorch and Warp.

---

## File Structure

```
newton_vqvae/
├── __init__.py              # Module marker
├── config.py                # All hyperparameters, constants, CLI parser
├── data_adapter.py          # Dataset, LightSAGEConv, MotionNormalizer
├── encdec.py                # Encoder/Decoder (torch_geometric-free)
├── skeleton_cache.py        # Per-subject Newton model cache
├── newton_bridge.py         # Differentiable simulation bridge
├── physics_losses.py        # 6 physics loss functions + scheduler
├── model.py                 # PhysicsRVQVAE (main model)
├── train.py                 # Training loop (PhysicsVQTrainer)
├── evaluate.py              # Physics evaluation metrics
├── test_pipeline.py         # 8-step integration test
├── test_physics_losses.py   # 11-test physics loss validation suite
├── plot_training_losses.py  # TensorBoard → matplotlib loss plots
└── xml_cache/               # Cached per-subject MJCF XMLs

scripts/
├── run_overfit_test.sh      # 10-clip overfit test
└── run_full_training.sh     # Full 50-epoch training
```

---

## Implementation Status

### ✅ Completed

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Configuration | `config.py` | ✅ Done | TrainingConfig dataclass, PhysicsLossWeights, CLI parser |
| Data loading | `data_adapter.py` | ✅ Done | InterHumanPhysicsDataset, matches InterMask's load_motion exactly |
| LightSAGEConv | `data_adapter.py` | ✅ Done | Drop-in SAGEConv replacement (lin, lin_l, lin_r layout matches checkpoint) |
| Encoder/Decoder | `encdec.py` | ✅ Done | No torch_geometric dependency, loads InterMask weights |
| Motion normalizer | `data_adapter.py` | ✅ Done | Supports both numpy and torch, lazy per-device init |
| Skeleton cache | `skeleton_cache.py` | ✅ Done | Per-subject XML/model cache, sphere feet, soft contact |
| Newton bridge | `newton_bridge.py` | ✅ Done | cont6d→euler, PD sim, Gaussian smoothing |
| Physics losses | `physics_losses.py` | ✅ Done | 6 losses: FK-MPJPE, Torque, Skyhook, SoftFlow, ZMP, ContactBudget |
| Loss scheduler | `physics_losses.py` | ✅ Done | Warmup (5 ep) → linear ramp (10 ep) → full |
| Main model | `model.py` | ✅ Done | PhysicsRVQVAE with kinematic + physics forward |
| Training loop | `train.py` | ✅ Done | PhysicsVQTrainer with TensorBoard logging |
| Evaluation | `evaluate.py` | ✅ Done | 8 physics metrics |
| Integration test | `test_pipeline.py` | ✅ Passing | 8 steps: config→data→losses→model→create→fwd→loss→bwd |
| Physics tests | `test_physics_losses.py` | ✅ Passing | 11 tests all pass |
| Loss plotting | `plot_training_losses.py` | ✅ Done | Multi-panel plot from TensorBoard events |
| Overfit test | `run_overfit_test.sh` | ✅ Run | 10 clips, 5 epochs, loss 5.75→0.39 |
| Full training | `run_full_training.sh` | ✅ Running | 6021 clips, 50 epochs, batch_size 32 |
| Sphere feet | `skeleton_cache.py` | ✅ Done | 4-sphere cluster per foot (heel, 2 ball, toe) |
| Contact softening | `skeleton_cache.py` | ✅ Done | Soft contact ke/kd, margin, friction |
| Z-up coordinate fix | `physics_losses.py` | ✅ Done | All losses use up_axis=2 (Newton Z-up) |

### 🔄 In Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Full kinematic training | Running | Epoch 1 in progress (3585 batches/epoch) |
| Physics-enabled training | Not yet | Need to run overfit with `--with-physics` first |

### 📋 Planned (Future)

| Component | Priority | Notes |
|-----------|----------|-------|
| Phase 2: Pair interaction | Medium | Two characters in same Newton sim for contact |
| Token quality evaluation | Medium | Compare VQ codebook usage before/after physics |
| Ablation study | Low | Individual physics loss contribution analysis |
| Multi-GPU training (DDP) | Low | DataParallel wrapper for PhysicsRVQVAE |

---

## Contact Force Explosion Prevention

The user correctly identified that narrow foot-ground contact (box feet on hard ground)
can cause force explosions. We address this with **5 layered measures**:

### 1. Sphere-Cluster Feet (Geometry)
**File:** `prepare2/gen_smpl_with_sphere_feet_xml.py` → `skeleton_cache.py`

Instead of box collision geometry for feet, we use 4 spheres per foot:
- **Heel sphere** — rear of foot
- **Inner ball sphere** — forefoot medial
- **Outer ball sphere** — forefoot lateral
- **Toe sphere** — on toe body

Spheres have continuous contact normals (no edge discontinuities), providing
much smoother force profiles than box edges.

### 2. Soft Contact Parameters (Physics)
**File:** `skeleton_cache.py` → `_apply_soft_contact_to_model()`

After building the Newton model, we apply:
- **Increased contact margin** (0.005m) — detects contacts earlier, before deep penetration
- **Softened contact stiffness** (ke=1000) — lower than default, gentler force onset
- **High damping ratio** (kd=100) — dissipates energy quickly, prevents bounce
- **Increased friction** (μ=1.0) — stable foot plants

### 3. Torque Clamping (Control)
**File:** `newton_bridge.py`, `config.py`

PD torques are clamped at `TORQUE_LIMIT = 1000 Nm` in the PD controller kernel.
This prevents the tracking controller from generating extreme corrective torques
when contact forces push the body away from targets.

### 4. SoftFlow Loss (Training Signal)
**File:** `physics_losses.py` → `loss_softflow()`

Uses **sigmoid contact detection** instead of hard thresholding:
- `contact_prob = σ(k · (threshold - height))` — smooth, differentiable
- Penetration: `σ(-h) · ReLU(-h)` — gradual onset, no discontinuity
- Sliding: weighted by contact probability — smooth gradient

This ensures the network receives clean gradient signals about foot-ground contact
quality, rather than binary yes/no contact decisions.

### 5. Contact Force Budget Loss (Safety Net)
**File:** `physics_losses.py` → `loss_contact_force_budget()`

Monitors the vertical component of root forces as a proxy for total contact reaction.
When forces exceed a safety budget (500 N), applies a quadratic penalty:

$$L_{\text{contact}} = \left(\frac{\text{ReLU}(|F_z| - F_{\text{budget}})}{F_{\text{budget}}}\right)^2$$

This gives the model a direct incentive to generate motions that don't require
extreme ground reaction forces.

### 6. Temporal Smoothing (Pre-processing)
**File:** `newton_bridge.py` → `gaussian_smooth_1d()`

Before feeding decoded motion to the PD controller, we apply 1D Gaussian smoothing
(configurable σ) to reduce high-frequency jitter. This prevents the PD controller
from chasing rapidly changing targets, which amplifies contact oscillations.

---

## Physics Loss Functions

### L_FK_MPJPE (α = 10.0)
Mean per-joint position error between simulated and FK-derived positions.
**Purpose:** Ensure the physics simulation reproduces the decoded motion.

### L_Torque (β = 0.001)
Normalized L2 of body joint PD torques (excluding root DOFs 0:5).
**Purpose:** Penalize motions requiring extreme actuator efforts.

### L_Skyhook (γ = 1.0)
Log-space penalty on root DOF forces with MAD outlier clipping.
**Purpose:** Minimize invisible support forces ("flying" artifacts).

### L_SoftFlow (δ = 50.0)
Sigmoid-based penetration + contact-weighted sliding penalty.
**Purpose:** Enforce proper foot-ground contact (no penetration, no sliding).

### L_ZMP (ε = 5.0)
CoM projection outside foot bounding box penalty.
**Purpose:** Ensure static/dynamic balance.

### L_ContactBudget (ζ = 0.1)
Quadratic penalty when vertical reaction forces exceed safety budget.
**Purpose:** Prevent contact force explosion.

---

## Training Schedule

```
Epoch 0-4:   Kinematic-only (physics_weight = 0)
             VQ-VAE learns basic motion reconstruction
             InterMask-style: rec + commit + velocity + bone_length + geo + foot_contact

Epoch 5-14:  Ramp physics (physics_weight = 0.0 → 1.0)
             Gradually introduce physics constraints
             Network learns to generate physically plausible tokens

Epoch 15-49: Full physics (physics_weight = 1.0)
             Both kinematic and physics losses at full strength
```

### Physics Batch Frequency
Physics simulation is expensive. By default, physics runs every N batches
(`physics_every_n_batches = 4`). This means 75% of batches are kinematic-only
(fast) and 25% include full Newton simulation.

---

## Data Pipeline

```
Raw .npy files (T, 492)
    ↓ extract 22-joint positions (66) + 21-joint rotations (126)
    ↓ = (T, 192) matching InterMask's load_motion
    ↓
process_motion_np:
    ↓ Y-up → Z-up coordinate transform
    ↓ floor alignment
    ↓ XZ centering (root joint)
    ↓ face Z+ direction
    ↓ compute velocities
    ↓ detect foot contacts
    ↓
(T-1, 262) = positions(66) + velocities(66) + rotations(126) + feet(4)
    ↓ normalize by global mean/std
    ↓
(W, 262) windows with stride
```

Person 2 is aligned relative to Person 1 using `rigid_transform` (same as InterMask).

---

## Environment & Dependencies

- **Python:** 3.10+ (mimickit conda env)
- **PyTorch:** 2.x with CUDA 12.9
- **Warp:** 1.12.0.dev20260219
- **Newton:** Physics engine (from workspace)
- **No torch_geometric** — replaced with LightSAGEConv

### Key missing packages installed:
- `einops` — required by InterMask's quantizer

---

## Running

### Quick test:
```bash
conda activate mimickit
cd /path/to/InterMask
python newton_vqvae/test_pipeline.py          # 8-step integration test
python newton_vqvae/test_physics_losses.py     # 11 physics loss tests
```

### Overfit test (kinematic only):
```bash
bash scripts/run_overfit_test.sh
```

### Overfit test (with physics):
```bash
bash scripts/run_overfit_test.sh --with-physics
```

### Full training:
```bash
bash scripts/run_full_training.sh              # kinematic only
bash scripts/run_full_training.sh --with-physics   # with physics losses
```

### Plot losses:
```bash
python newton_vqvae/plot_training_losses.py --logdir checkpoints/interhuman/overfit_test/logs
python newton_vqvae/plot_training_losses.py --logdir checkpoints/interhuman/newton_vq_phase1/logs
```

---

## Results (So Far)

### Overfit Test (10 clips, kinematic-only, 5 epochs)
| Metric | Epoch 1 | Epoch 5 |
|--------|---------|---------|
| Total Loss | 5.75 | 0.39 |
| Reconstruction | 0.73 | 0.23 |
| Physics | 0.00 (disabled) | 0.00 (disabled) |

### Full Training (6021 clips, kinematic-only, in progress)
| Metric | After ~700 iters |
|--------|-------------------|
| Total Loss | 0.32 |
| Reconstruction | 0.14 |

---

## Key Design Decisions

1. **LightSAGEConv** instead of torch_geometric SAGEConv — avoids Python 3.8 dependency,
   checkpoint-compatible weight layout (lin, lin_l, lin_r).

2. **Sphere feet** instead of box feet — prevents contact force explosions at cube edges.

3. **Sigmoid contact detection** in SoftFlow — smooth gradients vs. hard threshold discontinuity.

4. **Log-space skyhook loss** with MAD outlier clipping — prevents gradient explosion from
   extreme root forces.

5. **Curriculum learning** — kinematic warmup ensures codebook is initialized before
   physics losses add complexity.

6. **Physics batch subsampling** — running physics every N batches (default 4) trades
   accuracy for speed. ~4× faster training vs. physics every batch.
