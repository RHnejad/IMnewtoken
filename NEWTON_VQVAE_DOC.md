# Physics-Informed VQ-VAE (Newton VQ-VAE)

## Implementation Progress Documentation

**Last Updated:** Session 1  
**Status:** Core framework implemented, ready for integration testing

---

## 1. Overview

This module extends InterMask's RVQ-VAE to produce **physically valid** motion tokens
by incorporating a differentiable Newton physics simulator into the VQ-VAE training loop.

### Core Idea

Instead of learning tokens that are only kinematically plausible, we train the VQ-VAE
so that its reconstructed motions can be PD-tracked by a physically simulated humanoid
with minimal external support forces ("skyhook").

### Loss Function

```
L_VQ_Newton = α L_FK_MPJPE + β L_Torque + γ L_Skyhook + δ L_SoftFlow + ε L_ZMP
```

Where:
- **L_FK_MPJPE** (α=10): Forward kinematics position error between decoded motion and simulated motion
- **L_Torque** (β=0.001): Regularize PD tracking torques (large torques → unphysical motion)
- **L_Skyhook** (γ=1.0): Penalize root support forces (the "invisible hand" holding the character up)
- **L_SoftFlow** (δ=50): Smooth penalty contacts for foot-ground interaction (penetration + sliding)
- **L_ZMP** (ε=5.0): Zero Moment Point stability (CoM over support polygon)

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                   PhysicsRVQVAE                          │
│                                                          │
│  Input (B,T,262)                                         │
│     ↓                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ Encoder  │ →  │ RVQ Code │ →  │ Decoder  │ → x_hat   │
│  │(SAGEConv)│    │ book     │    │(Upsample)│           │
│  └──────────┘    └──────────┘    └──────────┘           │
│                                       ↓                  │
│                              ┌────────────────┐          │
│                              │ 6D_rot →       │          │
│                              │ Newton joint_q │          │
│                              └────────┬───────┘          │
│                                       ↓                  │
│                              ┌────────────────┐          │
│    Kinematic Losses ←───── x_hat              │          │
│    (rec, vel, geo,           │ Newton PD Sim  │          │
│     bone, fc, commit)        │ (Featherstone) │          │
│         ↓                    └────────┬───────┘          │
│         ↓                             ↓                  │
│    L_kinematic  +  ω(epoch) × ┌──────────────┐          │
│                               │Physics Losses│          │
│         ↓                     │ FK_MPJPE     │          │
│    ┌─────────┐                │ Torque       │          │
│    │ Total   │ ← ───────────  │ Skyhook      │          │
│    │ Loss    │                │ SoftFlow     │          │
│    └────┬────┘                │ ZMP          │          │
│         ↓                     └──────────────┘          │
│    loss.backward() → wp.Tape.backward() → gradients     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Module Structure

```
InterMask/
├── newton_vqvae/              # NEW: Physics-informed VQ-VAE module
│   ├── __init__.py            # ✅ Module docstring
│   ├── config.py              # ✅ TrainingConfig, PhysicsLossWeights, constants
│   ├── data_adapter.py        # ✅ LightSAGEConv, MotionNormalizer, Dataset
│   ├── skeleton_cache.py      # ✅ SkeletonCache (Newton model management)
│   ├── newton_bridge.py       # ✅ Differentiable Newton simulation bridge
│   ├── physics_losses.py      # ✅ 5 physics loss functions + scheduler
│   ├── model.py               # ✅ PhysicsRVQVAE (wraps InterMask RVQVAE)
│   ├── train.py               # ✅ PhysicsVQTrainer training loop
│   └── evaluate.py            # ✅ Physics metrics evaluation suite
├── scripts/
│   ├── train_phase1.sh        # ✅ Phase 1: per-character training
│   └── train_phase2.sh        # ✅ Phase 2: paired interaction training
└── NEWTON_VQVAE_DOC.md        # ✅ This documentation file
```

---

## 3. File-by-File Description

### 3.1 `config.py` — Configuration

**Constants:**
- `SIM_FREQ = 480` Hz simulation frequency
- `MOTION_FPS = 30` Hz motion data framerate
- `SIM_SUBSTEPS = 16` (480/30) substeps per frame
- `TORQUE_LIMIT = 1000` Nm maximum PD torque
- `DOFS_PER_PERSON = 75`, `COORDS_PER_PERSON = 76`, `BODIES_PER_PERSON = 24`
- `N_SMPL_JOINTS = 22`, `N_JOINT_Q = 76` (3 pos + 4 quat + 69 hinges)

**Dataclasses:**
- `PhysicsLossWeights`: α, β, γ, δ, ε (default values tuned from prepare2 experiments)
- `TrainingConfig`: All hyperparameters (lr, batch_size, epochs, physics schedule, etc.)
- `make_config_from_args()`: CLI argument parser → TrainingConfig

### 3.2 `data_adapter.py` — Data Loading

**`LightSAGEConv`**: Drop-in replacement for `torch_geometric.nn.SAGEConv` that works
without torch_geometric. Uses mean-aggregation + linear projection. ~100 lines.
This solves the Python 3.8 vs 3.10 dependency conflict.

**`MotionNormalizer`**: Loads InterMask's mean/std from `data/Mean.npy` and `data/Std.npy`.
Provides `forward()` (normalize) and `backward()` (denormalize) methods.

**`InterHumanPhysicsDataset`**: Extends InterHuman dataset to also load SMPL-X betas.
Returns `{'motion': (T, 262), 'betas': (10,), 'clip_id': str}`.

### 3.3 `skeleton_cache.py` — Newton Model Management

**`SkeletonCache`**: Caches Newton `Model` objects keyed by SMPL-X betas hash.
- `get_model(betas)`: Returns single-person Newton Model + ground plane
- `get_pair_model(betas1, betas2)`: Returns two-person Newton Model
- `get_body_offset(betas)`: Returns SMPL-X body-specific translation offset
- Uses `prepare2/gen_smpl_xml.py` for XML generation, `prepare2/pd_utils.py` for model building

### 3.4 `newton_bridge.py` — Differentiable Simulation Bridge

**Rotation Conversions (differentiable):**
- `cont6d_to_matrix()`: 6D continuous rotation → 3×3 rotation matrix
- `rotmat_to_euler_xyz()`: Rotation matrix → XYZ Euler angles (for Newton D6 joints)
- `rotmat_to_quat()`: Rotation matrix → quaternion (x,y,z,w) for Newton root

**`decoder_output_to_joint_q()`**: Converts VQ-VAE decoder output (B, T, 262) to
Newton joint coordinates (B, T, 76). Handles:
- Root position extraction from decoded positions
- Root orientation from hip-facing direction → quaternion
- Body joint rotations via R_ROT conjugation + XYZ Euler decomposition

**`DifferentiableNewtonSim`**: Core simulation class.
- `simulate_single()`: PD-tracked simulation for one character using SolverFeatherstone
  - 480Hz simulation with 16 substeps per 30fps frame
  - Uses `pd_torque_kernel` (Warp GPU kernel) for PD control within wp.Tape graph
  - Returns: sim_positions (T, 22, 3), pd_torques (T, 75), root_forces (T, 6), body_positions (T, 24, 3)
- `simulate_pair()`: Two characters in same simulation (Phase 2)

**`gaussian_smooth_1d()`**: Temporal Gaussian smoothing for decoded joint angles.
Configurable σ (default 1.0). Applied before PD tracking to reduce jitter.

### 3.5 `physics_losses.py` — Physics Loss Functions

Five loss functions + combined module + curriculum scheduler:

1. **`loss_fk_mpjpe()`**: Mean per-joint position error (simulated vs decoded FK)
2. **`loss_torque()`**: Body joint torque regularization (excludes root DOFs 0:6)
   - Clamps at torque_limit, normalizes, L2 or L1
3. **`loss_skyhook()`**: Root support force penalty with 3-tier regularization:
   - Log-space: `log(1 + ||F||)` to avoid gradient explosion
   - MAD outlier clipping at 3σ
   - Weights translational more than rotational
4. **`loss_softflow()`**: Smooth penalty contacts (foot-ground interaction):
   - Penetration: sigmoid(-h) × |h| for feet below ground
   - Sliding: penalize horizontal velocity during contact
   - Uses sigmoid for differentiable contact detection (no hard thresholding)
5. **`loss_zmp()`**: Zero Moment Point stability — CoM inside foot support polygon

**`PhysicsLoss`**: nn.Module combining all 5 losses with configurable weights.

**`PhysicsLossScheduler`**: Curriculum learning:
- Warmup epochs (default 5): physics weight = 0
- Ramp epochs (default 10): linear ramp 0 → 1
- After: full physics weight

### 3.6 `model.py` — PhysicsRVQVAE

Wraps InterMask's encoder/decoder/quantizer with physics simulation.

**Key methods:**
- `forward(x, betas, run_physics)`: Full forward pass with optional physics sim
- `_physics_forward(x_hat, betas)`: Run decoded motion through Newton sim
- `compute_losses(x, result, geo_losses)`: Combined kinematic + physics losses
- `load_intermask_checkpoint(ckpt_path)`: Load pre-trained InterMask weights
- `set_epoch(epoch)`: Update physics loss scheduling

**Training modes:**
- Kinematic-only (warmup): Standard VQ-VAE reconstruction losses
- Physics-enabled: Adds physics losses with scheduled weight

### 3.7 `train.py` — Training Loop

**`PhysicsVQTrainer`**: Complete training pipeline.
- AdamW optimizer (lr=2e-4, betas=(0.9, 0.99))
- LR warmup (¼ epoch) + MultiStepLR (70%/85%)
- Gradient clipping (max_norm=1.0)
- TensorBoard logging
- Checkpoint management (latest.tar, best.tar)
- Physics simulation every N batches (configurable, default 4) for efficiency
- Validation without physics (fast)

### 3.8 `evaluate.py` — Evaluation

Computes 8 physics quality metrics:
1. FK-MPJPE (mm)
2. Torque RMS (Nm)
3. Skyhook force (N)
4. Penetration rate (%)
5. Penetration depth (mm)
6. Foot sliding (m/frame)
7. ZMP violation (m)
8. Codebook perplexity

Outputs JSON results file.

---

## 4. Training Protocol

### Phase 1: Per-Character Training

```bash
conda activate mimickit
bash scripts/train_phase1.sh
```

1. **Load InterMask pre-trained VQ-VAE** (warm start from existing kinematic model)
2. **Kinematic warmup** (epochs 1-5): Standard VQ-VAE losses only
3. **Physics ramp** (epochs 6-15): Linearly increase physics loss weight 0 → 1
4. **Full physics** (epochs 16-50): Both kinematic and physics losses at full weight

Physics simulation runs on every 4th batch (configurable) to manage GPU time.

### Phase 2: Paired Interaction (Future)

```bash
conda activate mimickit
bash scripts/train_phase2.sh
```

Uses Phase 1 model as starting point. Both characters in same Newton simulation
enables proper contact forces between them.

---

## 5. Key Design Decisions

### 5.1 SolverFeatherstone (not MuJoCo)

Newton's `SolverFeatherstone` is used because:
- Differentiable via `wp.Tape` (MuJoCo LCP is not)
- Penalty contacts align with SoftFlow's smooth penalty model
- "Weakness" (no hard contacts) becomes a feature: gradients flow through contacts

### 5.2 LightSAGEConv (no torch_geometric)

InterMask uses `torch_geometric.nn.SAGEConv` which requires Python 3.8.
Newton/Warp requires Python 3.10+. Solution: re-implement SAGEConv as a simple
mean-aggregation + linear projection (~100 lines, equivalent behavior for
the skeleton graph used in InterMask).

### 5.3 Skyhook Explosion Mitigation

Root forces can explode to 10^16 Nm (see prepare2/notes.md bugs #1, #10).
Three-tier mitigation:
1. Temporal Gaussian smoothing of decoded joint angles (σ=1.0)
2. Log-space loss: `log(1 + ||F||)` instead of `||F||²`
3. Per-sample MAD outlier clipping at 3σ

### 5.4 Physics Every N Batches

Physics simulation is expensive (~200ms per sample at 64 frames).
Running physics on every batch would be prohibitive. Instead:
- Physics runs every 4th batch (configurable)
- Kinematic losses run every batch
- This gives ~4x speedup while still providing physics signal every few steps

---

## 6. Dependencies

### Required Environment: `mimickit` conda env

```
Python >= 3.10
PyTorch >= 2.0
NVIDIA Warp
Newton (local, from /media/rh/codes/sim/newton)
NumPy
SciPy
TensorBoard
```

### Shared with InterMask (from InterMask codebase):
- `models/vq/encdec.py` (Encoder, Decoder)
- `models/vq/residual_vq.py` (ResidualVQ)
- `models/vq/quantizer.py` (QuantizeEMAReset)
- `models/vq/resnet.py` (ResConvBlock)
- `models/losses.py` (Geometric_Losses)
- `prepare2/pd_utils.py` (PD controller, simulation utilities)
- `prepare2/retarget.py` (SMPL-X ↔ Newton mapping)
- `prepare2/gen_smpl_xml.py` (XML generation)

---

## 7. Known Issues / TODO

### Immediate

- [ ] Integration test: run full forward pass with small batch to verify gradient flow
- [ ] Verify `wp.Tape` actually propagates gradients through PD→Featherstone→positions
- [ ] Test `LightSAGEConv` weight loading from InterMask's `SAGEConv` checkpoints
- [ ] Profile GPU memory usage for 64-frame simulation

### Skyhook Bugs (from prepare2/notes.md)

- [ ] Bug #1: Smoothing on qd/qdd instead of joint_q before differentiation
- [ ] Bug #10: FPS/downsample mismatch between compute_torques and optimize_interaction
- [ ] Bug #11: Armature mismatch between inverse dynamics and optimization model

### Future Enhancements

- [ ] Phase 2: Full paired interaction training (simulate_pair integration)
- [ ] Gradient checkpointing for long sequences (>128 frames)
- [ ] Mixed-precision training (FP16 for encoder/decoder, FP32 for physics)
- [ ] Multi-GPU support (DDP with physics on one GPU)
- [ ] FID/matching score evaluation during training

---

## 8. Implementation Timeline

| Item | Status | File |
|------|--------|------|
| Module structure | ✅ Done | `newton_vqvae/__init__.py` |
| Configuration | ✅ Done | `newton_vqvae/config.py` |
| Data adapter | ✅ Done | `newton_vqvae/data_adapter.py` |
| Skeleton cache | ✅ Done | `newton_vqvae/skeleton_cache.py` |
| Newton bridge | ✅ Done | `newton_vqvae/newton_bridge.py` |
| Physics losses | ✅ Done | `newton_vqvae/physics_losses.py` |
| Model wrapper | ✅ Done | `newton_vqvae/model.py` |
| Training loop | ✅ Done | `newton_vqvae/train.py` |
| Evaluation | ✅ Done | `newton_vqvae/evaluate.py` |
| Shell scripts | ✅ Done | `scripts/train_phase1.sh`, `scripts/train_phase2.sh` |
| Documentation | ✅ Done | `NEWTON_VQVAE_DOC.md` |
| Integration test | ⬜ TODO | — |
| Gradient verification | ⬜ TODO | — |
| End-to-end training | ⬜ TODO | — |
