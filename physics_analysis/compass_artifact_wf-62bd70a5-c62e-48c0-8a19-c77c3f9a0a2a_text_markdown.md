# Sequential inverse kinematics for SMPL with Nvidia Warp

**Solving temporal IK for motion capture data requires jointly optimizing joint angles across frames with differentiable forward kinematics, temporal smoothness losses, and physical plausibility constraints.** Nvidia Warp's `wp.Tape()` autodiff mechanism provides the core gradient pipeline — differentiating through `eval_fk()` to compute ∂loss/∂joint_q — while SMPL's PyTorch-based FK chain enables integration with learned motion priors like VPoser and HuMoR. The practical recipe combines a multi-term loss function (position tracking + velocity/acceleration penalties + pose priors + contact constraints) with L-BFGS optimization over sliding windows of 16–120 frames, staged from coarse global alignment to fine articulation refinement.

## How SMPL forward kinematics creates the differentiable bridge

SMPL maps **72 pose parameters** (24 joints × 3 axis-angle) and **10 shape parameters** (β) to 6,890 mesh vertices and 24 joint positions through three differentiable stages. First, shape blend shapes displace the template mesh: `v_shaped = v_template + shapedirs @ beta`. Second, each axis-angle rotation θᵢ is converted to a 3×3 rotation matrix via Rodrigues' formula (`R = I + sin(α)·[k]× + (1-cos(α))·[k]×²`), and pose-dependent vertex corrections are added. Third, joint locations are regressed from the shaped mesh (`J = J_regressor @ v_shaped`), global transforms are computed recursively through the kinematic tree (pelvis → spine → chest → neck → head; pelvis → hip → knee → ankle), and linear blend skinning produces the final mesh.

The **entire pipeline is differentiable** in PyTorch via the `smplx` library. This means `loss.backward()` gives you ∂loss/∂pose_params directly:

```python
body_model = smplx.create('models/', model_type='smpl')
body_pose = torch.zeros(1, 69, requires_grad=True)
output = body_model(body_pose=body_pose, betas=betas)
loss = ((output.joints - target_joints)**2).sum()
loss.backward()  # body_pose.grad now contains the Jacobian-transpose direction
```

A critical design choice is **rotation representation**. SMPL natively uses axis-angle, but Zhou et al. (CVPR 2019) proved all representations in ≤4 dimensions are discontinuous in Euclidean space. The **6D continuous representation** — taking the first two columns of a rotation matrix and reconstructing via Gram-Schmidt — reduces optimization errors by 6–14× compared to quaternions or axis-angle. For gradient-based IK with good initialization (warm-starting from previous frames), axis-angle works adequately; for learning-based methods or large pose changes, the 6D representation is strongly preferred.

One important limitation: **SMPL uses ball-and-socket joints for ALL joints**, including knees and elbows that should be hinges. This allows biomechanically impossible configurations unless explicitly constrained. The MANIKIN model (ECCV 2024) addresses this by reducing DOFs at appropriate joints.

## Nvidia Warp's differentiable FK and the gradient-based IK pattern

Warp's IK approach uses three components: `ModelBuilder` to construct the articulated body, `eval_fk()` for differentiable forward kinematics, and `wp.Tape()` for reverse-mode autodiff. **Note that `warp.sim` was deprecated in v1.8 (July 2025)** and removed from documentation in v1.10, replaced by the Newton physics engine co-developed with Google DeepMind and Disney Research. The patterns below remain functional in Warp ≤1.9.x and translate conceptually to Newton.

The canonical IK example (`warp/examples/optim/example_inverse_kinematics.py`) demonstrates the complete pipeline:

```python
# 1. Build articulated chain
builder = wp.sim.ModelBuilder()
builder.add_articulation()
for i in range(chain_length):
    b = builder.add_body(origin=wp.transform([i, 0, 0], wp.quat_identity()), armature=0.1)
    builder.add_joint_revolute(
        parent=-1 if i == 0 else builder.joint_count - 1,
        child=b, axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform([chain_width, 0, 0], wp.quat_identity()),
        child_xform=wp.transform_identity(),
        limit_lower=-np.deg2rad(60.0), limit_upper=np.deg2rad(60.0))

model = builder.finalize(requires_grad=True)
state = model.state()

# 2. Forward + loss + backward
tape = wp.Tape()
with tape:
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state)
    wp.launch(compute_loss, dim=1, inputs=[state.body_q, end_effector_idx, loss])
tape.backward(loss=loss)

# 3. Gradient descent update
wp.launch(step_kernel, dim=len(model.joint_q),
          inputs=[model.joint_q, tape.gradients[model.joint_q], learning_rate])
tape.zero()
```

The gradient flow is: `joint_q → eval_fk → state.body_q (transforms) → loss kernel → scalar loss`, then `tape.backward()` computes ∂loss/∂joint_q automatically. Warp supports multiple joint types relevant for human bodies: **JOINT_REVOLUTE** (1 DOF, for elbows/knees), **JOINT_BALL** (3 DOF quaternion, for shoulders/hips), **JOINT_COMPOUND** (3 axes with per-axis limits), and **JOINT_FREE** (7 DOF, for root). Joint limits are specified per-axis via `limit_lower/limit_upper` parameters or `JointAxis` objects.

For PyTorch integration, Warp provides `wp.from_torch()`/`wp.to_torch()` zero-copy conversion, and the gradient can be wrapped in `torch.autograd.Function` to combine Warp FK with PyTorch-based losses and optimizers. One important caveat: always `.clone()` tensors before converting to prevent Warp from writing into PyTorch's gradient buffers.

## Temporal smoothness turns per-frame IK into trajectory optimization

Solving IK independently per frame — even with warm-starting from the previous solution — produces jitter and discontinuities when the optimizer jumps between local minima. The solution is to **optimize multiple frames jointly with temporal losses**. Three smoothness penalties form a hierarchy of increasing strictness:

**Velocity penalty** (first-order): `L_vel = Σ_t ||q_{t+1} - q_t||²` penalizes rapid joint angle changes. This is the minimum viable smoothness term, used in virtually all temporal motion optimization.

**Acceleration penalty** (second-order): `L_acc = Σ_t ||q_{t+1} - 2·q_t + q_{t-1}||²` penalizes changes in velocity, producing smoother transitions. CHOMP's primary smoothness metric uses this via a tridiagonal finite-difference matrix A where `L_acc = ξᵀ·Aᵀ·A·ξ`. In matrix form, the covariant gradient update `ξ ← ξ - η·A⁻¹·∇F` naturally produces smooth updates.

**Jerk penalty** (third-order): `L_jerk = Σ_t ||q_{t+2} - 3·q_{t+1} + 3·q_t - q_{t-1}||²` avoids abrupt acceleration changes. B-spline trajectory parameterization with 7th-order splines provides jerk continuity by construction.

Beyond finite-difference penalties, two powerful reparameterization strategies reduce the optimization dimensionality while enforcing smoothness by construction. **DCT (Discrete Cosine Transform) coefficients**: human motion concentrates energy in low-frequency modes, so optimizing K truncated DCT coefficients (K << T frames) inherently constrains the trajectory to be smooth. The DCT is a linear, invertible, differentiable operation. **B-spline control points**: the derivative of a B-spline is still a B-spline, joint limits can be enforced via the convex hull property, and N control points (N << T) parameterize a smooth trajectory.

**Gaussian Process trajectory priors** (from GPMP/GPMP2) provide a principled probabilistic framework. A GP prior generated by a linear time-varying SDE encodes smoothness: a constant-velocity prior encourages zero-acceleration trajectories, a constant-acceleration prior encourages zero-jerk. The GP prior cost is `F_gp = ½(ξ - μ)ᵀK⁻¹(ξ - μ)` where K is block-tridiagonal (exploiting Markov structure for O(N) inference). Sparse support states represent the trajectory, with dense GP interpolation filling in between — a natural coarse-to-fine approach.

## Learned motion priors constrain the solution to plausible human motion

Raw smoothness penalties don't prevent biomechanically impossible poses. Learned priors address this at two levels:

**VPoser** (per-frame pose prior) is a VAE trained on AMASS motion capture data that maps SMPL body pose (21 joints × 3 = 63D) into a **32-dimensional latent space** regularized to N(0,I). Instead of optimizing 63D pose parameters directly, you optimize in VPoser's latent space where `z ∈ ℝ³²` and `body_pose = vposer.decode(z)`. The prior loss is simply `L_prior = ||z||²`. This inherently constrains poses to the manifold of plausible human configurations and captures joint correlations. SMPLify-X replaced the earlier GMM prior with VPoser and achieved substantially better coverage of the valid pose space.

**HuMoR** (temporal motion prior) extends this to sequences via a conditional VAE that models the *distribution of state transitions* `p(xₜ | xₜ₋₁)` rather than static poses. The state includes root pose, joint angles, velocities, and contact labels. At test time, optimization variables are latent codes {zₜ} plus initial state x₀, with the prior `E_mot = -log p_θ(zₜ | xₜ₋₁)` keeping transitions on the manifold of plausible human dynamics. HuMoR's **3-stage optimization** is the gold standard pattern:

- **Stage 1** (30 steps): Optimize global translation/orientation only with strong regularization
- **Stage 2** (80 steps): Full pose optimization using VPoser prior + temporal smoothing (the "VPoser-t" baseline)  
- **Stage 3** (30 steps): Replace VPoser with full HuMoR CVAE motion prior, jointly optimizing latent codes across the sequence

More recent approaches include **MoManifold** (BMVC 2024), which models plausible motion as unsigned distance fields in acceleration space rather than VAE latent spaces, reportedly achieving lower errors than both VPoser-t and HuMoR across settings.

## Physical plausibility requires joint limits, ground contact, and collision avoidance

**Joint limits** are enforced through soft penalty terms (preferred for gradient-based optimization) or log-barrier functions. Typical human ranges include shoulder flexion 0–150°, elbow flexion 0–145°, knee flexion 0–140°, hip flexion 0–140°. In Warp, limits are set per-axis via `limit_lower`/`limit_upper` on `add_joint_revolute()` with `limit_ke`/`limit_kd` controlling enforcement stiffness. In PyTorch SMPL optimization, the standard pattern is:

```python
def joint_limit_loss(angles, lower, upper):
    violation = torch.clamp(lower - angles, min=0)**2 + torch.clamp(angles - upper, min=0)**2
    return violation.sum()
```

**Foot contact constraints** prevent the most visible artifact in motion synthesis — foot sliding. Detection uses height thresholds (< 2cm), velocity thresholds (< 0.01 m/frame), or neural networks trained on keypoint sequences. The anti-sliding loss penalizes horizontal foot velocity during detected contact: `L_foot = Σᵢ ||fᵢ ⊙ (FK(xⁱ) - FK(xⁱ⁻¹))||²` where fᵢ is a binary contact mask. Additional ground-plane terms penalize penetration (`relu(-min_vertex_height)²`) and floating (`min_vertex_height² × contact_mask`).

**Self-penetration avoidance** uses either capsule approximations (SMPLify's original approach), BVH-based triangle intersection detection (SMPLify-X), or the newer **VolumetricSMPL** which provides differentiable SDF-based collision queries as a plug-and-play SMPL extension:

```python
from VolumetricSMPL import attach_volume
attach_volume(body_model)
output = body_model(**smpl_data)
selfpen_loss = body_model.volume.selfpen_loss(output)
```

For full physics-based plausibility, **PhysDiff** (NVIDIA, ICCV 2023) projects denoised motion through Isaac Gym physics simulation at each diffusion step, while **PhysCap** uses a kinematic-then-physics two-stage pipeline with ground reaction forces for real-time plausible motion capture.

## The complete optimization loop in practice

The recommended loss function combines all terms with tuned weights:

```python
L_total = (w_pos * L_position          # 1.0 — match target keypoints
         + w_vel * L_velocity           # 0.1–10.0 — temporal smoothness (velocity)
         + w_acc * L_acceleration       # 0.1–1.0 — temporal smoothness (acceleration)
         + w_prior * L_pose_prior       # 0.001–0.1 — VPoser ||z||² or GMM
         + w_shape * L_shape            # 0.001–0.1 — ||β||²
         + w_limits * L_joint_limits    # 1.0–10.0 — penalty for violations
         + w_foot * L_foot_contact      # 1.0–5.0 — anti-sliding
         + w_ground * L_ground          # 1.0–10.0 — penetration prevention
         + w_coll * L_self_collision)   # 0.01–1.0 — interpenetration
```

**L-BFGS with Strong Wolfe line search** is the consensus optimizer for test-time fitting (used by SMPLify-X and HuMoR), converging in 30–80 iterations per stage. Adam (lr 1e-3) is preferred for batched training or when combining with neural network parameters. **Weight annealing** across stages is critical: start with strong priors (w_prior high) and gradually decrease while increasing data term influence.

For sequence processing, the practical pattern is:

- **Chunk into overlapping windows** of 16–120 frames (HuMoR found 16 frames sufficient for temporal context) with ~50% overlap
- **Blend overlapping regions** via linear interpolation to prevent boundary discontinuities  
- **Share shape parameters** (β) across all frames — optimize once globally
- **Warm-start each chunk** from the previous chunk's final state
- **Multi-stage optimization**: global alignment → full pose with VPoser → refinement with temporal/physics priors

For a Warp-native implementation, the pattern extends the basic IK example to multiple frames by constructing a batch of articulations (one per frame) in a single model, computing FK for all frames simultaneously, and adding temporal loss kernels that read adjacent frames' `body_q` transforms. Alternatively, the PyTorch path uses the `smplx` library for FK and Warp for specific GPU-accelerated components like collision queries or custom differentiable physics.

## Conclusion

The most effective architecture for temporal SMPL IK combines **PyTorch's smplx library** for differentiable forward kinematics with **VPoser or HuMoR** for motion priors and **L-BFGS** for optimization. Nvidia Warp adds value specifically for custom GPU-accelerated kernels (collision detection, parallel FK for robotics-style chains) and physics simulation, though its `warp.sim` module has been deprecated in favor of Newton. The key insight across all successful methods is that temporal IK is fundamentally a **trajectory optimization** problem, not a sequence of independent IK solves. Parameterizing the trajectory in a low-dimensional space — whether VPoser's 32D latent, HuMoR's transition latents, DCT coefficients, or B-spline control points — simultaneously reduces the optimization dimensionality and enforces smoothness by construction. The 3-stage optimization pattern (global → coarse pose → refined with priors) has proven robust across SMPLify-X, LEMO, and HuMoR, and should be the starting template for any new implementation.