"""
optimize_tracking.py — Differentiable trajectory optimization for motion tracking.

Uses Newton's SolverFeatherstone (fully differentiable) to optimize per-frame
PD target residuals (Δq) that minimize tracking error while respecting physics.

Core idea:
    target_q = ref_q + Δq
    τ = ke_joint * (target_q - sim_q) - kd_joint * sim_qd
    L = Σ_t ||sim_pos(t) - ref_pos(t)||² + λ||Δq||²
    Backprop through physics → gradient on Δq → Adam step

This is trajectory optimization (Option A), not RL (Option B / PHC).
Each iteration improves tracking by finding better PD targets.

Usage:
    optimizer = TrackingOptimizer(viewer, args)
    # Called by newton.examples.run() which calls step()/render()/gui()
"""
import os
import sys
import time
import numpy as np

import warp as wp

wp.config.verbose = False

import newton
from newton import CollisionPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare5.phc_config import (
    FPS, DT, SIM_FREQ, SIM_SUBSTEPS,
    COORDS_PER_PERSON, DOFS_PER_PERSON, BODIES_PER_PERSON,
    SMPL_TO_NEWTON, N_SMPL_JOINTS,
    PHC_BODY_GAINS, TORQUE_LIMIT, ARMATURE_HINGE, ARMATURE_ROOT,
    DEFAULT_BODY_MASS_KG,
)


# ═══════════════════════════════════════════════════════════════
# Warp kernels (must be module-level for wp.Tape)
# ═══════════════════════════════════════════════════════════════

@wp.kernel
def pd_compose_kernel(
    state_q: wp.array(dtype=wp.float32),
    state_qd: wp.array(dtype=wp.float32),
    ref_coords_flat: wp.array(dtype=wp.float32),
    delta_q_flat: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    dof_frame_offset: int,
    coord_frame_offset: int,
    n_dof_pp: int,
    n_coord_pp: int,
    ke_root: float,
    kd_root: float,
    ke_joint: float,
    kd_joint: float,
    clamp_val: float,
    root_mode: int,
):
    """Compose PD torques for single-person tracking.

    Root DOFs (0-5): controlled by root_mode:
        0 = free (no root forces)
        1 = orient only (orientation PD, no position)
        2 = skyhook (full position + orientation PD)
    Joint DOFs (6-74): PD toward (ref_q + Δq), Δq is learnable.
    """
    tid = wp.tid()  # 0..n_dof_pp-1

    if tid < 3:
        # ── Root translation ─────────────────────────────
        if root_mode == 2:  # skyhook
            rc = tid
            rrc = coord_frame_offset + rc
            pos_err = ref_coords_flat[rrc] - state_q[rc]
            vel = state_qd[tid]
            out[tid] = wp.clamp(
                ke_root * pos_err - kd_root * vel, -clamp_val, clamp_val
            )
        else:
            out[tid] = 0.0
    elif tid < 6:
        # ── Root rotation (quaternion PD) ────────────────
        if root_mode >= 1:  # orient or skyhook
            qc = 3  # quat start in state_q
            rqc = coord_frame_offset + 3

            cx = state_q[qc]
            cy = state_q[qc + 1]
            cz = state_q[qc + 2]
            cw = state_q[qc + 3]

            rx = ref_coords_flat[rqc]
            ry = ref_coords_flat[rqc + 1]
            rz = ref_coords_flat[rqc + 2]
            rw = ref_coords_flat[rqc + 3]

            # q_err = q_ref * conj(q_cur)
            ew = rw * cw + rx * cx + ry * cy + rz * cz
            ex = -rw * cx + rx * cw + rz * cy - ry * cz
            ey = -rw * cy + ry * cw + rx * cz - rz * cx
            ez = -rw * cz + rz * cw + ry * cx - rx * cy

            sign = wp.where(ew >= 0.0, 1.0, -1.0)
            rot_idx = tid - 3
            err = 0.0
            if rot_idx == 0:
                err = 2.0 * sign * ex
            elif rot_idx == 1:
                err = 2.0 * sign * ey
            else:
                err = 2.0 * sign * ez

            vel = state_qd[tid]
            out[tid] = wp.clamp(
                ke_root * err - kd_root * vel, -clamp_val, clamp_val
            )
        else:
            out[tid] = 0.0
    else:
        # ── Joint DOFs: PD toward (ref + Δq) ────────────
        coord_idx = 7 + (tid - 6)  # skip 3 pos + 4 quat
        ref_idx = coord_frame_offset + coord_idx
        delta_idx = dof_frame_offset + tid

        target_q = ref_coords_flat[ref_idx] + delta_q_flat[delta_idx]
        pos_err = target_q - state_q[coord_idx]
        vel = state_qd[tid]
        pd_torque = ke_joint * pos_err - kd_joint * vel

        out[tid] = wp.clamp(pd_torque, -clamp_val, clamp_val)


@wp.kernel
def position_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    ref_pos: wp.array(dtype=wp.vec3),
    mapping: wp.array(dtype=wp.int32),
    n_joints: int,
    loss: wp.array(dtype=float),
):
    """MSE between simulated and reference body positions (single person)."""
    tid = wp.tid()  # 0..n_joints-1
    body_idx = mapping[tid]
    sim_pos = wp.transform_get_translation(body_q[body_idx])
    ref = ref_pos[tid]
    diff = sim_pos - ref
    wp.atomic_add(loss, 0, wp.dot(diff, diff) / wp.float32(n_joints))


@wp.kernel
def delta_reg_kernel(
    delta_flat: wp.array(dtype=wp.float32),
    weight: float,
    offset: int,
    n: int,
    loss: wp.array(dtype=float),
):
    """L2 regularization on Δq at given frame offset."""
    tid = wp.tid()
    val = delta_flat[offset + tid]
    wp.atomic_add(loss, 0, weight * val * val / wp.float32(n))


# ═══════════════════════════════════════════════════════════════
# Main optimizer class
# ═══════════════════════════════════════════════════════════════

class TrackingOptimizer:
    """Differentiable trajectory optimization for single-person tracking.

    Pattern matches Newton's diffsim examples:
      __init__  → build model, pre-allocate ALL arrays
      forward() → simulate window + compute loss (inside wp.Tape)
      step()    → forward + backward + Adam update on Δq
      render()  → visualize current best trajectory
      gui()     → side panel with loss/iteration info
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device if hasattr(args, "device") and args.device else "cuda:0"
        self.fps = getattr(args, 'fps', 20)
        self.frame = 0
        self._wall_start = None
        self.sim_time = 0.0

        # ── Optimization params ──────────────────────────────
        self.lr = getattr(args, 'lr', 1e-3)
        self.reg_lambda = getattr(args, 'reg_lambda', 1e-4)
        self.window_size = getattr(args, 'window', 10)
        self.n_epochs = getattr(args, 'epochs', 50)
        self.ke_joint = getattr(args, 'ke_joint', 200.0)
        self.kd_joint = getattr(args, 'kd_joint', 20.0)
        self.ke_root = getattr(args, 'ke_root', 5000.0)
        self.kd_root = getattr(args, 'kd_root', 500.0)

        # Root mode: 0=free, 1=orient, 2=skyhook
        rm = getattr(args, 'root_mode', 'free')
        self.root_mode_int = {"free": 0, "orient": 1, "skyhook": 2}[rm]

        self.train_iter = 0
        self.loss_history = []
        self.epoch = 0
        self.current_window = 0
        self._optimizing = True

        # ── Adam state ───────────────────────────────────────
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0
        self._adam_m = None
        self._adam_v = None

        # ── Simulation params ────────────────────────────────
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = 1.0 / SIM_FREQ

        # ── Load data ────────────────────────────────────────
        self._load_motion_data(args)

        # ── Build Newton model (differentiable) ──────────────
        self._build_model(args)

        # ── Pre-allocate all GPU arrays ──────────────────────
        self._preallocate(args)

        # ── Viewer setup ─────────────────────────────────────
        self.viewer.set_model(self.model)
        self._setup_camera()

        # ── Window tracking ──────────────────────────────────
        self.n_windows = max(1, (self.T + self.window_size - 1) // self.window_size)

        # ── Best result tracking ─────────────────────────────
        self._best_loss = float('inf')
        self._best_delta_np = None
        self._playback_jq = None  # filled after optimization

        print(f"\nReady: {self.T} frames, {self.n_windows} windows × "
              f"{self.window_size} frames, {self.n_epochs} epochs")
        print(f"  ke_joint={self.ke_joint}, kd_joint={self.kd_joint}, "
              f"root_mode={rm}, lr={self.lr}")

    # ─────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────
    def _load_motion_data(self, args):
        """Load and retarget motion data."""
        from prepare5.run_phc_tracker import load_clip, retarget_person

        clip_id = args.clip_id
        source = args.source

        print(f"Loading clip {clip_id} ({source})...")
        persons, text = load_clip(clip_id, source)
        self._text = text
        self._clip_id = clip_id
        self._source = source

        joint_q, betas = retarget_person(persons[0], source, device=self.device)
        self.ref_jq = joint_q.astype(np.float32)
        self.betas = betas
        self.T = self.ref_jq.shape[0]
        print(f"  {self.T} frames, {self.T / FPS:.1f}s")

    # ─────────────────────────────────────────────────────────
    # Model building
    # ─────────────────────────────────────────────────────────
    def _build_model(self, args):
        """Build single-person Newton model with Featherstone solver."""
        from prepare4.gen_xml import get_or_create_xml
        from prepare4.dynamics import set_segment_masses

        foot_geom = getattr(args, 'foot_geom', 'sphere')
        xml_path = get_or_create_xml(self.betas, foot_geom=foot_geom)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(
            device=self.device, requires_grad=True
        )

        set_segment_masses(self.model, total_mass=DEFAULT_BODY_MASS_KG,
                           verbose=False)

        # Disable passive springs
        self.model.mujoco.dof_passive_stiffness.fill_(0.0)
        self.model.mujoco.dof_passive_damping.fill_(0.0)
        self.model.joint_target_ke.fill_(0.0)
        self.model.joint_target_kd.fill_(0.0)

        # Disable joint limits (critical for generated data)
        n_dof = self.model.joint_dof_count
        self.model.joint_limit_lower = wp.array(
            np.full(n_dof, -1e6, dtype=np.float32),
            dtype=wp.float32, device=self.device
        )
        self.model.joint_limit_upper = wp.array(
            np.full(n_dof, 1e6, dtype=np.float32),
            dtype=wp.float32, device=self.device
        )

        # Armature for stability
        arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
        arm[:6] = ARMATURE_ROOT
        self.model.joint_armature = wp.array(
            arm, dtype=wp.float32, device=self.device
        )

        self.n_dof = n_dof
        self.n_coords = self.model.joint_coord_count

        # Featherstone solver (differentiable)
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # Collision pipeline
        self.collision_pipeline = CollisionPipeline(
            self.model, broad_phase="explicit", requires_grad=False,
        )
        self.contacts = self.collision_pipeline.contacts()

    # ─────────────────────────────────────────────────────────
    # Pre-allocation
    # ─────────────────────────────────────────────────────────
    def _preallocate(self, args):
        """Pre-allocate all GPU arrays. CRITICAL: nothing created inside tape."""
        # State chain for one window
        max_steps = self.window_size * self.sim_substeps + 1
        self.states = [self.model.state() for _ in range(max_steps)]

        # Control
        self.control = self.model.control()

        # Δq: learnable angle offsets, flat (T * n_dof,)
        self.delta_q = wp.array(
            np.zeros(self.T * self.n_dof, dtype=np.float32),
            dtype=wp.float32, device=self.device, requires_grad=True,
        )

        # Per-frame combined torque arrays (separate so tape retains values)
        self.combined_frames = [
            wp.zeros(self.n_dof, dtype=wp.float32,
                     device=self.device, requires_grad=True)
            for _ in range(self.window_size)
        ]

        # Reference coords flat (T * n_coords,) — constant
        ref_flat = self.ref_jq.flatten()
        # Pad to n_coords if model has extra coords (ground plane)
        if self.n_coords > COORDS_PER_PERSON:
            ref_padded = np.zeros((self.T, self.n_coords), dtype=np.float32)
            ref_padded[:, :COORDS_PER_PERSON] = self.ref_jq
            ref_flat = ref_padded.flatten()
        self.ref_coords_flat = wp.array(
            ref_flat, dtype=wp.float32, device=self.device,
        )

        # Reference positions via FK (T, 22, 3)
        self._compute_ref_positions()

        # Per-frame ref positions as wp.vec3 arrays
        self.ref_pos_wp = [
            wp.array(self.ref_positions[t], dtype=wp.vec3, device=self.device)
            for t in range(self.T)
        ]

        # SMPL → body index mapping
        smpl_body = np.array(
            [SMPL_TO_NEWTON[j] for j in range(N_SMPL_JOINTS)], dtype=np.int32
        )
        self.smpl_to_body_wp = wp.array(
            smpl_body, dtype=wp.int32, device=self.device
        )

        # Loss scalar
        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)

        # For playback: store simulated positions
        self._sim_positions = np.zeros((self.T, N_SMPL_JOINTS, 3), dtype=np.float32)
        self._sim_jq = np.zeros((self.T, COORDS_PER_PERSON), dtype=np.float32)

    def _compute_ref_positions(self):
        """Compute reference body positions for all frames via FK."""
        self.ref_positions = np.zeros((self.T, N_SMPL_JOINTS, 3), dtype=np.float32)
        state = self.model.state()
        jqd = wp.zeros(self.n_dof, dtype=wp.float32, device=self.device)

        for t in range(self.T):
            q = np.zeros(self.n_coords, dtype=np.float32)
            q[:COORDS_PER_PERSON] = self.ref_jq[t]
            state.joint_q = wp.array(q, dtype=wp.float32, device=self.device)
            newton.eval_fk(self.model, state.joint_q, jqd, state)
            body_q = state.body_q.numpy().reshape(-1, 7)
            for j, b in SMPL_TO_NEWTON.items():
                self.ref_positions[t, j] = body_q[b, :3]

    # ─────────────────────────────────────────────────────────
    # State helpers
    # ─────────────────────────────────────────────────────────
    def _set_state_from_ref(self, state, frame_idx):
        """Set state to reference pose at given frame (OUTSIDE tape)."""
        t = min(frame_idx, self.T - 1)
        q = np.zeros(self.n_coords, dtype=np.float32)
        q[:COORDS_PER_PERSON] = self.ref_jq[t]
        state.joint_q = wp.array(q, dtype=wp.float32, device=self.device)
        state.joint_qd = wp.zeros(self.n_dof, dtype=wp.float32,
                                   device=self.device)

    # ─────────────────────────────────────────────────────────
    # Forward simulation (called inside wp.Tape)
    # ─────────────────────────────────────────────────────────
    def forward(self, w_start, w_end):
        """Simulate one window and compute loss. Called inside wp.Tape."""
        step_idx = 0
        for f, t_frame in enumerate(range(w_start, w_end)):
            # Compose PD torques with learnable Δq
            dof_offset = t_frame * self.n_dof
            coord_offset = t_frame * self.n_coords
            wp.launch(
                pd_compose_kernel,
                dim=self.n_dof,
                inputs=[
                    self.states[step_idx].joint_q,
                    self.states[step_idx].joint_qd,
                    self.ref_coords_flat,
                    self.delta_q,
                    self.combined_frames[f],
                    dof_offset,
                    coord_offset,
                    DOFS_PER_PERSON,
                    COORDS_PER_PERSON,
                    self.ke_root,
                    self.kd_root,
                    self.ke_joint,
                    self.kd_joint,
                    TORQUE_LIMIT,
                    self.root_mode_int,
                ],
                device=self.device,
            )

            # Set control
            self.control.joint_f = self.combined_frames[f]

            # Physics substeps
            for _ in range(self.sim_substeps):
                self.solver.step(
                    self.states[step_idx],
                    self.states[step_idx + 1],
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )
                step_idx += 1

            # Position loss at this frame
            if t_frame < len(self.ref_pos_wp):
                wp.launch(
                    position_loss_kernel,
                    dim=N_SMPL_JOINTS,
                    inputs=[
                        self.states[step_idx].body_q,
                        self.ref_pos_wp[t_frame],
                        self.smpl_to_body_wp,
                        N_SMPL_JOINTS,
                        self.loss,
                    ],
                    device=self.device,
                )

        # L2 regularization on Δq
        for t_frame in range(w_start, w_end):
            offset = t_frame * self.n_dof
            wp.launch(
                delta_reg_kernel,
                dim=self.n_dof,
                inputs=[self.delta_q, self.reg_lambda,
                        offset, self.n_dof, self.loss],
                device=self.device,
            )

        self._last_step_idx = step_idx

    # ─────────────────────────────────────────────────────────
    # Optimization step
    # ─────────────────────────────────────────────────────────
    def _step_optimize(self):
        """One gradient-descent iteration on current window."""
        w_start = self.current_window * self.window_size
        w_end = min(w_start + self.window_size, self.T)

        # Init state from reference (OUTSIDE tape)
        self._set_state_from_ref(self.states[0], w_start)

        # Collision detection (OUTSIDE tape — non-differentiable)
        self.collision_pipeline.collide(self.states[0], self.contacts)

        # Zero loss
        self.loss.zero_()

        # Forward + backward
        tape = wp.Tape()
        with tape:
            self.forward(w_start, w_end)
        tape.backward(self.loss)

        loss_val = self.loss.numpy()[0]
        self.loss_history.append(loss_val)

        # Adam update on Δq
        grad_norm = 0.0
        if self.delta_q.grad is not None:
            g = self.delta_q.grad.numpy().reshape(self.T, self.n_dof)

            # Zero root DOF gradients (root handled by PD, not learned)
            g[:, :6] = 0.0

            # Only keep gradients for current window
            mask = np.zeros((self.T, 1), dtype=np.float32)
            mask[w_start:w_end] = 1.0
            g *= mask

            g_flat = g.flatten()
            grad_norm = float(np.linalg.norm(g_flat))

            # Gradient clipping
            if grad_norm > 100.0:
                g_flat *= 100.0 / grad_norm

            # Adam
            if self._adam_m is None:
                self._adam_m = np.zeros_like(g_flat)
                self._adam_v = np.zeros_like(g_flat)
            self.adam_t += 1
            self._adam_m = (self.adam_beta1 * self._adam_m
                           + (1.0 - self.adam_beta1) * g_flat)
            self._adam_v = (self.adam_beta2 * self._adam_v
                           + (1.0 - self.adam_beta2) * g_flat ** 2)
            m_hat = self._adam_m / (1.0 - self.adam_beta1 ** self.adam_t)
            v_hat = self._adam_v / (1.0 - self.adam_beta2 ** self.adam_t)

            delta_np = self.delta_q.numpy()
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            delta_np -= update
            self.delta_q = wp.array(
                delta_np.astype(np.float32),
                dtype=wp.float32, device=self.device,
                requires_grad=True,
            )

        tape.zero()

        # Advance window
        self.current_window += 1
        if self.current_window >= self.n_windows:
            self.current_window = 0
            self.epoch += 1

            # Track best
            epoch_loss = np.mean(self.loss_history[-self.n_windows:])
            if epoch_loss < self._best_loss:
                self._best_loss = epoch_loss
                self._best_delta_np = self.delta_q.numpy().copy()

            print(f"  === Epoch {self.epoch}/{self.n_epochs} complete | "
                  f"mean loss={epoch_loss:.6f} | best={self._best_loss:.6f}")

            if self.epoch >= self.n_epochs:
                self._optimizing = False
                self._finalize_optimization()

        self.train_iter += 1

        if self.train_iter % 5 == 1:
            dq = self.delta_q.numpy().reshape(self.T, self.n_dof)
            dq_joint = np.abs(dq[:, 6:])
            print(f"  iter {self.train_iter:4d} | epoch {self.epoch} "
                  f"win {self.current_window}/{self.n_windows} | "
                  f"loss={loss_val:.6f} | grad={grad_norm:.2e} | "
                  f"Δq={dq_joint.mean():.4f} (max={dq_joint.max():.4f})")

    def _finalize_optimization(self):
        """After optimization: run forward with best Δq to get final trajectory."""
        print(f"\n{'='*60}")
        print(f"Optimization complete! Running final forward pass...")
        print(f"{'='*60}")

        # Use best delta
        if self._best_delta_np is not None:
            delta_np = self._best_delta_np
        else:
            delta_np = self.delta_q.numpy()

        delta_2d = delta_np.reshape(self.T, self.n_dof)

        # Forward simulate with optimized Δq (no tape needed)
        state_0 = self.model.state()
        state_1 = self.model.state()
        control = self.model.control()

        # Use MuJoCo solver for final forward (better contacts)
        solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton",
            njmax=450, nconmax=150,
            impratio=10, iterations=100, ls_iterations=50,
        )

        # Init
        q0 = np.zeros(self.n_coords, dtype=np.float32)
        q0[:COORDS_PER_PERSON] = self.ref_jq[0]
        state_0.joint_q = wp.array(q0, dtype=wp.float32, device=self.device)
        state_0.joint_qd = wp.zeros(self.n_dof, dtype=wp.float32,
                                     device=self.device)
        newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)

        self._sim_jq[0] = self.ref_jq[0]

        # Extract initial positions
        body_q = state_0.body_q.numpy().reshape(-1, 7)
        for j, b in SMPL_TO_NEWTON.items():
            self._sim_positions[0, j] = body_q[b, :3]

        from scipy.spatial.transform import Rotation

        for t in range(1, self.T):
            ref_q = self.ref_jq[t]
            dq = delta_2d[t]

            for sub in range(self.sim_substeps):
                cq = state_0.joint_q.numpy()
                cqd = state_0.joint_qd.numpy()

                tau = np.zeros(self.n_dof, dtype=np.float32)

                # Root forces based on root_mode
                if self.root_mode_int == 2:  # skyhook
                    tau[:3] = (self.ke_root * (ref_q[:3] - cq[:3])
                               - self.kd_root * cqd[:3])
                if self.root_mode_int >= 1:  # orient or skyhook
                    q_cur = cq[3:7].copy()
                    qn = np.linalg.norm(q_cur)
                    if qn > 1e-8:
                        q_cur /= qn
                    R_err = (Rotation.from_quat(ref_q[3:7])
                             * Rotation.from_quat(q_cur).inv()).as_rotvec()
                    tau[3:6] = (self.ke_root * R_err
                                - self.kd_root * cqd[3:6])

                # Joint PD with optimized Δq
                target = ref_q[7:COORDS_PER_PERSON] + dq[6:DOFS_PER_PERSON]
                tau[6:] = (self.ke_joint * (target - cq[7:COORDS_PER_PERSON])
                           - self.kd_joint * cqd[6:DOFS_PER_PERSON])

                tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

                joint_f = np.zeros(self.n_dof, dtype=np.float32)
                joint_f[:DOFS_PER_PERSON] = tau
                control.joint_f = wp.array(
                    joint_f, dtype=wp.float32, device=self.device
                )

                state_0.clear_forces()
                contacts = self.model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, self.sim_dt)
                state_0, state_1 = state_1, state_0

            # Extract results
            newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd,
                          state_0)
            cq_np = state_0.joint_q.numpy()
            self._sim_jq[t] = cq_np[:COORDS_PER_PERSON]
            body_q = state_0.body_q.numpy().reshape(-1, 7)
            for j, b in SMPL_TO_NEWTON.items():
                self._sim_positions[t, j] = body_q[b, :3]

        # Compute MPJPE
        errors = np.linalg.norm(
            self._sim_positions - self.ref_positions, axis=-1
        )
        mpjpe_mm = errors.mean() * 1000
        print(f"  Final MPJPE: {mpjpe_mm:.1f} mm")
        print(f"  Max error: {errors.max() * 1000:.1f} mm")

        # Build playback trajectory
        self._playback_jq = self._sim_jq.copy()
        self._mpjpe_mm = mpjpe_mm

        # Save results
        output_dir = os.path.join(
            "output", "phc_tracker",
            f"clip_{self._clip_id}_{self._source}_optimized"
        )
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "optimized_result.npz"),
            sim_positions=self._sim_positions,
            ref_positions=self.ref_positions,
            sim_joint_q=self._sim_jq,
            delta_q=delta_2d,
        )
        print(f"  Saved to {output_dir}/")

    # ─────────────────────────────────────────────────────────
    # Newton examples interface
    # ─────────────────────────────────────────────────────────
    def step(self):
        if self._optimizing:
            self._step_optimize()
        else:
            # Playback mode: cycle through optimized trajectory
            now = time.perf_counter()
            if self._wall_start is None:
                self._wall_start = now
            elapsed = now - self._wall_start
            self.frame = int(elapsed * self.fps) % self.T
            self._set_playback_frame(self.frame)

    def _set_playback_frame(self, t):
        """Set viewer to show ref vs optimized side-by-side."""
        combined_q = np.zeros(self.n_coords, dtype=np.float32)

        if self._playback_jq is not None:
            combined_q[:COORDS_PER_PERSON] = self._playback_jq[
                min(t, self.T - 1)
            ]
        else:
            combined_q[:COORDS_PER_PERSON] = self.ref_jq[min(t, self.T - 1)]

        self.states[0].joint_q = wp.array(
            combined_q, dtype=wp.float32, device=self.device
        )
        jqd = wp.zeros(self.n_dof, dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, self.states[0].joint_q, jqd, self.states[0])

    def render(self):
        if self._optimizing:
            # Show last simulated state during optimization
            step_idx = getattr(self, '_last_step_idx', 0)
            state = self.states[min(step_idx, len(self.states) - 1)]
        else:
            state = self.states[0]

        self.sim_time = time.perf_counter() - (self._wall_start or time.perf_counter())
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(state)
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.separator()
        if self._optimizing:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.8, 0.2, 1.0),
                "[ OPTIMIZING ]"
            )
            imgui.text(f"Epoch:  {self.epoch}/{self.n_epochs}")
            imgui.text(f"Window: {self.current_window}/{self.n_windows}")
            imgui.text(f"Iter:   {self.train_iter}")
            if self.loss_history:
                imgui.text(f"Loss:   {self.loss_history[-1]:.6f}")
                imgui.text(f"Best:   {self._best_loss:.6f}")
        else:
            imgui.text_colored(
                imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
                "[ PLAYBACK — OPTIMIZED ]"
            )
            imgui.text(f"Frame: {self.frame}/{self.T - 1}")
            if hasattr(self, '_mpjpe_mm'):
                imgui.text(f"MPJPE: {self._mpjpe_mm:.1f} mm")

        imgui.separator()
        imgui.text(f"Clip:   {self._clip_id}")
        imgui.text(f"Source: {self._source}")
        imgui.text(f"Frames: {self.T}")
        if hasattr(self, '_text'):
            imgui.separator()
            imgui.text_wrapped(self._text)

    def _setup_camera(self):
        center = self.ref_jq[0, :3]
        cam_pos = wp.vec3(
            float(center[0]),
            float(center[1]) - 5.0,
            2.0,
        )
        self.viewer.set_camera(cam_pos, -15.0, 90.0)
