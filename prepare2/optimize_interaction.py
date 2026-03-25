"""
optimize_interaction.py — Differentiable ΔT optimization for interaction torques.

Combines two-person forward simulation with gradient-descent optimization
of interaction torque deltas, following Newton's diffsim pattern.

Core idea:
    T_actual = T_solo + ΔT
    L = Σ_t ||sim_pos(t) - ref_pos(t)||² + λ||ΔT||²
    Backprop through physics → gradient on ΔT → optimizer step

Modes:
    --mode forward    Run forward-only sim with solo torques (no ΔT),
                      measure position error for diagnosis.
    --mode optimize   Run full optimization loop (forward + backward +
                      gradient update on ΔT).
    --mode playback   Load saved ΔT and replay with full torques.

Usage:
    # Diagnose: forward-only, see where errors are
    python prepare2/optimize_interaction.py --clip 1000 --mode forward

    # Optimize deltas (with live viewer)
    python prepare2/optimize_interaction.py --clip 1000 --mode optimize

    # Play back optimized result
    python prepare2/optimize_interaction.py --clip 1000 --mode playback

    # Headless batch optimization
    python prepare2/optimize_interaction.py --dataset interhuman --mode optimize
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples
from newton import CollisionPipeline

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import get_or_create_xml

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
DOFS_PER_PERSON = 75       # 6 root + 69 hinges
COORDS_PER_PERSON = 76     # 7 root (3 pos + 4 quat) + 69 hinges
N_BODIES_PER_PERSON = 24   # Pelvis + 23 child bodies
N_JOINTS = 22              # SMPL tracked joints

# SMPL joint → Newton body index (within one person)
SMPL_TO_BODY = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}


# ═══════════════════════════════════════════════════════════════
# Warp kernels
# ═══════════════════════════════════════════════════════════════
@wp.kernel
def compose_from_flat_kernel(
    solo_flat: wp.array(dtype=wp.float32),
    delta_flat: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    offset: int,
    clamp_val: float,
):
    """Compose torques: out[i] = clamp(solo_flat[offset+i] + delta_flat[offset+i])

    Reads directly from flat (T*n_dof,) arrays at the given frame offset.
    No intermediate copies needed — tape can backprop through delta_flat directly.
    """
    tid = wp.tid()
    idx = offset + tid
    out[tid] = wp.clamp(
        solo_flat[idx] + delta_flat[idx], -clamp_val, clamp_val
    )


@wp.kernel
def compose_with_root_pd_kernel(
    state_q: wp.array(dtype=wp.float32),
    state_qd: wp.array(dtype=wp.float32),
    ref_coords_flat: wp.array(dtype=wp.float32),
    solo_flat: wp.array(dtype=wp.float32),
    delta_flat: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    dof_frame_offset: int,
    coord_frame_offset: int,
    n_dof_pp: int,
    n_coord_pp: int,
    n_persons: int,
    ke_root: float,
    kd_root: float,
    ke_joint: float,
    kd_joint: float,
    clamp_val: float,
):
    """Compose torques with full-body PD for ALL persons (joint method).

    Root DOFs (0-5) per person: PD toward reference pose.
    Joint DOFs (6-74) per person: PD tracking + solo + delta (learnable).
    """
    tid = wp.tid()  # 0..n_dof_total-1

    # Which person and local DOF?
    person = tid // n_dof_pp
    local = tid - person * n_dof_pp

    if person >= n_persons:
        out[tid] = 0.0
        return

    if local < 3:
        # ── Root translation PD ──────────────────────
        coord_start = person * n_coord_pp
        rc = coord_start + local
        rrc = coord_frame_offset + rc
        pos_err = ref_coords_flat[rrc] - state_q[rc]
        vel = state_qd[tid]
        out[tid] = wp.clamp(
            ke_root * pos_err - kd_root * vel, -clamp_val, clamp_val
        )
    elif local < 6:
        # ── Root rotation PD (quaternion) ────────────
        coord_start = person * n_coord_pp
        qc = coord_start + 3        # quat start in state_q
        rqc = coord_frame_offset + qc  # quat start in ref

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

        rot_idx = local - 3
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
        # ── Joint DOFs: PD tracking toward (ref + delta_q) + solo ─
        # delta_flat stores Δq (position offsets, NOT torque offsets).
        # PD amplifies delta_q by ke_joint, giving 200x stronger
        # signal than direct torque deltas.
        coord_start = person * n_coord_pp
        coord_idx = coord_start + 7 + (local - 6)
        ref_idx = coord_frame_offset + coord_idx

        idx = dof_frame_offset + tid
        ref_q = ref_coords_flat[ref_idx] + delta_flat[idx]

        pos_err = ref_q - state_q[coord_idx]
        vel = state_qd[tid]
        pd_torque = ke_joint * pos_err - kd_joint * vel

        out[tid] = wp.clamp(
            pd_torque + solo_flat[idx],
            -clamp_val, clamp_val
        )


@wp.kernel
def position_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    ref_pos: wp.array(dtype=wp.vec3),
    mapping: wp.array(dtype=wp.int32),
    n_persons: int,
    n_bodies_pp: int,
    n_joints: int,
    loss: wp.array(dtype=float),
):
    """MSE between simulated and reference body positions."""
    tid = wp.tid()
    person = tid // n_joints
    joint = tid % n_joints
    body_idx = person * n_bodies_pp + mapping[joint]
    sim_pos = wp.transform_get_translation(body_q[body_idx])
    ref = ref_pos[tid]
    diff = sim_pos - ref
    wp.atomic_add(loss, 0, wp.dot(diff, diff) / wp.float32(n_persons * n_joints))


@wp.kernel
def delta_reg_from_flat_kernel(
    delta_flat: wp.array(dtype=wp.float32),
    weight: float,
    offset: int,
    n: int,
    loss: wp.array(dtype=float),
):
    """L2 regularization on delta torques at given frame offset."""
    tid = wp.tid()
    val = delta_flat[offset + tid]
    wp.atomic_add(loss, 0, weight * val * val / wp.float32(n))


@wp.kernel
def gradient_step_kernel(
    param: wp.array(dtype=wp.float32),
    grad: wp.array(dtype=wp.float32),
    lr: float,
):
    """param -= lr * grad"""
    tid = wp.tid()
    param[tid] = param[tid] - lr * grad[tid]


# ───────────────────────────────────────────────────────────────
# Kernels for alternating optimization
# ───────────────────────────────────────────────────────────────
@wp.kernel
def copy_dof_values_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    src_offset: int,
):
    """Copy a frame's DOF values from flat (T*n_dof) array to (n_dof) array."""
    tid = wp.tid()
    dst[tid] = src[src_offset + tid]


@wp.kernel
def alternating_compose_kernel(
    state_q: wp.array(dtype=wp.float32),
    state_qd: wp.array(dtype=wp.float32),
    ref_coords_flat: wp.array(dtype=wp.float32),
    solo_flat: wp.array(dtype=wp.float32),
    delta_flat: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    dof_frame_offset: int,
    coord_frame_offset: int,
    fixed_dof_start: int,
    fixed_coord_start: int,
    n_dof_pp: int,
    ke_root: float,
    kd_root: float,
    clamp_val: float,
):
    """Compose torques for alternating optimization.

    Fixed person's root (DOFs 0-5): PD toward reference via quaternion error.
    Fixed person's hinges (DOFs 6-74): 0 — solver's built-in PD handles it
        via model.joint_target_ke/kd + control.joint_target_pos.
    Free person's DOFs (all 75): solo + delta (learnable).
    """
    tid = wp.tid()  # 0..n_dof_total-1

    fixed_local = tid - fixed_dof_start

    if fixed_local >= 0 and fixed_local < n_dof_pp:
        # ── Fixed person DOF ────────────────────────────────
        if fixed_local < 3:
            # Root translation: PD on xyz position
            rc = fixed_coord_start + fixed_local
            rrc = coord_frame_offset + rc
            pos_err = ref_coords_flat[rrc] - state_q[rc]
            vel = state_qd[tid]
            out[tid] = wp.clamp(
                ke_root * pos_err - kd_root * vel, -clamp_val, clamp_val
            )
        elif fixed_local < 6:
            # Root rotation: quaternion PD
            qc = fixed_coord_start + 3       # quat start in state_q
            rqc = coord_frame_offset + qc    # quat start in ref

            # Current quaternion (xyzw)
            cx = state_q[qc]
            cy = state_q[qc + 1]
            cz = state_q[qc + 2]
            cw = state_q[qc + 3]
            # Reference quaternion
            rx = ref_coords_flat[rqc]
            ry = ref_coords_flat[rqc + 1]
            rz = ref_coords_flat[rqc + 2]
            rw = ref_coords_flat[rqc + 3]

            # q_err = q_ref * conj(q_cur), xyzw Hamilton product
            ew = rw * cw + rx * cx + ry * cy + rz * cz
            ex = -rw * cx + rx * cw + rz * cy - ry * cz
            ey = -rw * cy + ry * cw + rx * cz - rz * cx
            ez = -rw * cz + rz * cw + ry * cx - rx * cy

            # Shortest path
            sign = wp.where(ew >= 0.0, 1.0, -1.0)

            # Axis-angle error ≈ 2 * sign * (ex, ey, ez)
            rot_idx = fixed_local - 3
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
            # Hinge DOF: set 0 — built-in PD via ke/kd handles this
            out[tid] = 0.0
    else:
        # ── Free person DOF: solo + delta ────────────────────
        idx = dof_frame_offset + tid
        out[tid] = wp.clamp(
            solo_flat[idx] + delta_flat[idx], -clamp_val, clamp_val
        )


# ═══════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════
class InteractionOptimizer:
    """
    Differentiable two-person physics simulation for ΔT optimization.

    Follows the Newton diffsim pattern:
      __init__  → build model, pre-allocate ALL arrays
      forward() → simulate window + compute loss (inside wp.Tape)
      step()    → forward + backward + gradient update
      render()  → visualize current trajectory
      gui()     → side panel info

    Key design decisions matching diffsim examples:
      - SolverFeatherstone: works in generalized coordinates (joint_q/joint_qd),
        handles free joints natively, proven stable with our MJCF model.
        All internal operations are wp.launch → fully differentiable.
      - Pre-allocated state chain: states[0]→...→states[N], no double-buffering.
      - Per-frame combined torque arrays: needed so the tape can backprop
        correctly when different control signals are used at different frames.
      - No array creation inside the tape: all arrays pre-allocated in __init__.
      - State initialization outside the tape: reference poses are constant.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device if hasattr(args, "device") and args.device else "cuda:0"
        self.fps = args.fps
        self.downsample = getattr(args, 'downsample', 1)
        self.mode = args.mode
        self.frame = 0
        self._wall_start = None
        self.sim_time = 0.0

        # ── Optimization params ──────────────────────────────
        self.lr = args.lr
        self.reg_lambda = args.reg_lambda
        self.window_size = args.window
        self.torque_limit = 5000.0    # Must exceed max solo torque magnitude
        self.train_iter = 0
        self.loss_history = []
        self._clip_id = args.clip

        # ── Adam optimizer state ─────────────────────────────
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0
        # m and v will be initialized on first gradient step
        self._adam_m = None
        self._adam_v = None

        # ── Simulation params ────────────────────────────────
        self.sim_freq = getattr(args, 'sim_freq', 480)
        self.sim_substeps = self.sim_freq // self.fps
        self.sim_dt = 1.0 / self.sim_freq

        data_dir = args.data_dir
        clip_id = args.clip

        # ── Load data for both persons ───────────────────────
        self.all_ref_jq = []
        self.all_torques_solo = []
        self.all_qvel = []
        self.all_xml_paths = []
        self.person_indices = []

        for p_idx in [0, 1]:
            jq_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
            betas_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_betas.npy")
            torques_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_torques_solo.npy")
            qvel_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_qvel.npy")

            # Fallback: look in compute_torques dir if torques not in data_dir
            if not os.path.exists(torques_path):
                ct_dir = data_dir.replace("retargeted_v2", "compute_torques")
                alt = os.path.join(ct_dir, f"{clip_id}_person{p_idx}_torques_solo.npy")
                if os.path.exists(alt):
                    torques_path = alt
                    alt_qv = os.path.join(ct_dir, f"{clip_id}_person{p_idx}_qvel.npy")
                    if os.path.exists(alt_qv):
                        qvel_path = alt_qv

            needed = [jq_path, betas_path, torques_path]
            if not all(os.path.exists(p) for p in needed):
                continue

            jq = np.load(jq_path).astype(np.float32)
            betas = np.load(betas_path)
            torques = np.load(torques_path).astype(np.float32)
            qvel = (np.load(qvel_path).astype(np.float32)
                    if os.path.exists(qvel_path)
                    else np.zeros_like(torques))
            xml_path = get_or_create_xml(betas)

            # Downsample from data FPS to target FPS
            ds = self.downsample
            if ds > 1:
                jq = jq[::ds]
                torques = torques[::ds]
                qvel = qvel[::ds]

            self.all_ref_jq.append(jq)
            self.all_torques_solo.append(torques)
            self.all_qvel.append(qvel)
            self.all_xml_paths.append(xml_path)
            self.person_indices.append(p_idx)
            print(f"Loaded person{p_idx}: {jq.shape[0]} frames, "
                  f"torques range=[{torques.min():.0f}, {torques.max():.0f}]")

        if len(self.person_indices) < 2:
            raise FileNotFoundError(
                f"Need both persons. Found {self.person_indices} for clip {clip_id}.\n"
                f"Run: python prepare2/compute_torques.py --clip {clip_id} "
                f"--method inverse --save")

        self.n_persons = 2
        self.T = min(
            min(jq.shape[0] for jq in self.all_ref_jq),
            min(t.shape[0] for t in self.all_torques_solo),
        )

        # ── Load reference positions for loss ────────────────
        self.ref_positions = self._load_ref_positions(data_dir, clip_id)
        if self.ref_positions is not None:
            self.T = min(self.T, self.ref_positions.shape[0])

        print(f"Clip {clip_id}: {self.n_persons} persons, "
              f"{self.T} frames, FPS={self.fps}")

        # ── Build two-person Newton model ────────────────────
        use_grad = (self.mode == "optimize")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for xml_path in self.all_xml_paths:
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(
            device=self.device, requires_grad=use_grad
        )

        self.n_dof = self.model.joint_dof_count     # 150
        self.n_coords = self.model.joint_coord_count  # 152

        # ── Disable passive springs ──────────────────────────
        self.model.mujoco.dof_passive_stiffness.fill_(0.0)
        self.model.mujoco.dof_passive_damping.fill_(0.0)
        self.model.joint_target_ke.fill_(0.0)
        self.model.joint_target_kd.fill_(0.0)

        # ── Armature for stability ───────────────────────────
        arm = np.full(self.n_dof, 0.5, dtype=np.float32)
        for i in range(self.n_persons):
            off = i * DOFS_PER_PERSON
            arm[off:off + 6] = 5.0  # Higher armature for root DOFs
        self.model.joint_armature = wp.array(
            arm, dtype=wp.float32, device=self.device
        )

        # ── Solver: SolverFeatherstone ───────────────────────
        # Featherstone works in generalized coordinates (joint_q/joint_qd),
        # handles free joints natively, all internal ops are wp.launch
        # → fully differentiable through wp.Tape.
        # (SolverSemiImplicit works in maximal coords → unstable for
        #  articulated MJCF bodies, causes NaN.)
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        # ── Collision pipeline ───────────────────────────────
        # Use requires_grad=False because the narrow-phase kernels
        # are NOT differentiable (enable_backward=False). However,
        # Collision detection runs OUTSIDE the tape (one-shot at window
        # start). However, the solver's eval_body_contact kernel runs
        # INSIDE the tape and references contact arrays (stiffness,
        # damping, friction). With requires_grad=False, these are None
        # and crash tape.backward(). So we use requires_grad=True to
        # allocate proper arrays, even though collide() is outside tape.
        self.collision_pipeline = CollisionPipeline(
            self.model, broad_phase="explicit", requires_grad=True,
        )
        self.contacts = self.collision_pipeline.contacts()

        # ── Pre-allocate state chain for one window ──────────
        # Pattern: states[0]→states[1]→...→states[N]
        # solver.step(states[t], states[t+1], ...)
        max_steps = self.window_size * self.sim_substeps + 1
        self.states = [self.model.state() for _ in range(max_steps)]

        # ── Control (re-used, but joint_f switched per frame) ─
        self.control = self.model.control()

        # ── Pre-allocate GPU arrays ──────────────────────────
        # (CRITICAL: no array creation allowed inside wp.Tape)

        # Solo torques: flat (T * n_dof,) — constant, no grad
        # CRITICAL: Zero out root DOFs (0-5) for each person.
        # I.D. root forces are virtual (not real actuators) and
        # would destabilize forward simulation.  Root tracking
        # is handled by explicit PD in the compose kernels.
        solo_flat = np.zeros((self.T, self.n_dof), dtype=np.float32)
        for i in range(self.n_persons):
            d = i * DOFS_PER_PERSON
            solo_flat[:, d:d + DOFS_PER_PERSON] = (
                self.all_torques_solo[i][:self.T]
            )
            # Zero root virtual forces — only keep joint torques
            solo_flat[:, d:d + 6] = 0.0
        self.solo_all = wp.array(
            solo_flat.flatten(), dtype=wp.float32, device=self.device
        )
        # Keep numpy copy for forward/playback mode
        self.solo_np = solo_flat

        # Delta torques: flat (T * n_dof,) — learnable, requires_grad
        if self.mode == "playback":
            delta_init = self._load_deltas_np(data_dir, clip_id)
        else:
            delta_init = np.zeros((self.T, self.n_dof), dtype=np.float32)
        self.delta_all = wp.array(
            delta_init.flatten(), dtype=wp.float32,
            device=self.device, requires_grad=use_grad,
        )

        # Per-frame combined torques: separate array per frame
        # so the tape retains correct values for backward pass.
        self.combined_frames = [
            wp.zeros(self.n_dof, dtype=wp.float32,
                     device=self.device, requires_grad=use_grad)
            for _ in range(self.window_size)
        ]

        # Reference positions per frame as wp.vec3 arrays
        if self.ref_positions is not None:
            # ref_positions: (T, 2, 22, 3)
            ref_flat = self.ref_positions.reshape(self.T, -1, 3)
            self.ref_pos_wp = [
                wp.array(ref_flat[t], dtype=wp.vec3, device=self.device)
                for t in range(self.T)
            ]
        else:
            self.ref_pos_wp = None

        # SMPL → body index mapping for loss kernel
        smpl_body = np.array(
            [SMPL_TO_BODY[j] for j in range(N_JOINTS)], dtype=np.int32
        )
        self.smpl_to_body_wp = wp.array(
            smpl_body, dtype=wp.int32, device=self.device
        )

        # Loss scalar
        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)

        # ── Root PD for gravity compensation ─────────────────
        # I.D. root virtual forces are zeroed above; root PD keeps
        # the pelvis near the reference while joints do the real work.
        self.ke_root = getattr(args, 'ke_root', 5000.0)
        self.kd_root = getattr(args, 'kd_root', 500.0)

        # ── Joint PD for tracking ────────────────────────────
        # Compensates for solver mismatch (I.D. computed with MuJoCo
        # but simulated with Newton/Featherstone). Without this, the
        # solo torques alone produce ~37cm RMSE.
        self.ke_joint = getattr(args, 'ke_joint', 200.0)
        self.kd_joint = getattr(args, 'kd_joint', 20.0)

        # Pre-compute reference trajectory in coord space for root PD
        # (used by both joint and alternating methods)
        ref_coords = np.zeros((self.T, self.n_coords), dtype=np.float32)
        for i, jq in enumerate(self.all_ref_jq):
            c = i * COORDS_PER_PERSON
            ref_coords[:, c:c + COORDS_PER_PERSON] = jq[:self.T]
        self.ref_coords_flat = wp.array(
            ref_coords.flatten(), dtype=wp.float32, device=self.device,
        )
        # Numpy copy for forward/playback CPU PD
        self._ref_coords_np = ref_coords

        # ── Window tracking ──────────────────────────────────
        self.n_windows = max(1, (self.T + self.window_size - 1) // self.window_size)
        self.current_window = 0
        self.epoch = 0
        self._last_step_idx = 0

        # ── For forward/playback: double-buffer states ───────
        self.state_fwd_0 = self.model.state()
        self.state_fwd_1 = self.model.state()
        self._render_frame = 0
        self._fwd_initialized = False

        # ── Viewer ───────────────────────────────────────────
        self.viewer.set_model(self.model)
        self._setup_camera()

        # ── Alternating method setup ─────────────────────────
        self.opt_method = getattr(args, 'method', 'joint')
        if self.opt_method == 'alternating':
            self._setup_alternating(args)

        method_str = f"method={self.opt_method}, " if self.mode == "optimize" else ""
        print(f"Ready: mode={self.mode}, {method_str}solver=Featherstone, "
              f"n_dof={self.n_dof}, windows={self.n_windows}, "
              f"substeps={self.sim_substeps}")

    # ─────────────────────────────────────────────────────────
    # Data helpers
    # ─────────────────────────────────────────────────────────
    def _load_ref_positions(self, data_dir, clip_id):
        """Load (T, 2, 22, 3) reference positions from per-person files."""
        parts = []
        for p_idx in self.person_indices:
            path = os.path.join(data_dir, f"{clip_id}_person{p_idx}.npy")
            if os.path.exists(path):
                pos = np.load(path)
                if self.downsample > 1:
                    pos = pos[::self.downsample]
                parts.append(pos)
        if len(parts) < 2:
            return None
        T = min(p.shape[0] for p in parts)
        return np.stack([p[:T] for p in parts], axis=1).astype(np.float32)

    def _load_deltas_np(self, data_dir, clip_id):
        """Load previously saved ΔT for playback."""
        delta = np.zeros((self.T, self.n_dof), dtype=np.float32)
        for i, p_idx in enumerate(self.person_indices):
            path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_delta_torques.npy"
            )
            if os.path.exists(path):
                d = np.load(path).astype(np.float32)
                off = i * DOFS_PER_PERSON
                T_ = min(d.shape[0], self.T)
                delta[:T_, off:off + DOFS_PER_PERSON] = d[:T_]
                print(f"Loaded ΔT for person{p_idx}: {d.shape}")
            else:
                print(f"WARNING: {path} not found, using zero ΔT")
        return delta

    def _set_state_from_ref(self, state, frame_idx):
        """Reset a state to the reference trajectory at given frame."""
        q = np.zeros(self.n_coords, dtype=np.float32)
        qd = np.zeros(self.n_dof, dtype=np.float32)
        for i, jq in enumerate(self.all_ref_jq):
            c = i * COORDS_PER_PERSON
            d = i * DOFS_PER_PERSON
            t = min(frame_idx, jq.shape[0] - 1)
            q[c:c + COORDS_PER_PERSON] = jq[t]
            t_v = min(t, self.all_qvel[i].shape[0] - 1)
            qd[d:d + DOFS_PER_PERSON] = self.all_qvel[i][t_v]

        state.joint_q = wp.array(
            q, dtype=wp.float32, device=self.device
        )
        state.joint_qd = wp.array(
            qd, dtype=wp.float32, device=self.device
        )
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def _setup_camera(self):
        centers = [jq[0, :3] for jq in self.all_ref_jq]
        center = np.mean(centers, axis=0).astype(float)
        cam_pos = wp.vec3(
            float(center[0]), float(center[1]) - 5.0, 2.0
        )
        self.viewer.set_camera(cam_pos, -15.0, 90.0)

    # ─────────────────────────────────────────────────────────
    # Alternating method setup
    # ─────────────────────────────────────────────────────────
    def _setup_alternating(self, args):
        """Pre-allocate arrays for alternating optimization.

        In alternating mode, one person is position-controlled (fixed)
        while the other's torques are optimized, then they swap.

        Fixed person:
          - Hinge DOFs: Newton's built-in PD via joint_target_ke/kd
            + control.joint_target_pos (differentiable, inside tape)
          - Root DOFs: Custom PD via control.joint_f using
            alternating_compose_kernel (FREE joints skip built-in PD)

        Free person:
          - All DOFs: solo + delta via control.joint_f
        """
        # PD gains for fixed person (very stiff tracking)
        self.fixed_ke_hinge = getattr(args, 'fixed_ke', 5000.0)
        self.fixed_kd_hinge = getattr(args, 'fixed_kd', 500.0)
        self.fixed_ke_root = getattr(args, 'fixed_ke_root', 50000.0)
        self.fixed_kd_root = getattr(args, 'fixed_kd_root', 5000.0)

        # Phase tracking
        self.free_person = 0          # start by optimizing person 0
        self.phase_epochs = getattr(args, 'phase_epochs', 1)
        self._prev_phase = -1         # force initial gain setup

        # Pre-compute reference trajectory in DOF space for target_pos
        # (maps joint_q coords → DOF positions for hinge joints)
        ref_target_pos = np.zeros((self.T, self.n_dof), dtype=np.float32)
        for p in range(self.n_persons):
            d_off = p * DOFS_PER_PERSON + 6  # first hinge DOF
            ref_jq = self.all_ref_jq[p][:self.T]
            ref_target_pos[:, d_off:d_off + 69] = ref_jq[:, 7:76]
        self.ref_target_flat = wp.array(
            ref_target_pos.flatten(), dtype=wp.float32, device=self.device,
        )

        # ref_coords_flat already created in __init__ for root PD

        # Per-frame target_pos arrays (one per window frame, pre-allocated)
        # Needed so the tape retains correct values per frame.
        self.target_pos_frames = [
            wp.zeros(self.n_dof, dtype=wp.float32, device=self.device)
            for _ in range(self.window_size)
        ]

        # Pre-computed PD gain arrays for each phase
        # (swap model.joint_target_ke/kd when switching fixed person)
        self._ke_phase = {}
        self._kd_phase = {}
        for free_p in [0, 1]:
            fixed_p = 1 - free_p
            ke = np.zeros(self.n_dof, dtype=np.float32)
            kd = np.zeros(self.n_dof, dtype=np.float32)
            fd = fixed_p * DOFS_PER_PERSON
            ke[fd + 6:fd + DOFS_PER_PERSON] = self.fixed_ke_hinge
            kd[fd + 6:fd + DOFS_PER_PERSON] = self.fixed_kd_hinge
            self._ke_phase[free_p] = wp.array(
                ke, dtype=wp.float32, device=self.device,
            )
            self._kd_phase[free_p] = wp.array(
                kd, dtype=wp.float32, device=self.device,
            )

        print(f"  Alternating: phase_epochs={self.phase_epochs}, "
              f"fixed_ke={self.fixed_ke_hinge}, fixed_kd={self.fixed_kd_hinge}, "
              f"root_ke={self.fixed_ke_root}, root_kd={self.fixed_kd_root}")

    def _set_phase_gains(self, free_person):
        """Swap model PD gains for the current alternating phase."""
        self.model.joint_target_ke = self._ke_phase[free_person]
        self.model.joint_target_kd = self._kd_phase[free_person]

    def _extract_body_positions(self, state):
        """Extract (n_persons, 22, 3) body positions from state."""
        body_q = state.body_q.numpy().reshape(-1, 7)
        pos = np.zeros((self.n_persons, N_JOINTS, 3), dtype=np.float32)
        for p in range(self.n_persons):
            off = p * N_BODIES_PER_PERSON
            for smpl_j, body_idx in SMPL_TO_BODY.items():
                pos[p, smpl_j] = body_q[off + body_idx, :3]
        return pos

    @property
    def delta_np(self):
        """Current delta torques as numpy (T, n_dof)."""
        return self.delta_all.numpy().reshape(self.T, self.n_dof)

    # ─────────────────────────────────────────────────────────
    # Forward pass (inside wp.Tape)
    # ─────────────────────────────────────────────────────────
    def forward(self, w_start, w_end):
        """
        Simulate one window and compute loss.

        MUST be called inside wp.Tape for optimization mode.
        All GPU ops are Warp kernel launches — no Python array
        creation inside the tape.

        State initialization (states[0]) must be done BEFORE
        entering the tape scope.
        """
        step_idx = 0
        for f, t_frame in enumerate(range(w_start, w_end)):
            # ── Compose torques: root PD + (solo+delta) joints ─
            # Root DOFs get PD toward reference (gravity compensation).
            # Joint DOFs get solo I.D. + learnable delta.
            # Each frame gets its own combined array so the tape
            # retains the correct values for backward pass.
            dof_offset = t_frame * self.n_dof
            coord_offset = t_frame * self.n_coords
            wp.launch(
                compose_with_root_pd_kernel,
                dim=self.n_dof,
                inputs=[
                    self.states[step_idx].joint_q,
                    self.states[step_idx].joint_qd,
                    self.ref_coords_flat,
                    self.solo_all, self.delta_all,
                    self.combined_frames[f],
                    dof_offset,
                    coord_offset,
                    DOFS_PER_PERSON,
                    COORDS_PER_PERSON,
                    self.n_persons,
                    self.ke_root,
                    self.kd_root,
                    self.ke_joint,
                    self.kd_joint,
                    self.torque_limit,
                ],
                device=self.device,
            )

            # ── Set control ──────────────────────────────────
            self.control.joint_f = self.combined_frames[f]

            # ── Physics substeps ─────────────────────────────
            # Collision detection done once outside tape (in _step_optimize).
            for _ in range(self.sim_substeps):
                self.solver.step(
                    self.states[step_idx],
                    self.states[step_idx + 1],
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )
                step_idx += 1

            # ── Position loss at this control frame ──────────
            if self.ref_pos_wp is not None and t_frame < len(self.ref_pos_wp):
                wp.launch(
                    position_loss_kernel,
                    dim=self.n_persons * N_JOINTS,
                    inputs=[
                        self.states[step_idx].body_q,
                        self.ref_pos_wp[t_frame],
                        self.smpl_to_body_wp,
                        self.n_persons,
                        N_BODIES_PER_PERSON,
                        N_JOINTS,
                        self.loss,
                    ],
                    device=self.device,
                )

        # ── L2 regularization on delta ───────────────────────
        # Reads directly from delta_all (no copy needed)
        for t_frame in range(w_start, w_end):
            offset = t_frame * self.n_dof
            wp.launch(
                delta_reg_from_flat_kernel,
                dim=self.n_dof,
                inputs=[self.delta_all, self.reg_lambda,
                        offset, self.n_dof, self.loss],
                device=self.device,
            )

        self._last_step_idx = step_idx

    # ─────────────────────────────────────────────────────────
    # Step methods
    # ─────────────────────────────────────────────────────────
    def step(self):
        if self.mode == "optimize":
            if getattr(self, 'opt_method', 'joint') == 'alternating':
                self._step_optimize_alternating()
            else:
                self._step_optimize()
        elif self.mode in ("forward", "playback"):
            self._step_forward()

    def _step_optimize(self):
        """One gradient-descent iteration on current window."""
        w_start = self.current_window * self.window_size
        w_end = min(w_start + self.window_size, self.T)

        # Initialize state from reference (OUTSIDE tape — constant)
        self._set_state_from_ref(self.states[0], w_start)

        # One-shot collision detection OUTSIDE tape (following Newton
        # diffsim examples). Contact detection is non-differentiable;
        # only the penalty-based force computation is differentiable
        # and that runs inside the solver (inside the tape).
        self.collision_pipeline.collide(self.states[0], self.contacts)

        # Zero loss
        self.loss.zero_()

        # Forward + backward through wp.Tape
        tape = wp.Tape()
        with tape:
            self.forward(w_start, w_end)
        tape.backward(self.loss)

        loss_val = self.loss.numpy()[0]
        self.loss_history.append(loss_val)

        # Adam update on delta_all with grad norm tracking
        grad_norm = 0.0
        if self.delta_all.grad is not None:
            g = self.delta_all.grad.numpy().reshape(self.T, self.n_dof)

            # Zero root DOF gradients — root is PD-controlled, not learned
            for p in range(self.n_persons):
                d = p * DOFS_PER_PERSON
                g[:, d:d + 6] = 0.0

            # Only keep gradients for current window frames
            # (out-of-window grads should be zero, but enforce explicitly)
            mask = np.zeros((self.T, 1), dtype=np.float32)
            mask[w_start:w_end] = 1.0
            g *= mask

            g_flat = g.flatten()
            grad_norm = float(np.linalg.norm(g_flat))

            # Clip gradients for stability (max norm = 100)
            if grad_norm > 100.0:
                g_flat *= 100.0 / grad_norm

            # Adam update (on CPU, then write back to GPU)
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

            # Only update delta for current window frames
            delta_np = self.delta_all.numpy()
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            update_2d = update.reshape(self.T, self.n_dof)
            update_2d[:w_start] = 0.0
            update_2d[w_end:] = 0.0
            delta_np -= update.flatten()
            self.delta_all = wp.array(
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

        self.train_iter += 1

        # Progress every 10 steps or at epoch boundary
        if self.train_iter % 10 == 1 or self.current_window == 0:
            d = self.delta_np
            d_hinge_mean = np.abs(d[:, 6:]).mean()
            d_hinge_max = np.abs(d[:, 6:]).max()
            print(f"  step {self.train_iter:4d} | epoch {self.epoch} "
                  f"win {self.current_window:3d}/{self.n_windows} | "
                  f"loss={loss_val:.6f} | grad={grad_norm:.2e} | "
                  f"Δ={d_hinge_mean:.2e} (max={d_hinge_max:.2e})",
                  flush=True)

    # ─────────────────────────────────────────────────────────
    # Alternating optimization
    # ─────────────────────────────────────────────────────────
    def forward_alternating(self, w_start, w_end):
        """Simulate one window with alternating position/torque control.

        Fixed person: high-gain PD tracking via:
          - Hinge DOFs: Newton's built-in PD (model.joint_target_ke/kd
            + control.joint_target_pos, set per-frame via kernel)
          - Root DOFs: custom PD via alternating_compose_kernel
            (FREE joints skip built-in PD)

        Free person: solo + delta via alternating_compose_kernel.

        MUST be called inside wp.Tape and AFTER _set_phase_gains().
        """
        fixed_person = 1 - self.free_person
        step_idx = 0

        for f, t_frame in enumerate(range(w_start, w_end)):
            # ── Set built-in PD target for fixed person's hinges ─
            # Copy pre-computed DOF-space reference to target_pos frame
            wp.launch(
                copy_dof_values_kernel,
                dim=self.n_dof,
                inputs=[self.ref_target_flat,
                        self.target_pos_frames[f],
                        t_frame * self.n_dof],
                device=self.device,
            )
            self.control.joint_target_pos = self.target_pos_frames[f]

            # ── Compose joint_f: root PD + free person solo+delta ─
            # alternating_compose_kernel handles:
            #   fixed root DOFs → PD from state
            #   fixed hinge DOFs → 0 (built-in PD handles them)
            #   free person DOFs → solo + delta
            wp.launch(
                alternating_compose_kernel,
                dim=self.n_dof,
                inputs=[
                    self.states[step_idx].joint_q,
                    self.states[step_idx].joint_qd,
                    self.ref_coords_flat,
                    self.solo_all,
                    self.delta_all,
                    self.combined_frames[f],
                    t_frame * self.n_dof,       # dof_frame_offset
                    t_frame * self.n_coords,    # coord_frame_offset
                    fixed_person * DOFS_PER_PERSON,   # fixed_dof_start
                    fixed_person * COORDS_PER_PERSON,  # fixed_coord_start
                    DOFS_PER_PERSON,            # n_dof_pp
                    self.fixed_ke_root,
                    self.fixed_kd_root,
                    self.torque_limit,
                ],
                device=self.device,
            )

            # ── Set control ──────────────────────────────────
            self.control.joint_f = self.combined_frames[f]

            # ── Physics substeps ─────────────────────────────
            # Collision detection done once outside tape.
            for _ in range(self.sim_substeps):
                self.solver.step(
                    self.states[step_idx],
                    self.states[step_idx + 1],
                    self.control,
                    self.contacts,
                    self.sim_dt,
                )
                step_idx += 1

            # ── Position loss (both persons, same as joint) ──
            if self.ref_pos_wp is not None and t_frame < len(self.ref_pos_wp):
                wp.launch(
                    position_loss_kernel,
                    dim=self.n_persons * N_JOINTS,
                    inputs=[
                        self.states[step_idx].body_q,
                        self.ref_pos_wp[t_frame],
                        self.smpl_to_body_wp,
                        self.n_persons,
                        N_BODIES_PER_PERSON,
                        N_JOINTS,
                        self.loss,
                    ],
                    device=self.device,
                )

        # ── L2 regularization on FREE person's delta only ────
        for t_frame in range(w_start, w_end):
            free_offset = (t_frame * self.n_dof
                           + self.free_person * DOFS_PER_PERSON)
            wp.launch(
                delta_reg_from_flat_kernel,
                dim=DOFS_PER_PERSON,
                inputs=[self.delta_all, self.reg_lambda,
                        free_offset, DOFS_PER_PERSON, self.loss],
                device=self.device,
            )

        self._last_step_idx = step_idx

    def _step_optimize_alternating(self):
        """One alternating gradient-descent iteration."""
        # Determine phase (which person to optimize)
        phase = (self.epoch // self.phase_epochs) % 2
        self.free_person = phase
        fixed_person = 1 - phase

        # Update model PD gains if phase changed (OUTSIDE tape)
        if phase != self._prev_phase:
            self._set_phase_gains(self.free_person)
            self._prev_phase = phase
            if self.train_iter > 0:
                print(f"  ── Phase switch: optimizing person {self.free_person}, "
                      f"fixing person {fixed_person} ──")

        w_start = self.current_window * self.window_size
        w_end = min(w_start + self.window_size, self.T)

        # Initialize state from reference (OUTSIDE tape)
        self._set_state_from_ref(self.states[0], w_start)

        # One-shot collision detection outside tape
        self.collision_pipeline.collide(self.states[0], self.contacts)

        # Zero loss
        self.loss.zero_()

        # Forward + backward through wp.Tape
        tape = wp.Tape()
        with tape:
            self.forward_alternating(w_start, w_end)
        tape.backward(self.loss)

        loss_val = self.loss.numpy()[0]
        self.loss_history.append(loss_val)

        # Gradient masking: zero out gradients for fixed person's DOFs
        grad_norm = 0.0
        if self.delta_all.grad is not None:
            g = self.delta_all.grad.numpy().reshape(self.T, self.n_dof)

            # Zero fixed person's delta gradients
            fd = fixed_person * DOFS_PER_PERSON
            g[:, fd:fd + DOFS_PER_PERSON] = 0.0

            # Only keep gradients for current window frames
            mask = np.zeros((self.T, 1), dtype=np.float32)
            mask[w_start:w_end] = 1.0
            g *= mask

            g_flat = g.flatten()
            grad_norm = float(np.linalg.norm(g_flat))

            # Clip gradients
            if grad_norm > 100.0:
                g_flat *= 100.0 / grad_norm

            # Adam update (on CPU, then write back to GPU)
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

            # Only update delta for current window frames
            delta_np = self.delta_all.numpy()
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            update_2d = update.reshape(self.T, self.n_dof)
            update_2d[:w_start] = 0.0
            update_2d[w_end:] = 0.0
            delta_np -= update.flatten()
            self.delta_all = wp.array(
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

        self.train_iter += 1

        # Progress printing
        if self.train_iter % 10 == 1 or self.current_window == 0:
            d = self.delta_np
            free_d = d[:, self.free_person * DOFS_PER_PERSON + 6:
                         self.free_person * DOFS_PER_PERSON + DOFS_PER_PERSON]
            d_mean = np.abs(free_d).mean()
            d_max = np.abs(free_d).max()
            print(f"  step {self.train_iter:4d} | epoch {self.epoch} "
                  f"win {self.current_window:3d}/{self.n_windows} | "
                  f"free=P{self.free_person} | "
                  f"loss={loss_val:.6f} | grad={grad_norm:.2e} | "
                  f"Δ={d_mean:.2e} (max={d_max:.2e})",
                  flush=True)

    def _step_forward(self):
        """Forward-only: simulate one control frame for real-time playback."""
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now

        self.sim_time = now - self._wall_start
        target_frame = int(self.sim_time * self.fps) % self.T

        # Re-init on first frame or loop
        if not self._fwd_initialized or target_frame == 0:
            self._set_state_from_ref(self.state_fwd_0, 0)
            self._fwd_initialized = True

        # Compose torques: root PD + joint PD + (solo+delta) joints
        delta_np = self.delta_all.numpy().reshape(self.T, self.n_dof)
        tau = np.clip(
            self.solo_np[target_frame] + delta_np[target_frame],
            -self.torque_limit, self.torque_limit,
        ).astype(np.float32)

        # Overwrite root DOFs with PD toward reference
        # and add joint PD tracking for solver mismatch compensation
        from scipy.spatial.transform import Rotation
        state_q = self.state_fwd_0.joint_q.numpy()
        state_qd = self.state_fwd_0.joint_qd.numpy()
        ref = self._ref_coords_np[target_frame]
        for p in range(self.n_persons):
            d = p * DOFS_PER_PERSON
            c = p * COORDS_PER_PERSON
            # Position PD
            tau[d:d + 3] = np.clip(
                self.ke_root * (ref[c:c + 3] - state_q[c:c + 3])
                - self.kd_root * state_qd[d:d + 3],
                -self.torque_limit, self.torque_limit,
            )
            # Rotation PD (quaternion → axis-angle)
            q_cur = state_q[c + 3:c + 7].copy()
            qn = np.linalg.norm(q_cur)
            if qn > 1e-8:
                q_cur /= qn
            R_err = (
                Rotation.from_quat(ref[c + 3:c + 7])
                * Rotation.from_quat(q_cur).inv()
            ).as_rotvec()
            tau[d + 3:d + 6] = np.clip(
                self.ke_root * R_err - self.kd_root * state_qd[d + 3:d + 6],
                -self.torque_limit, self.torque_limit,
            )
            # Joint PD tracking
            for j in range(6, DOFS_PER_PERSON):
                coord_idx = c + 7 + (j - 6)
                pos_err = ref[coord_idx] - state_q[coord_idx]
                vel = state_qd[d + j]
                tau[d + j] += self.ke_joint * pos_err - self.kd_joint * vel
            tau[d + 6:d + DOFS_PER_PERSON] = np.clip(
                tau[d + 6:d + DOFS_PER_PERSON],
                -self.torque_limit, self.torque_limit,
            )

        self.control.joint_f = wp.array(
            tau, dtype=wp.float32, device=self.device
        )

        for _ in range(self.sim_substeps):
            self.collision_pipeline.collide(
                self.state_fwd_0, self.contacts
            )
            self.solver.step(
                self.state_fwd_0, self.state_fwd_1,
                self.control, self.contacts, self.sim_dt,
            )
            self.state_fwd_0, self.state_fwd_1 = (
                self.state_fwd_1, self.state_fwd_0
            )

        self._render_frame = target_frame

    # ─────────────────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────────────────
    def render(self):
        if self.mode == "optimize":
            self._render_optimize()
        else:
            self._render_realtime()

    def _render_optimize(self):
        """Show the last state of the last simulated window."""
        idx = min(self._last_step_idx, len(self.states) - 1)
        state = self.states[idx]

        self.viewer.begin_frame(self.frame * (1.0 / self.fps))
        self.viewer.log_state(state)
        if self.loss_history:
            self.viewer.log_scalar("/loss", self.loss_history[-1])
        self.viewer.end_frame()
        self.frame += 1

    def _render_realtime(self):
        """Render current state for forward/playback mode."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_fwd_0)
        self.viewer.end_frame()

    # ─────────────────────────────────────────────────────────
    # GUI
    # ─────────────────────────────────────────────────────────
    def gui(self, imgui):
        imgui.separator()
        color = {
            "optimize": imgui.ImVec4(1.0, 0.4, 0.4, 1.0),
            "forward":  imgui.ImVec4(0.4, 0.8, 1.0, 1.0),
            "playback": imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
        }[self.mode]
        imgui.text_colored(color, f"[ {self.mode.upper()} ]")
        imgui.separator()
        imgui.text(f"Clip:       {self._clip_id}")
        imgui.text(f"Persons:    {self.n_persons}")
        imgui.text(f"Frames:     {self.T}")
        imgui.text(f"FPS:        {self.fps}")
        imgui.text(f"Sim freq:   {self.sim_freq} Hz")
        imgui.text(f"Substeps:   {self.sim_substeps}")

        if self.mode == "optimize":
            imgui.separator()
            opt_method = getattr(self, 'opt_method', 'joint')
            imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.5, 1.0),
                               "Optimization:")
            imgui.text(f"Method:     {opt_method}")
            if opt_method == 'alternating':
                free_p = getattr(self, 'free_person', 0)
                imgui.text(f"Free:       person {free_p}")
                imgui.text(f"Fixed:      person {1 - free_p}")
            imgui.text(f"Epoch:      {self.epoch}")
            imgui.text(f"Window:     {self.current_window}/{self.n_windows}")
            imgui.text(f"Iteration:  {self.train_iter}")
            imgui.text(f"LR:         {self.lr:.1e}")
            imgui.text(f"Lambda:     {self.reg_lambda:.3f}")
            if self.loss_history:
                imgui.text(f"Loss:       {self.loss_history[-1]:.6f}")
                if len(self.loss_history) > 1:
                    imgui.text(f"Loss (min): {min(self.loss_history):.6f}")

            delta_np = self.delta_np
            abs_d = np.abs(delta_np[:, 6:])
            if abs_d.max() > 0:
                imgui.separator()
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 1.0, 1.0),
                                   "Delta stats (hinges):")
                imgui.text(f"Mean |DT|:  {abs_d.mean():.2f} Nm")
                imgui.text(f"Max  |DT|:  {abs_d.max():.2f} Nm")
                frame_mag = np.linalg.norm(delta_np[:, 6:], axis=1)
                sparse = (frame_mag < 1.0).mean()
                imgui.text(f"Sparsity:   {100*sparse:.0f}%")
        else:
            imgui.separator()
            imgui.text(f"Frame:      {self._render_frame}")

            # Show position error if ref positions available
            if self.ref_positions is not None:
                sim_pos = self._extract_body_positions(self.state_fwd_0)
                ref = self.ref_positions[self._render_frame]
                err = np.linalg.norm(sim_pos - ref, axis=-1).mean() * 100
                imgui.text(f"MPJPE:      {err:.1f} cm")

    # ─────────────────────────────────────────────────────────
    # Save / print
    # ─────────────────────────────────────────────────────────
    def save_results(self, data_dir, output_dir=None):
        """Save optimized delta and full torques.

        Args:
            data_dir: original data directory (used as default output)
            output_dir: optional separate output directory
        """
        save_dir = output_dir if output_dir else data_dir
        os.makedirs(save_dir, exist_ok=True)
        clip_id = self._clip_id
        delta_np = self.delta_np

        for i, p_idx in enumerate(self.person_indices):
            d = i * DOFS_PER_PERSON
            d_end = d + DOFS_PER_PERSON

            delta_person = delta_np[:, d:d_end]
            solo_person = self.all_torques_solo[i][:self.T]
            full_person = solo_person + delta_person

            delta_path = os.path.join(
                save_dir, f"{clip_id}_person{p_idx}_delta_torques.npy"
            )
            full_path = os.path.join(
                save_dir, f"{clip_id}_person{p_idx}_torques_full.npy"
            )

            np.save(delta_path, delta_person.astype(np.float32))
            np.save(full_path, full_person.astype(np.float32))

        print(f"Saved delta and full torques for clip {clip_id}")
        self._print_delta_stats()

    def _print_delta_stats(self):
        delta_np = self.delta_np
        print(f"\nDelta stats (clip {self._clip_id}):")
        for i, p_idx in enumerate(self.person_indices):
            d = i * DOFS_PER_PERSON
            delta = delta_np[:, d:d + DOFS_PER_PERSON]
            abs_d = np.abs(delta)
            frame_mag = np.linalg.norm(delta[:, 6:], axis=1)
            sparse_frac = (frame_mag < 1.0).mean()
            print(f"  Person {p_idx}:")
            print(f"    Root forces  (0:3):  mean={abs_d[:, :3].mean():.2f} "
                  f"max={abs_d[:, :3].max():.2f} N")
            print(f"    Root torques (3:6):  mean={abs_d[:, 3:6].mean():.2f} "
                  f"max={abs_d[:, 3:6].max():.2f} Nm")
            print(f"    Hinges       (6:):   mean={abs_d[:, 6:].mean():.2f} "
                  f"max={abs_d[:, 6:].max():.2f} Nm")
            print(f"    Sparsity: {100*sparse_frac:.0f}% frames < 1 Nm")


# ═══════════════════════════════════════════════════════════════
# Headless batch runner
# ═══════════════════════════════════════════════════════════════
def run_headless(args):
    """Run optimization without viewer."""
    from prepare2.compute_torques import list_retargeted_clips

    data_dir = args.data_dir

    if args.clip:
        clips = [args.clip]
    elif args.dataset:
        clips = list_retargeted_clips(data_dir)
        valid = [
            c for c in clips
            if all(
                os.path.exists(os.path.join(
                    data_dir, f"{c}_person{p}_torques_solo.npy"
                ))
                for p in [0, 1]
            )
        ]
        clips = valid
        print(f"Found {len(clips)} clips with solo torques for both persons")
    else:
        print("ERROR: specify --clip or --dataset")
        sys.exit(1)

    class StubViewer:
        def set_model(self, m): pass
        def set_camera(self, *a, **kw): pass
        def begin_frame(self, t): pass
        def log_state(self, s): pass
        def log_scalar(self, k, v): pass
        def end_frame(self): pass
        def close(self): pass
        def register_ui_callback(self, *a, **kw): pass

    for clip_id in clips:
        out_path = os.path.join(
            data_dir, f"{clip_id}_person0_delta_torques.npy"
        )
        if os.path.exists(out_path) and not args.force:
            print(f"Skipping {clip_id} (exists, use --force)")
            continue

        try:
            print(f"\n{'='*60}")
            print(f"Clip: {clip_id}")
            print(f"{'='*60}")

            args_copy = argparse.Namespace(**vars(args))
            args_copy.clip = clip_id
            args_copy.mode = "optimize"

            opt = InteractionOptimizer(StubViewer(), args_copy)

            max_iters = args.epochs * opt.n_windows
            t0 = time.time()
            for iteration in range(max_iters):
                opt.step()

            elapsed = time.time() - t0
            print(f"Optimization done in {elapsed:.1f}s")
            opt.save_results(data_dir)

        except Exception as e:
            print(f"ERROR on clip {clip_id}: {e}")
            import traceback
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, default="1000",
                        help="Clip ID")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["interhuman", "interx"],
                        help="Process entire dataset (headless)")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman",
                        help="Data directory")
    parser.add_argument("--mode",
                        choices=["forward", "optimize", "playback"],
                        default="optimize",
                        help="forward=diagnose, optimize=train ΔT, "
                             "playback=replay saved ΔT")
    parser.add_argument("--fps", type=int, default=30,
                        help="Data playback FPS (should match data frame rate; "
                             "InterHuman raw=60, processed=30)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample loaded data by this factor "
                             "(2 = 60→30 fps to match InterMask evaluation)")
    parser.add_argument("--sim-freq", type=int, default=480,
                        help="Simulation frequency in Hz (default 480). "
                             "Lower = faster optimization but less accurate. "
                             "120 Hz is a good speed/quality tradeoff for batch.")
    parser.add_argument("--lr", type=float, default=1.0,
                        help="Gradient descent learning rate "
                             "(high default needed: gradients attenuate "
                             "through many solver substeps)")
    parser.add_argument("--reg-lambda", type=float, default=0.01,
                        help="L2 regularization weight on ΔT")
    parser.add_argument("--ke-root", type=float, default=5000.0,
                        help="Root PD position/rotation gain (both methods)")
    parser.add_argument("--kd-root", type=float, default=500.0,
                        help="Root PD damping gain (both methods)")
    parser.add_argument("--ke-joint", type=float, default=200.0,
                        help="Joint PD position gain (compensates solver mismatch)")
    parser.add_argument("--kd-joint", type=float, default=20.0,
                        help="Joint PD damping gain")
    parser.add_argument("--window", type=int, default=10,
                        help="Frames per optimization window")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of full passes (headless only)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--method", type=str, default="joint",
                        choices=["joint", "alternating"],
                        help="Optimization method: "
                             "'joint' = optimize both persons' ΔT simultaneously; "
                             "'alternating' = fix one person with position control, "
                             "optimize the other, then swap each epoch")
    parser.add_argument("--phase-epochs", type=int, default=1,
                        help="Switch fixed/free person every N epochs "
                             "(alternating method only)")
    parser.add_argument("--fixed-ke", type=float, default=5000.0,
                        help="PD position gain for fixed person's hinges "
                             "(alternating method only)")
    parser.add_argument("--fixed-kd", type=float, default=500.0,
                        help="PD velocity gain for fixed person's hinges "
                             "(alternating method only)")
    parser.add_argument("--fixed-ke-root", type=float, default=50000.0,
                        help="PD position gain for fixed person's root "
                             "(alternating method only)")
    parser.add_argument("--fixed-kd-root", type=float, default=5000.0,
                        help="PD velocity gain for fixed person's root "
                             "(alternating method only)")

    # Check if headless mode
    pre_args, _ = parser.parse_known_args()
    is_headless = pre_args.dataset or getattr(pre_args, "headless", False)

    if is_headless:
        args = parser.parse_args()
        if args.dataset:
            args.data_dir = os.path.join(
                PROJECT_ROOT, f"data/retargeted_v2/{args.dataset}"
            )
        else:
            args.data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
        if not hasattr(args, "device") or args.device is None:
            args.device = "cuda:0"
        run_headless(args)
    else:
        viewer, args = newton.examples.init(parser)
        if not hasattr(args, "device") or args.device is None:
            args.device = "cuda:0"
        args.data_dir = os.path.join(PROJECT_ROOT, args.data_dir)

        example = InteractionOptimizer(viewer, args)
        newton.examples.run(example, args)

        # Auto-save on exit if optimizing
        if args.mode == "optimize" and example.train_iter > 0:
            example.save_results(args.data_dir)
