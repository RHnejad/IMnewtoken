"""
rollout_env.py — Vectorised Newton environment for PPO RL tracking.

One Newton model containing N_ENVS character copies, constructed via
builder.replicate() (the ProtoMotions-verified API). Each env tracks
a reference motion independently, enabling massively parallel rollouts.

Key design decisions (verified from ProtoMotions source):
  - Multi-env: builder.replicate(robot_builder, N) — NOT add_mjcf per env
  - PD control: Warp kernel (compute_pd_torques_kernel pattern)
  - Passive forces: zeroed after finalize to avoid double-counting
  - Actions: residual offsets on top of reference joint angles
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare6.rl_config import (
    FPS, SIM_FREQ, SIM_SUBSTEPS, DT,
    COORDS_PER_PERSON, DOFS_PER_PERSON, BODIES_PER_PERSON,
    N_SMPL_JOINTS, SMPL_TO_NEWTON,
    PHC_BODY_GAINS, BODY_NAMES,
    ROOT_POS_KP, ROOT_POS_KD, ROOT_ROT_KP, ROOT_ROT_KD,
    TORQUE_LIMIT,
    DEFAULT_BODY_MASS_KG, SETTLE_FRAMES,
    TERMINATION_DISTANCE, MIN_HEIGHT,
    N_ENVS, OBS_DIM_SOLO, ACT_DIM_SOLO,
    ACTION_SCALE, RSI_PROB,
)
from prepare6.obs_builder import build_obs_batch
from prepare5.phc_reward import compute_imitation_reward, extract_body_state


# ── Warp PD kernel (ProtoMotions compute_pd_torques_kernel pattern) ──────────
_pd_kernel_registered = False


def _ensure_pd_kernel():
    global _pd_kernel_registered
    if _pd_kernel_registered:
        return
    import warp as wp

    @wp.kernel
    def _pd_torques_kernel(
        joint_q:   wp.array(dtype=wp.float32),
        joint_qd:  wp.array(dtype=wp.float32),
        joint_f:   wp.array(dtype=wp.float32),
        targets:   wp.array(dtype=wp.float32),
        kp:        wp.array(dtype=wp.float32),
        kd:        wp.array(dtype=wp.float32),
        limits:    wp.array(dtype=wp.float32),
        q_stride:  int,
        qd_stride: int,
        q_start:   int,
        qd_start:  int,
        num_dofs:  int,
    ):
        tid    = wp.tid()
        env_id = tid // num_dofs
        dof_id = tid % num_dofs
        q_idx  = env_id * q_stride  + q_start  + dof_id
        qd_idx = env_id * qd_stride + qd_start + dof_id
        tau = kp[dof_id] * (targets[tid] - joint_q[q_idx]) \
              - kd[dof_id] * joint_qd[qd_idx]
        joint_f[qd_idx] = wp.clamp(tau, -limits[dof_id], limits[dof_id])

    # store in module so it survives
    _ensure_pd_kernel._kernel = _pd_torques_kernel
    _pd_kernel_registered = True


# ── FK helper ────────────────────────────────────────────────────────────────

def _run_fk(model, state, device):
    import warp as wp
    import newton
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _extract_all_envs_body_state(state, n_envs):
    """Extract body_q/body_qd for all envs from Newton state.

    Returns:
        body_q_all:  (n_envs, BODIES_PER_PERSON, 7)  pos(3)+quat(4)
        body_qd_all: (n_envs, BODIES_PER_PERSON, 6)  vel(3)+omega(3)
    """
    bq = state.body_q.numpy().reshape(-1, 7)    # (n_envs * BODIES_PER_PERSON, 7)
    bqd = state.body_qd.numpy().reshape(-1, 6)
    bq_all  = bq.reshape(n_envs, BODIES_PER_PERSON, 7)
    bqd_all = bqd.reshape(n_envs, BODIES_PER_PERSON, 6)
    return bq_all, bqd_all


def _body_state_to_smpl(bq_all, bqd_all, n_envs):
    """Map Newton body indices to SMPL joint ordering.

    Returns:
        body_pos:     (n_envs, 22, 3)
        body_rot:     (n_envs, 22, 4)  xyzw
        body_vel:     (n_envs, 22, 3)
        body_ang_vel: (n_envs, 22, 3)
    """
    body_pos     = np.zeros((n_envs, N_SMPL_JOINTS, 3), dtype=np.float32)
    body_rot     = np.zeros((n_envs, N_SMPL_JOINTS, 4), dtype=np.float32)
    body_vel     = np.zeros((n_envs, N_SMPL_JOINTS, 3), dtype=np.float32)
    body_ang_vel = np.zeros((n_envs, N_SMPL_JOINTS, 4-1), dtype=np.float32)

    for smpl_j, newton_b in SMPL_TO_NEWTON.items():
        body_pos[:, smpl_j, :]     = bq_all[:, newton_b, :3]
        body_rot[:, smpl_j, :]     = bq_all[:, newton_b, 3:7]
        body_vel[:, smpl_j, :]     = bqd_all[:, newton_b, :3]
        body_ang_vel[:, smpl_j, :] = bqd_all[:, newton_b, 3:6]

    return body_pos, body_rot, body_vel, body_ang_vel


# ── RolloutEnv ────────────────────────────────────────────────────────────────

class RolloutEnv:
    """Vectorised Newton environment for PPO motion tracking.

    N_ENVS character copies in a single Newton model, all tracking
    the same reference clip but at different random start frames (RSI).
    """

    def __init__(self, ref_joint_q, betas, n_envs=N_ENVS, device="cuda:0",
                 verbose=True):
        """
        Args:
            ref_joint_q: (T, 76) reference joint coordinates
            betas:       (10,) SMPL-X shape params
            n_envs:      number of parallel environments
            device:      CUDA device string
            verbose:     print build info
        """
        self.ref_joint_q = ref_joint_q.astype(np.float32)
        self.betas = betas
        self.n_envs = n_envs
        self.device = device
        self.verbose = verbose
        self.T = ref_joint_q.shape[0]

        self._dt_sim = 1.0 / SIM_FREQ
        self._sim_steps = SIM_SUBSTEPS  # 16 substeps / frame

        # current frame index for each env
        self.env_frames = np.zeros(n_envs, dtype=np.int32)
        self._done = np.zeros(n_envs, dtype=bool)

        # Pre-compute reference body positions via FK (for obs + reward)
        self._precompute_ref_body_states()

        # Build Newton model
        self._build_model()

        # Build PD gain arrays (per-hinge-DOF, length ACT_DIM_SOLO=69)
        self._build_gains()

        if verbose:
            print(f"[RolloutEnv] {n_envs} envs, T={self.T}, device={device}")

    # ── Model construction ────────────────────────────────────────────────────

    def _build_model(self):
        import warp as wp
        import newton
        from prepare4.gen_xml import get_or_create_xml
        from prepare4.dynamics import set_segment_masses

        xml_path = get_or_create_xml(self.betas, foot_geom="sphere")

        # Step 1: single-robot builder
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
        robot_builder.add_mjcf(
            xml_path,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )

        # Step 2: replicate N_ENVS times + ground plane
        world_builder = newton.ModelBuilder()
        world_builder.replicate(robot_builder, self.n_envs)
        world_builder.add_ground_plane()
        self.model = world_builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        # Step 3: zero MJCF passive forces (avoid double-counting our explicit PD)
        if hasattr(self.model, 'mujoco'):
            if hasattr(self.model.mujoco, 'dof_passive_stiffness'):
                self.model.mujoco.dof_passive_stiffness.fill_(0.0)
            if hasattr(self.model.mujoco, 'dof_passive_damping'):
                self.model.mujoco.dof_passive_damping.fill_(0.0)

        # Disable built-in PD (we use explicit PD via joint_f)
        self.model.joint_target_ke.fill_(0.0)
        self.model.joint_target_kd.fill_(0.0)

        # Disable joint limits (generated data can exceed MJCF-defined limits)
        import warp as wp
        n_dof = self.model.joint_dof_count
        self.model.joint_limit_lower = wp.array(
            np.full(n_dof, -1e6, dtype=np.float32), dtype=wp.float32,
            device=self.device
        )
        self.model.joint_limit_upper = wp.array(
            np.full(n_dof, 1e6, dtype=np.float32), dtype=wp.float32,
            device=self.device
        )

        # Armature
        arm = np.full(n_dof, 0.5, dtype=np.float32)
        for i in range(self.n_envs):
            off = i * DOFS_PER_PERSON
            arm[off:off + 6] = 5.0
        self.model.joint_armature = wp.array(arm, dtype=wp.float32,
                                              device=self.device)

        # Strides for vectorised kernel
        self._q_stride  = COORDS_PER_PERSON   # 76 per env
        self._qd_stride = DOFS_PER_PERSON      # 75 per env
        # hinge DOFs start at index 6 within each env's block
        self._q_hinge_start  = 7   # after 7 free-joint coords
        self._qd_hinge_start = 6   # after 6 free-joint vel DOFs
        self._n_hinge_dofs   = ACT_DIM_SOLO    # 69

        # Solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver="newton",
            njmax=450 * self.n_envs,
            nconmax=150 * self.n_envs,
            impratio=10, iterations=50, ls_iterations=25,
        )

        # State buffers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        if self.verbose:
            print(f"[RolloutEnv] Model built: {n_dof} total DOFs, "
                  f"{self.n_envs} envs × {DOFS_PER_PERSON} DOFs/env")

    def _build_gains(self):
        """Build per-hinge-DOF kp/kd/limit arrays (length ACT_DIM_SOLO=69)."""
        kp = np.zeros(ACT_DIM_SOLO, dtype=np.float32)
        kd = np.zeros(ACT_DIM_SOLO, dtype=np.float32)
        lim = np.full(ACT_DIM_SOLO, TORQUE_LIMIT, dtype=np.float32)

        for b_idx in range(23):
            s = b_idx * 3
            body_name = BODY_NAMES[1 + b_idx]
            kp_val, kd_val = PHC_BODY_GAINS.get(body_name, (300, 30))
            kp[s:s + 3] = kp_val
            kd[s:s + 3] = kd_val

        import warp as wp
        self._kp_wp  = wp.array(kp,  dtype=wp.float32, device=self.device)
        self._kd_wp  = wp.array(kd,  dtype=wp.float32, device=self.device)
        self._lim_wp = wp.array(lim, dtype=wp.float32, device=self.device)

    # ── Reference state pre-computation ──────────────────────────────────────

    def _precompute_ref_body_states(self):
        """Run FK on all T reference frames to get ref_body_pos/rot.

        Stores:
            self._ref_body_pos: (T, 22, 3)
            self._ref_body_rot: (T, 22, 4) xyzw
            self._ref_body_vel: (T, 22, 3)   (finite-difference)
            self._ref_body_ang_vel: (T, 22, 3)
        """
        import warp as wp
        import newton

        if self.verbose:
            print("[RolloutEnv] Pre-computing reference body states (FK)...")

        T = self.T
        ref_pos     = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        ref_rot     = np.zeros((T, N_SMPL_JOINTS, 4), dtype=np.float32)
        ref_vel     = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        ref_ang_vel = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)

        # Build a temporary single-person model for FK
        from prepare4.gen_xml import get_or_create_xml
        xml_path = get_or_create_xml(self.betas, foot_geom="sphere")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_mjcf(xml_path, enable_self_collisions=False,
                         collapse_fixed_joints=False)
        fk_model = builder.finalize()

        state_fk = fk_model.state()
        n_coords = fk_model.joint_coord_count
        n_dof = fk_model.joint_dof_count

        for t in range(T):
            jq = np.zeros(n_coords, dtype=np.float32)
            jq[:COORDS_PER_PERSON] = self.ref_joint_q[t]
            state_fk.joint_q = wp.array(jq, dtype=wp.float32)

            # velocity via finite difference
            if t > 0:
                jqd_np = np.zeros(n_dof, dtype=np.float32)
                dq_hinge = (self.ref_joint_q[t, 7:] - self.ref_joint_q[t-1, 7:])
                jqd_np[6:DOFS_PER_PERSON] = dq_hinge * FPS
            else:
                jqd_np = np.zeros(n_dof, dtype=np.float32)
            state_fk.joint_qd = wp.array(jqd_np, dtype=wp.float32)

            newton.eval_fk(fk_model, state_fk.joint_q, state_fk.joint_qd, state_fk)

            body_pos, body_rot, body_vel, body_ang_vel = extract_body_state(
                state_fk, person_idx=0
            )
            ref_pos[t]     = body_pos
            ref_rot[t]     = body_rot
            ref_vel[t]     = body_vel
            ref_ang_vel[t] = body_ang_vel

        self._ref_body_pos     = ref_pos
        self._ref_body_rot     = ref_rot
        self._ref_body_vel     = ref_vel
        self._ref_body_ang_vel = ref_ang_vel

        if self.verbose:
            print(f"[RolloutEnv] FK done: {T} frames")

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, env_ids=None):
        """Reset environments to random reference frames (RSI).

        Args:
            env_ids: list/array of env indices to reset, or None for all

        Returns:
            obs: (n_envs, OBS_DIM_SOLO) if all reset, else subset
        """
        import warp as wp

        if env_ids is None:
            env_ids = np.arange(self.n_envs)

        # Current full joint_q (numpy)
        jq_full  = self.state_0.joint_q.numpy().copy()
        jqd_full = self.state_0.joint_qd.numpy().copy()

        for env_i in env_ids:
            # RSI: random frame with probability RSI_PROB
            if np.random.rand() < RSI_PROB and self.T > 1:
                start_frame = np.random.randint(0, self.T)
            else:
                start_frame = 0
            self.env_frames[env_i] = start_frame
            self._done[env_i] = False

            off_q  = env_i * COORDS_PER_PERSON
            off_qd = env_i * DOFS_PER_PERSON
            jq_full[off_q:off_q + COORDS_PER_PERSON]   = self.ref_joint_q[start_frame]
            jqd_full[off_qd:off_qd + DOFS_PER_PERSON]  = 0.0

        self.state_0.joint_q  = wp.array(jq_full,  dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.array(jqd_full, dtype=wp.float32, device=self.device)
        _run_fk(self.model, self.state_0, self.device)

        return self._get_obs()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions):
        """Step all envs one physics frame.

        Args:
            actions: (n_envs, ACT_DIM_SOLO) action residuals in [-1, 1]

        Returns:
            obs:     (n_envs, OBS_DIM_SOLO)
            rewards: (n_envs,)
            dones:   (n_envs,) bool
            info:    dict
        """
        import warp as wp

        actions = np.asarray(actions, dtype=np.float32)

        # Advance each env's frame counter
        self.env_frames = np.minimum(self.env_frames + 1, self.T - 1)

        # Build PD targets: ref hinge angles + tanh(action) * ACTION_SCALE * pi
        targets_np = np.zeros(self.n_envs * ACT_DIM_SOLO, dtype=np.float32)
        for i in range(self.n_envs):
            frame_i = self.env_frames[i]
            ref_hinge = self.ref_joint_q[frame_i, 7:COORDS_PER_PERSON]  # (69,)
            delta = np.tanh(actions[i]) * ACTION_SCALE * np.pi
            targets_np[i * ACT_DIM_SOLO:(i + 1) * ACT_DIM_SOLO] = ref_hinge + delta

        targets_wp = wp.array(targets_np, dtype=wp.float32, device=self.device)

        # Root PD forces (skyhook for root stability during RL training)
        jq_np  = self.state_0.joint_q.numpy()
        jqd_np = self.state_0.joint_qd.numpy()
        joint_f_np = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for i in range(self.n_envs):
            frame_i = self.env_frames[i]
            off_q  = i * COORDS_PER_PERSON
            off_qd = i * DOFS_PER_PERSON
            cq  = jq_np[off_q:off_q + COORDS_PER_PERSON]
            cqd = jqd_np[off_qd:off_qd + DOFS_PER_PERSON]
            ref  = self.ref_joint_q[frame_i]
            root_f = self._compute_root_pd(cq, cqd, ref)
            joint_f_np[off_qd:off_qd + 6] = root_f

        # Physics substeps: compute ALL torques (root + hinge) in one joint_f array
        kp_np = self._kp_wp.numpy()   # (69,) hinge gains
        kd_np = self._kd_wp.numpy()
        lim_np = self._lim_wp.numpy()

        for _sub in range(self._sim_steps):
            jq_sub  = self.state_0.joint_q.numpy()
            jqd_sub = self.state_0.joint_qd.numpy()

            jf = joint_f_np.copy()   # start with root PD forces; zeros elsewhere

            # Hinge PD for all envs
            for i in range(self.n_envs):
                frame_i  = self.env_frames[i]
                off_q    = i * COORDS_PER_PERSON
                off_qd   = i * DOFS_PER_PERSON
                # current hinge angles and velocities
                cur_q    = jq_sub[off_q + 7:off_q + COORDS_PER_PERSON]   # (69,)
                cur_qd   = jqd_sub[off_qd + 6:off_qd + DOFS_PER_PERSON]  # (69,)
                # target = ref + residual action
                tgt      = targets_np[i * ACT_DIM_SOLO:(i + 1) * ACT_DIM_SOLO]
                tau      = kp_np * (tgt - cur_q) - kd_np * cur_qd
                tau      = np.clip(tau, -lim_np, lim_np)
                jf[off_qd + 6:off_qd + DOFS_PER_PERSON] = tau

            self.control.joint_f = wp.array(jf, dtype=wp.float32, device=self.device)

            self.state_0.clear_forces()
            contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts,
                             self._dt_sim)
            self.state_0, self.state_1 = self.state_1, self.state_0

        _run_fk(self.model, self.state_0, self.device)

        # Compute rewards + terminations
        rewards = self._compute_rewards()
        dones   = self._check_terminations()

        # Auto-reset terminated envs
        terminated_ids = np.where(dones)[0]
        if len(terminated_ids) > 0:
            self.reset(env_ids=terminated_ids)

        obs = self._get_obs()
        return obs, rewards, dones, {}

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self):
        bq_all, bqd_all = _extract_all_envs_body_state(self.state_0, self.n_envs)
        body_pos, body_rot, body_vel, body_ang_vel = _body_state_to_smpl(
            bq_all, bqd_all, self.n_envs
        )

        ref_pos_batch = np.array([
            self._ref_body_pos[min(f, self.T - 1)] for f in self.env_frames
        ])
        ref_rot_batch = np.array([
            self._ref_body_rot[min(f, self.T - 1)] for f in self.env_frames
        ])

        phases = self.env_frames / max(self.T - 1, 1)

        obs = build_obs_batch(
            body_pos, body_rot, body_vel, body_ang_vel,
            ref_pos_batch, ref_rot_batch,
            phases,
        )
        return obs.astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_rewards(self):
        bq_all, bqd_all = _extract_all_envs_body_state(self.state_0, self.n_envs)
        body_pos, body_rot, body_vel, body_ang_vel = _body_state_to_smpl(
            bq_all, bqd_all, self.n_envs
        )

        rewards = np.zeros(self.n_envs, dtype=np.float32)
        for i in range(self.n_envs):
            frame_i = self.env_frames[i]
            rew, _ = compute_imitation_reward(
                body_pos[i], body_rot[i], body_vel[i], body_ang_vel[i],
                self._ref_body_pos[frame_i], self._ref_body_rot[frame_i],
                self._ref_body_vel[frame_i], self._ref_body_ang_vel[frame_i],
            )
            rewards[i] = rew
        return rewards

    # ── Termination ───────────────────────────────────────────────────────────

    def _check_terminations(self):
        jq_np = self.state_0.joint_q.numpy()
        dones = np.zeros(self.n_envs, dtype=bool)

        # Also done when reached end of clip
        dones |= (self.env_frames >= self.T - 1)

        for i in range(self.n_envs):
            if dones[i]:
                continue
            off_q = i * COORDS_PER_PERSON
            frame_i = self.env_frames[i]
            cur_root = jq_np[off_q:off_q + 3]
            ref_root = self.ref_joint_q[frame_i, :3]
            if np.linalg.norm(cur_root - ref_root) > TERMINATION_DISTANCE:
                dones[i] = True
            if jq_np[off_q + 2] < MIN_HEIGHT:
                dones[i] = True

        return dones

    # ── Root PD (skyhook) ─────────────────────────────────────────────────────

    def _compute_root_pd(self, cq, cqd, ref_q):
        """Compute root PD forces for one env."""
        tau = np.zeros(6, dtype=np.float32)

        # Position PD
        tau[:3] = ROOT_POS_KP * (ref_q[:3] - cq[:3]) - ROOT_POS_KD * cqd[:3]

        # Orientation PD
        q_cur = cq[3:7].copy()
        qn = np.linalg.norm(q_cur)
        if qn > 1e-8:
            q_cur /= qn
        R_err = (
            Rotation.from_quat(ref_q[3:7])
            * Rotation.from_quat(q_cur).inv()
        ).as_rotvec()
        tau[3:6] = ROOT_ROT_KP * R_err - ROOT_ROT_KD * cqd[3:6]

        return np.clip(tau, -TORQUE_LIMIT * 10, TORQUE_LIMIT * 10)

    # ── Evaluation helper ─────────────────────────────────────────────────────

    def evaluate_single_pass(self, policy_fn):
        """Run one deterministic evaluation pass (no RSI, start frame 0).

        Args:
            policy_fn: callable(obs (1, OBS_DIM)) -> action (1, ACT_DIM)

        Returns:
            sim_positions: (T, 22, 3)
            ref_positions: (T, 22, 3)
            sim_joint_q:   (T, 76) simulated joint coordinates (for Newton viz)
        """
        import warp as wp

        T = self.T
        sim_pos = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        sim_jq  = np.zeros((T, COORDS_PER_PERSON), dtype=np.float32)
        ref_pos = self._ref_body_pos.copy()

        # Reset ALL envs to frame 0 so no env has zero/invalid quaternions
        jq_full = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        for i in range(self.n_envs):
            jq_full[i * COORDS_PER_PERSON:(i + 1) * COORDS_PER_PERSON] = (
                self.ref_joint_q[0]
            )
        self.state_0.joint_q  = wp.array(jq_full,  dtype=wp.float32, device=self.device)
        self.state_0.joint_qd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32,
                                          device=self.device)
        self.env_frames[:] = 0
        _run_fk(self.model, self.state_0, self.device)

        bq0, _ = _extract_all_envs_body_state(self.state_0, self.n_envs)
        body_pos0, _, _, _ = _body_state_to_smpl(bq0, bq0 * 0, self.n_envs)
        sim_pos[0] = body_pos0[0]
        sim_jq[0]  = self.ref_joint_q[0]

        for t in range(1, T):
            obs = self._get_obs()
            act = policy_fn(obs[:1])   # single env obs
            act_full = np.zeros((self.n_envs, ACT_DIM_SOLO), dtype=np.float32)
            act_full[0] = act[0]

            _, _, _, _ = self.step(act_full)

            bq, bqd = _extract_all_envs_body_state(self.state_0, self.n_envs)
            bp, _, _, _ = _body_state_to_smpl(bq, bqd, self.n_envs)
            sim_pos[t] = bp[0]

            # Extract joint_q for env 0
            jq_np = self.state_0.joint_q.numpy()
            sim_jq[t] = jq_np[:COORDS_PER_PERSON]

        return sim_pos, ref_pos, sim_jq
