"""
newton_mimic_env.py — Newton-based RL environment for motion tracking.

Standalone Gymnasium-style environment that wraps Newton physics to train
a policy to track retargeted reference motions. Designed to be compatible
with MimicKit's PPO / ADD agent but does not inherit from MimicKit's env
hierarchy (since MimicKit has no Newton engine backend).

Key design principles:
- SolverMuJoCo (NOT SolverFeatherstone): LCP contacts, no penalty tuning
- requires_grad=False everywhere: no wp.Tape, pure forward sim
- 69-DOF action space: hinge joints ONLY (indices 6..74)
- Root PD (skyhook) stabilises pelvis at reference — matching prepare2
- PD re-computed at every substep (480 Hz) for stability
- Per-body gains from prepare2/pd_utils.py for accurate tracking
- Local-coordinate observations: no global X/Y, heading-relative
- Phase-conditioned: sin/cos phase encoding for cyclic motions
- DeepMimic-style reward: pose + vel + root_pose + root_vel + key_pos

Environment state:
    joint_q  (76,) = root_pos(3) + root_quat_xyzw(4) + hinge_angles(69)
    joint_qd (75,) = root_vel(3) + root_ang_vel(3) + hinge_vel(69)

Action: (69,) hinge torques/position targets → padded to (75,) with 6 zeros

Usage:
    env = NewtonMimicEnv(config)
    obs, info = env.reset()
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""
import os
import sys
import numpy as np
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Defer heavy imports (warp, newton) to avoid import errors in test
# environments that don't have GPU packages installed.
# They are imported lazily inside NewtonMimicEnv.__init__.
_wp = None
_newton = None

def _ensure_physics_imports():
    """Lazy-import warp and newton on first use."""
    global _wp, _newton
    if _wp is None:
        import warp as wp
        import newton
        _wp = wp
        _newton = newton
    return _wp, _newton

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
BODIES_PER_PERSON = 24
COORDS_PER_PERSON = 76   # 3 pos + 4 quat + 69 hinge
DOFS_PER_PERSON = 75     # 3 lin_vel + 3 ang_vel + 69 hinge_vel
HINGE_DOF_COUNT = 69
ROOT_DOF_COUNT = 6

# Forward sim frequency (must match prepare2 for stability)
DEFAULT_SIM_FREQ = 480    # Hz physics  (was 240, matching prepare2)
DEFAULT_CONTROL_FREQ = 30 # Hz policy

# Root PD gains = skyhook (matching prepare2/pd_utils.py)
ROOT_POS_KP = 2000.0    # N/m
ROOT_POS_KD = 400.0
ROOT_ROT_KP = 1000.0    # Nm/rad
ROOT_ROT_KD = 200.0

# Per-body hinge PD gains (from prepare2/pd_utils.py)
BODY_GAINS = {
    "L_Hip": (300, 30),   "L_Knee": (300, 30),   "L_Ankle": (200, 20),
    "L_Toe": (100, 10),   "R_Hip": (300, 30),    "R_Knee": (300, 30),
    "R_Ankle": (200, 20), "R_Toe": (100, 10),
    "Torso": (500, 50),   "Spine": (500, 50),    "Chest": (500, 50),
    "Neck": (200, 20),    "Head": (100, 10),
    "L_Thorax": (200, 20), "L_Shoulder": (200, 20), "L_Elbow": (150, 15),
    "L_Wrist": (100, 10),  "L_Hand": (50, 5),
    "R_Thorax": (200, 20), "R_Shoulder": (200, 20), "R_Elbow": (150, 15),
    "R_Wrist": (100, 10),  "R_Hand": (50, 5),
}
DEFAULT_HINGE_KP = 200.0  # fallback
DEFAULT_HINGE_KD = 20.0

# Armature for numerical stability (must match prepare2)
ARMATURE_HINGE = 0.5
ARMATURE_ROOT = 5.0

# Torque limit (matching prepare2)
TORQUE_LIMIT = 1000.0  # Nm

# Foot contact pattern (matching prepare2/pd_utils.py)
FOOT_SHAPE_PATTERN = "*Ankle*"

# Action clip range for policy offsets (radians)
ACTION_CLIP_RANGE = 0.2   # ±0.2 rad ≈ ±11° (was 0.5)

# Action regularization: penalise large offsets to discourage
# fighting the PD controller (DeepMimic does NOT include this,
# but it is needed here because the skyhook PD already tracks well)
ACTION_REG_WEIGHT = 0.1   # reward -= w * mean(action²)
ACTION_REG_SCALE = 5.0    # exp(-scale * mean(action²))

# Default reward weights (DeepMimic style)
DEFAULT_REWARD_WEIGHTS = {
    "pose_w": 0.45,
    "vel_w": 0.1,
    "root_pose_w": 0.2,
    "root_vel_w": 0.1,
    "key_pos_w": 0.05,
    "action_reg_w": 0.1,
}
DEFAULT_REWARD_SCALES = {
    "pose_scale": 5.0,
    "vel_scale": 0.1,
    "root_pose_scale": 5.0,
    "root_vel_scale": 2.0,
    "key_pos_scale": 10.0,
}

# Early termination
FALL_HEIGHT_THRESHOLD = 0.3   # root height below which episode terminates
MAX_ROOT_ROT_DEVIATION = 3.0  # radians — if root tilts too far from ref


# ═══════════════════════════════════════════════════════════════
# Helper: load MimicKit-format motion
# ═══════════════════════════════════════════════════════════════
def load_motion_data(motion_path):
    """
    Load a MimicKit-style .pkl motion file.

    Returns:
        frames: (T, 75) array — [root_pos(3), root_expmap(3), dof_pos(69)]
        fps: int
        loop_mode: int (0=CLAMP, 1=WRAP)
    """
    with open(motion_path, "rb") as f:
        data = pickle.load(f)

    frames = np.array(data["frames"], dtype=np.float32)
    fps = int(data["fps"])
    loop_mode = int(data["loop_mode"])
    return frames, fps, loop_mode


def _exp_map_to_quat(e):
    """Convert exponential map (3,) to quaternion (4,) in [x, y, z, w] format."""
    angle = np.linalg.norm(e)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = e / angle
    half = angle * 0.5
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)],
                    dtype=np.float32)


def _quat_to_exp_map(q):
    """Convert quaternion (4,) [x,y,z,w] to exponential map (3,)."""
    x, y, z, w = q
    # Ensure w >= 0 for shortest path
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = np.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = np.array([x, y, z], dtype=np.float32) / sin_half
    return axis * angle


def _quat_angle_diff(q1, q2):
    """Compute geodesic angle between two quaternions (xyzw)."""
    dot = np.clip(np.abs(np.sum(q1 * q2)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _heading_rotation_inv(quat_xyzw):
    """Extract heading (yaw) inverse rotation matrix from quaternion.

    Projects the quaternion to its yaw component (rotation about Z-up),
    then returns the inverse (transpose) of the 3x3 rotation matrix.

    Args:
        quat_xyzw: (4,) quaternion in [x, y, z, w] format

    Returns:
        R_inv: (3, 3) heading-inverse rotation matrix
    """
    x, y, z, w = quat_xyzw
    # Yaw angle from quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    # R_heading_inv = R_heading^T (rotation matrix transposed)
    R_inv = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return R_inv


def _frame_to_joint_q(frame):
    """Convert MimicKit frame (75,) → Newton joint_q (76,).

    frame  = [root_pos(3), root_expmap(3), dof_pos(69)]
    joint_q = [root_pos(3), root_quat_xyzw(4), dof_pos(69)]
    """
    root_pos = frame[:3]
    root_quat = _exp_map_to_quat(frame[3:6])
    dof_pos = frame[6:]
    return np.concatenate([root_pos, root_quat, dof_pos]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Newton Mimic Environment
# ═══════════════════════════════════════════════════════════════
class NewtonMimicEnv:
    """
    Gymnasium-style environment for single-character motion tracking
    using Newton's SolverMuJoCo (forward-only, no differentiable sim).

    Action space: (69,) hinge DOF torques (root DOFs always zero)
    Observation space: heading-invariant proprioceptive features

    The observation vector (per step):
        - root_height (1)
        - local root orientation (6): gravity + up in body frame
        - local root velocity (3)
        - local root angular velocity (3)
        - joint positions (69): hinge angles
        - joint velocities (69): hinge angular velocities
        - phase encoding (2): sin(phase), cos(phase)
        - (optional) target joint angles for next step (69)
        Total: 1 + 6 + 3 + 3 + 69 + 69 + 2 + 69 = 222
    """

    def __init__(self, config):
        """
        Args:
            config: dict with keys:
                - motion_file: path to .pkl motion file
                - betas_file: path to .npy betas (10,) array
                - device: str (default "cuda:0")
                - sim_freq: int (default 240)
                - control_freq: int (default 30) — policy frequency
                - control_mode: "torque" | "pd" (default "pd")
                - max_episode_length: float seconds (default 10.0)
                - enable_early_termination: bool (default True)
                - reward_weights: dict (optional override)
                - reward_scales: dict (optional override)
                - rand_init: bool — randomize start time (default True)
                - enable_tar_obs: bool — include target in obs (default True)
        """
        self.device = config.get("device", "cuda:0")
        self.sim_freq = config.get("sim_freq", DEFAULT_SIM_FREQ)
        self.control_freq = config.get("control_freq", DEFAULT_CONTROL_FREQ)
        self.sim_substeps = self.sim_freq // self.control_freq
        self.sim_dt = 1.0 / self.sim_freq
        self.control_mode = config.get("control_mode", "pd")
        self.max_episode_length = config.get("max_episode_length", 10.0)
        self.enable_early_termination = config.get("enable_early_termination", True)
        self.rand_init = config.get("rand_init", True)
        self.enable_tar_obs = config.get("enable_tar_obs", True)

        # Reward weights & scales
        rw = dict(DEFAULT_REWARD_WEIGHTS)
        rw.update(config.get("reward_weights", {}))
        self.reward_weights = rw

        rs = dict(DEFAULT_REWARD_SCALES)
        rs.update(config.get("reward_scales", {}))
        self.reward_scales = rs

        # ── Load reference motion ────────────────────────────
        motion_file = config["motion_file"]
        self.ref_frames, self.ref_fps, self.loop_mode = load_motion_data(motion_file)
        self.ref_T = self.ref_frames.shape[0]
        self.ref_duration = self.ref_T / self.ref_fps

        # Pre-compute joint_q for each reference frame
        self.ref_joint_q = np.array([_frame_to_joint_q(f) for f in self.ref_frames],
                                    dtype=np.float32)  # (T, 76)

        # ── Build Newton model from betas ────────────────────
        wp, newton = _ensure_physics_imports()
        from prepare3.xml_builder import get_or_create_xml

        betas_file = config["betas_file"]
        self.betas = np.load(betas_file)
        xml_path = get_or_create_xml(self.betas)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        # Validate model
        assert self.model.body_count == BODIES_PER_PERSON, (
            f"Expected {BODIES_PER_PERSON} bodies, got {self.model.body_count}"
        )

        # ── Configure model properties ───────────────────────
        self._setup_model()

        # ── Solver ───────────────────────────────────────────
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            njmax=450,
            nconmax=150,
            impratio=10,
            iterations=100,
            ls_iterations=50,
        )

        # ── States ───────────────────────────────────────────
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # ── PD gains (only used in "pd" control mode) ────────
        self._build_pd_gains()

        # ── Identify key body indices ────────────────────────
        self._find_key_body_ids()

        # ── Observation / action dimensions ──────────────────
        self.action_dim = HINGE_DOF_COUNT  # 69
        self.obs_dim = self._compute_obs_dim()

        # ── Episode state ────────────────────────────────────
        self.time = 0.0
        self.motion_time = 0.0
        self._step_count = 0

    def _setup_model(self):
        """Configure model properties: disable passive springs, set armature."""
        n_dof = self.model.joint_dof_count

        # Disable passive springs — we apply explicit control
        self.model.mujoco.dof_passive_stiffness.fill_(0.0)
        self.model.mujoco.dof_passive_damping.fill_(0.0)
        self.model.joint_target_ke.fill_(0.0)
        self.model.joint_target_kd.fill_(0.0)

        # Armature for stability
        arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
        arm[:ROOT_DOF_COUNT] = ARMATURE_ROOT
        wp, _ = _ensure_physics_imports()
        self.model.joint_armature = wp.array(arm, dtype=wp.float32,
                                             device=self.device)

    def _build_pd_gains(self):
        """Build per-DOF PD gain arrays matching prepare2/pd_utils.py.

        Root DOFs get skyhook gains (virtual forces to keep pelvis
        at the reference position/orientation). Hinge DOFs get
        per-body gains tuned for each joint type.
        """
        n_dof = self.model.joint_dof_count
        self.kp = np.zeros(n_dof, dtype=np.float32)
        self.kd = np.zeros(n_dof, dtype=np.float32)

        # Root PD gains (skyhook)
        self.kp[0:3] = ROOT_POS_KP
        self.kd[0:3] = ROOT_POS_KD
        self.kp[3:6] = ROOT_ROT_KP
        self.kd[3:6] = ROOT_ROT_KD

        # Per-body hinge gains (23 bodies × 3 DOF each)
        for b_idx in range(23):   # bodies 1..23
            s = ROOT_DOF_COUNT + b_idx * 3
            body_label = self.model.body_label[1 + b_idx]
            body_name = body_label.rsplit('/', 1)[-1]
            k, kd_val = BODY_GAINS.get(body_name, (DEFAULT_HINGE_KP, DEFAULT_HINGE_KD))
            self.kp[s:s + 3] = k
            self.kd[s:s + 3] = kd_val

    def _find_key_body_ids(self):
        """Identify important body indices for reward computation.

        Key bodies = hands, feet, head — endpoints that matter most for
        visual quality of motion tracking.
        """
        key_names = {"L_Hand", "R_Hand", "L_Ankle", "R_Ankle",
                     "L_Toe", "R_Toe", "Head"}
        self.key_body_ids = []
        for i in range(self.model.body_count):
            label = self.model.body_label[i]
            name = label.rsplit("/", 1)[-1]
            if name in key_names:
                self.key_body_ids.append(i)

        # Contact bodies (non-foot contacts trigger termination)
        foot_names = {"L_Ankle", "R_Ankle", "L_Toe", "R_Toe"}
        self.contact_body_ids = []
        for i in range(self.model.body_count):
            label = self.model.body_label[i]
            name = label.rsplit("/", 1)[-1]
            if name not in foot_names and name != "Pelvis":
                self.contact_body_ids.append(i)

    def _compute_obs_dim(self):
        """Compute total observation dimension."""
        dim = 0
        dim += 1       # root height
        dim += 6       # local root orientation (gravity_local + up_local)
        dim += 3       # local root linear velocity
        dim += 3       # local root angular velocity
        dim += HINGE_DOF_COUNT  # joint angles (69)
        dim += HINGE_DOF_COUNT  # joint velocities (69)
        dim += 2       # phase (sin, cos)
        if self.enable_tar_obs:
            dim += HINGE_DOF_COUNT  # target joint angles (69)
        return dim

    # ───────────────────────────────────────────────────────────
    # Gymnasium interface
    # ───────────────────────────────────────────────────────────

    def reset(self, seed=None):
        """Reset environment. Returns (observation, info)."""
        if seed is not None:
            np.random.seed(seed)

        # Choose start time in reference motion
        if self.rand_init:
            t0 = np.random.uniform(0.0, self.ref_duration * 0.9)
        else:
            t0 = 0.0

        self.motion_time = t0
        self.time = 0.0
        self._step_count = 0
        self._last_action = None  # no action yet

        # Get reference frame at t0
        ref_q = self._get_ref_joint_q(t0)
        self._cached_ref_q = ref_q  # cache for PD computation

        # Set initial state
        wp, newton = _ensure_physics_imports()
        self.state_0.joint_q = wp.array(
            ref_q, dtype=wp.float32, device=self.device
        )
        self.state_0.joint_qd = wp.zeros(
            DOFS_PER_PERSON, dtype=wp.float32, device=self.device
        )

        # Forward kinematics to initialize body transforms
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
        )

        obs = self._compute_observation()
        info = {"motion_time": self.motion_time}
        return obs, info

    def step(self, action):
        """
        Step the environment.

        Args:
            action: (69,) numpy array — hinge DOF commands

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self.action_dim,), (
            f"Expected action shape ({self.action_dim},), got {action.shape}"
        )

        # ── Simulate substeps (PD recomputed each substep for stability) ──
        wp, _ = _ensure_physics_imports()
        # Cache reference once per control step (same for all substeps)
        self._cached_ref_q = self._get_ref_joint_q(self.motion_time)
        self._last_action = action.copy()  # store for reward regularization
        for _ in range(self.sim_substeps):
            full_tau = self._compute_pd_torques(action)
            self.control.joint_f = wp.array(
                full_tau, dtype=wp.float32, device=self.device
            )
            contacts = self.model.collide(self.state_0)
            self.solver.step(
                self.state_0, self.state_1,
                self.control, contacts, self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        # ── Advance time ─────────────────────────────────────
        self.time += 1.0 / self.control_freq
        self.motion_time += 1.0 / self.control_freq
        self._step_count += 1

        # Wrap motion time for looping motions
        if self.loop_mode == 1:  # WRAP
            self.motion_time = self.motion_time % self.ref_duration

        # ── Compute obs, reward, done ────────────────────────
        obs = self._compute_observation()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.time >= self.max_episode_length
        info = {
            "motion_time": self.motion_time,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    # ───────────────────────────────────────────────────────────
    # Action → Torque conversion
    # ───────────────────────────────────────────────────────────

    def _compute_pd_torques(self, action):
        """Compute full PD torques including root stabilisation.

        Called at every substep (480 Hz) for stability, matching prepare2.

        Root DOFs 0..5: PD towards reference position/orientation (skyhook).
        Hinge DOFs 6..74: PD towards reference + action offset.

        Args:
            action: (69,) hinge correction offsets (clipped ±0.5 rad)

        Returns:
            tau: (75,) torque vector
        """
        cq = self.state_0.joint_q.numpy()
        cqd = self.state_0.joint_qd.numpy()

        # Use cached reference (set once per control step in step())
        ref_q = self._cached_ref_q

        full_tau = np.zeros(DOFS_PER_PERSON, dtype=np.float32)

        # ── Root position PD (virtual force / skyhook) ───────
        full_tau[0:3] = (
            self.kp[0:3] * (ref_q[:3] - cq[:3])
            - self.kd[0:3] * cqd[0:3]
        )

        # ── Root orientation PD (quaternion error → axis-angle) ──
        q_cur = cq[3:7].copy()
        qn = np.linalg.norm(q_cur)
        if qn > 1e-8:
            q_cur /= qn

        # Inline quaternion error → rotation vector (avoids scipy call)
        # err_q = ref_q * cur_q^-1
        ref_xyzw = ref_q[3:7]
        cur_inv = np.array([-q_cur[0], -q_cur[1], -q_cur[2], q_cur[3]])
        # quaternion multiply: ref * cur_inv
        x1, y1, z1, w1 = ref_xyzw
        x2, y2, z2, w2 = cur_inv
        err_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        err_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        err_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        err_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        # Ensure shortest path
        if err_w < 0:
            err_x, err_y, err_z, err_w = -err_x, -err_y, -err_z, -err_w
        # Convert to axis-angle (rotation vector)
        sin_half = np.sqrt(err_x*err_x + err_y*err_y + err_z*err_z)
        if sin_half > 1e-8:
            angle = 2.0 * np.arctan2(sin_half, err_w)
            R_err = np.array([err_x, err_y, err_z]) * (angle / sin_half)
        else:
            R_err = np.zeros(3, dtype=np.float32)

        full_tau[3:6] = (
            self.kp[3:6] * R_err
            - self.kd[3:6] * cqd[3:6]
        )

        # ── Hinge PD (reference + action offset) ────────────
        q_hinge = cq[7:]
        qd_hinge = cqd[ROOT_DOF_COUNT:]
        ref_hinge = ref_q[7:]

        # Action = offset from reference (bounded ±ACTION_CLIP_RANGE)
        action_clipped = np.clip(action, -ACTION_CLIP_RANGE, ACTION_CLIP_RANGE)
        target = ref_hinge + action_clipped

        full_tau[ROOT_DOF_COUNT:] = (
            self.kp[ROOT_DOF_COUNT:] * (target - q_hinge)
            - self.kd[ROOT_DOF_COUNT:] * qd_hinge
        )

        # Clip torques
        full_tau = np.clip(full_tau, -TORQUE_LIMIT, TORQUE_LIMIT)

        return full_tau

    # ───────────────────────────────────────────────────────────
    # Observations (heading-invariant, local coordinates)
    # ───────────────────────────────────────────────────────────

    def _compute_observation(self):
        """
        Build heading-invariant observation vector.

        All world-frame quantities are rotated into the character's
        heading frame (yaw-only). Global X/Y position is excluded.

        Components:
            - root_height (1)
            - local_root_orientation (6): gravity + up in body frame
            - local_root_vel (3): heading-relative
            - local_root_ang_vel (3): heading-relative
            - joint_angles (69)
            - joint_velocities (69)
            - phase (2): sin(2πφ), cos(2πφ)
            - [optional] target_joint_angles (69)
        """
        cq = self.state_0.joint_q.numpy()    # (76,)
        cqd = self.state_0.joint_qd.numpy()  # (75,)

        root_pos = cq[:3]
        root_quat = cq[3:7]  # xyzw
        root_vel = cqd[:3]
        root_ang_vel = cqd[3:6]
        hinge_pos = cq[7:]
        hinge_vel = cqd[ROOT_DOF_COUNT:]

        # Heading-inverse rotation matrix (yaw only, Z-up)
        R_inv = _heading_rotation_inv(root_quat)

        obs_parts = []

        # 1. Root height (1)
        obs_parts.append(np.array([root_pos[2]], dtype=np.float32))

        # 2. Local root orientation (6)
        # Gravity and up vectors in body-local frame
        # Inline quaternion → rotation matrix (xyzw format)
        x, y, z, w = root_quat
        R_body = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        gravity_local = R_body.T @ gravity_world  # body-frame gravity
        up_local = R_body.T @ up_world            # body-frame up
        obs_parts.append(gravity_local.astype(np.float32))
        obs_parts.append(up_local.astype(np.float32))

        # 3. Local root velocity (3) — heading-relative
        local_vel = R_inv @ root_vel
        obs_parts.append(local_vel.astype(np.float32))

        # 4. Local root angular velocity (3) — heading-relative
        local_ang_vel = R_inv @ root_ang_vel
        obs_parts.append(local_ang_vel.astype(np.float32))

        # 5. Joint angles (69)
        obs_parts.append(hinge_pos.astype(np.float32))

        # 6. Joint velocities (69)
        obs_parts.append(hinge_vel.astype(np.float32))

        # 7. Phase encoding (2)
        phase = self.motion_time / max(self.ref_duration, 1e-6)
        if self.loop_mode == 1:
            phase = phase % 1.0
        else:
            phase = min(phase, 1.0)
        obs_parts.append(np.array([
            np.sin(2.0 * np.pi * phase),
            np.cos(2.0 * np.pi * phase),
        ], dtype=np.float32))

        # 8. Target joint angles for next step (69, optional)
        if self.enable_tar_obs:
            tar_q = self._get_ref_joint_q(self.motion_time)
            tar_hinge = tar_q[7:]  # reference hinge angles
            obs_parts.append(tar_hinge.astype(np.float32))

        obs = np.concatenate(obs_parts)
        # Clamp NaN/Inf from diverged simulation states
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = np.clip(obs, -1e6, 1e6)
        return obs

    # ───────────────────────────────────────────────────────────
    # Reward (DeepMimic-style tracking)
    # ───────────────────────────────────────────────────────────

    def _compute_reward(self):
        """
        DeepMimic-style tracking reward:

        r = w_p * exp(-s_p * pose_err)
          + w_v * exp(-s_v * vel_err)
          + w_rp * exp(-s_rp * root_pose_err)
          + w_rv * exp(-s_rv * root_vel_err)
          + w_kp * exp(-s_kp * key_pos_err)

        All comparisons done in local / heading-relative coordinates
        to avoid rewarding global position tracking.
        """
        cq = self.state_0.joint_q.numpy()
        cqd = self.state_0.joint_qd.numpy()

        ref_q = self._get_ref_joint_q(self.motion_time)
        ref_qd = self._get_ref_joint_qd(self.motion_time)

        # ── 1. Pose error (hinge angles) ─────────────────────
        pose_diff = ref_q[7:] - cq[7:]
        pose_err = np.sum(pose_diff * pose_diff)

        # ── 2. Velocity error (hinge velocities) ─────────────
        vel_diff = ref_qd[ROOT_DOF_COUNT:] - cqd[ROOT_DOF_COUNT:]
        vel_err = np.sum(vel_diff * vel_diff)

        # ── 3. Root pose error ───────────────────────────────
        # Height difference only (no X/Y tracking for heading-invariance)
        root_h_diff = ref_q[2] - cq[2]
        root_h_err = root_h_diff * root_h_diff

        # Root orientation (geodesic angle)
        root_rot_err = _quat_angle_diff(cq[3:7], ref_q[3:7])
        root_rot_err = root_rot_err * root_rot_err

        root_pose_err = root_h_err + 0.1 * root_rot_err

        # ── 4. Root velocity error (heading-relative) ────────
        R_inv = _heading_rotation_inv(cq[3:7])
        local_vel = R_inv @ cqd[:3]
        ref_R_inv = _heading_rotation_inv(ref_q[3:7])
        ref_local_vel = ref_R_inv @ ref_qd[:3]
        root_vel_diff = ref_local_vel - local_vel
        root_vel_err = np.sum(root_vel_diff * root_vel_diff)

        local_ang_vel = R_inv @ cqd[3:6]
        ref_local_ang_vel = ref_R_inv @ ref_qd[3:6]
        root_ang_vel_diff = ref_local_ang_vel - local_ang_vel
        root_ang_vel_err = np.sum(root_ang_vel_diff * root_ang_vel_diff)

        root_vel_total = root_vel_err + 0.1 * root_ang_vel_err

        # ── 5. Key position error (root-relative) ───────────
        key_pos_err = 0.0
        if len(self.key_body_ids) > 0:
            # Get body positions from sim state
            body_pos = self._get_body_positions()
            ref_body_pos = self._get_ref_body_positions(self.motion_time)

            if body_pos is not None and ref_body_pos is not None:
                # Root-relative key body positions
                sim_root = cq[:3]
                ref_root = ref_q[:3]

                for bid in self.key_body_ids:
                    if bid < body_pos.shape[0] and bid < ref_body_pos.shape[0]:
                        sim_local = body_pos[bid] - sim_root
                        ref_local = ref_body_pos[bid] - ref_root
                        diff = ref_local - sim_local
                        key_pos_err += np.sum(diff * diff)

        # ── Combine rewards ──────────────────────────────────
        w = self.reward_weights
        s = self.reward_scales

        pose_r = np.exp(-s["pose_scale"] * pose_err)
        vel_r = np.exp(-s["vel_scale"] * vel_err)
        root_pose_r = np.exp(-s["root_pose_scale"] * root_pose_err)
        root_vel_r = np.exp(-s["root_vel_scale"] * root_vel_total)
        key_pos_r = np.exp(-s["key_pos_scale"] * key_pos_err)

        # ── 6. Action regularization ────────────────────────
        action_reg_r = 1.0  # default (no action yet / zero action)
        if hasattr(self, '_last_action') and self._last_action is not None:
            action_sq = np.mean(self._last_action ** 2)
            action_reg_r = np.exp(-ACTION_REG_SCALE * action_sq)

        reward = (
            w["pose_w"] * pose_r
            + w["vel_w"] * vel_r
            + w["root_pose_w"] * root_pose_r
            + w["root_vel_w"] * root_vel_r
            + w["key_pos_w"] * key_pos_r
            + w.get("action_reg_w", 0.0) * action_reg_r
        )

        return float(reward)

    # ───────────────────────────────────────────────────────────
    # Termination
    # ───────────────────────────────────────────────────────────

    def _check_termination(self):
        """Check if episode should terminate early."""
        if not self.enable_early_termination:
            return False

        cq = self.state_0.joint_q.numpy()

        # 1. Root height too low (character has fallen)
        root_height = cq[2]
        if root_height < FALL_HEIGHT_THRESHOLD:
            return True

        # 2. Root orientation too far from reference
        ref_q = self._get_ref_joint_q(self.motion_time)
        rot_diff = _quat_angle_diff(cq[3:7], ref_q[3:7])
        if rot_diff > MAX_ROOT_ROT_DEVIATION:
            return True

        # 3. NaN/Inf detection
        if not np.all(np.isfinite(cq)):
            return True

        # 4. Motion ended (non-looping only)
        if self.loop_mode == 0 and self.motion_time >= self.ref_duration:
            return True

        return False

    # ───────────────────────────────────────────────────────────
    # Reference motion interpolation
    # ───────────────────────────────────────────────────────────

    def _get_ref_joint_q(self, t):
        """Get reference joint_q at time t (linearly interpolated).

        Args:
            t: time in seconds

        Returns:
            joint_q: (76,) array
        """
        if self.loop_mode == 1:  # WRAP
            t = t % self.ref_duration

        # Continuous frame index
        frame_f = t * self.ref_fps
        frame_f = np.clip(frame_f, 0.0, self.ref_T - 1.0)
        f0 = int(frame_f)
        f1 = min(f0 + 1, self.ref_T - 1)
        alpha = frame_f - f0

        q0 = self.ref_joint_q[f0]
        q1 = self.ref_joint_q[f1]

        # Interpolate position and hinge DOFs linearly
        q_interp = (1.0 - alpha) * q0 + alpha * q1

        # Interpolate root quaternion via SLERP
        quat0 = q0[3:7]
        quat1 = q1[3:7]
        q_interp[3:7] = self._slerp(quat0, quat1, alpha)

        return q_interp

    def _get_ref_joint_qd(self, t):
        """Get reference joint velocities at time t (finite differences).

        Returns:
            joint_qd: (75,) array — [root_vel(3), root_ang_vel(3), hinge_vel(69)]
        """
        dt_ref = 1.0 / self.ref_fps
        q_curr = self._get_ref_joint_q(t)
        q_next = self._get_ref_joint_q(t + dt_ref)

        qd = np.zeros(DOFS_PER_PERSON, dtype=np.float32)

        # Root linear velocity
        qd[:3] = (q_next[:3] - q_curr[:3]) / dt_ref

        # Root angular velocity (from quaternion difference)
        quat_curr = q_curr[3:7]
        quat_next = q_next[3:7]
        dq = _quat_angle_diff(quat_curr, quat_next)
        if dq > 1e-6:
            # axis-angle from quaternion difference
            exp_diff = _quat_to_exp_map(
                self._quat_mul(
                    quat_next,
                    self._quat_inv(quat_curr)
                )
            )
            qd[3:6] = exp_diff / dt_ref
        else:
            qd[3:6] = 0.0

        # Hinge velocities
        qd[ROOT_DOF_COUNT:] = (q_next[7:] - q_curr[7:]) / dt_ref

        return qd

    def _get_body_positions(self):
        """Get current body positions from simulation state.

        Returns:
            positions: (n_bodies, 3) numpy array, or None if unavailable
        """
        try:
            body_q = self.state_0.body_q.numpy()
            # body_q is (n_bodies, 7): [px, py, pz, qx, qy, qz, qw]
            n_bodies = body_q.shape[0]
            positions = body_q[:, :3].copy()
            return positions
        except Exception:
            return None

    def _get_ref_body_positions(self, t):
        """Get reference body positions via forward kinematics.

        This creates a temporary state, sets reference joint_q, runs
        FK, and reads body positions. Only used for key-point reward.

        Returns:
            positions: (n_bodies, 3) numpy array, or None if unavailable
        """
        try:
            ref_q = self._get_ref_joint_q(t)
            # Use a temporary state for FK
            tmp_state = self.model.state()
            wp, newton = _ensure_physics_imports()
            tmp_state.joint_q = wp.array(
                ref_q, dtype=wp.float32, device=self.device
            )
            tmp_state.joint_qd = wp.zeros(
                DOFS_PER_PERSON, dtype=wp.float32, device=self.device
            )
            newton.eval_fk(
                self.model,
                tmp_state.joint_q,
                tmp_state.joint_qd,
                tmp_state,
            )
            body_q = tmp_state.body_q.numpy()
            return body_q[:, :3].copy()
        except Exception:
            return None

    # ───────────────────────────────────────────────────────────
    # Quaternion utilities (numpy, single-quaternion)
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _slerp(q0, q1, t):
        """Spherical linear interpolation between two quaternions (xyzw)."""
        dot = np.sum(q0 * q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        result = s0 * q0 + s1 * q1
        return result / np.linalg.norm(result)

    @staticmethod
    def _quat_inv(q):
        """Quaternion inverse (for unit quaternion = conjugate). xyzw format."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    @staticmethod
    def _quat_mul(q1, q2):
        """Quaternion multiplication. xyzw format."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float32)

    # ───────────────────────────────────────────────────────────
    # Utility
    # ───────────────────────────────────────────────────────────

    def get_action_dim(self):
        return self.action_dim

    def get_obs_dim(self):
        return self.obs_dim

    def get_ref_motion_duration(self):
        return self.ref_duration

    def get_ref_fps(self):
        return self.ref_fps

    def close(self):
        """Clean up resources."""
        pass
