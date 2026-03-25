"""
actor_network.py — Neural actor for interaction torque correction.

Architecture:
    MLP: observation → Δq (hinge angle offsets)

    Input observation (per person per frame):
        ref_hinge_angles (69)
      + ego_relative_other_positions (66)   ← in ego root frame
      + solo_hinge_torques (69)
      + root_height (1)                     ← scalar, invariant
      + root_local_orientation (6)          ← heading-projected (sin,cos + pitch,roll via gravity)
      + normalized_time (1)
      = 212 total

    Output: Δq (69) hinge angle offsets

    All spatial features are expressed in the ego character's local
    (heading-aligned) frame, making the actor invariant to global
    translation and yaw rotation.

    Zero-initialized output layer ensures the network starts as identity
    (Δq = 0 → same as baseline solo-torque behavior). The network learns
    residual interaction corrections during training.

Gradient bridge (manual, no torch.autograd.Function):
    1. Actor forward  (PyTorch)  → Δq tensor with grad_fn
    2. Detach + scatter to flat   → numpy → Warp array (requires_grad=True)
    3. Warp simulation + tape.backward → gradient on Warp array
    4. Extract hinge gradients    → numpy → torch tensor
    5. delta_q.backward(gradient) → gradients flow to actor parameters
    6. PyTorch optimizer.step()

Usage:
    from prepare2.actor_network import InteractionActor, build_observations
    actor = InteractionActor().to("cuda:0")
    obs = build_observations(ref_jq, ref_pos, torques, w_start, w_end, T)
    delta_q = actor(torch.from_numpy(obs).cuda())  # (batch, 69)
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial.transform import Rotation

# ═══════════════════════════════════════════════════════════════
# Constants (must match optimize_interaction.py)
# ═══════════════════════════════════════════════════════════════
DOFS_PER_PERSON = 75        # 6 root + 69 hinge
COORDS_PER_PERSON = 76      # 7 root (3pos + 4quat) + 69 hinge
N_HINGE_DOFS = 69           # hinge-only DOFs per person
N_SMPL_JOINTS = 22          # tracked SMPL joints for loss
N_BODIES_PER_PERSON = 24    # Newton bodies per person

# Observation feature dimensions
OBS_REF_HINGE = N_HINGE_DOFS          # 69
OBS_OTHER_POS = N_SMPL_JOINTS * 3     # 66  (ego-relative positions)
OBS_SOLO_TORQUE = N_HINGE_DOFS        # 69
OBS_ROOT_LOCAL = 7                     # height(1) + heading_sin_cos(2) + gravity_proj(4)
OBS_TIME = 1                           # normalized frame index
OBS_DIM = (OBS_REF_HINGE + OBS_OTHER_POS + OBS_SOLO_TORQUE
           + OBS_ROOT_LOCAL + OBS_TIME)  # 212


# ═══════════════════════════════════════════════════════════════
# Coordinate Transform Utilities
# ═══════════════════════════════════════════════════════════════
def _quat_to_heading_rot(quat_xyzw: np.ndarray) -> np.ndarray:
    """Extract yaw rotation matrix from quaternion (Z-up convention).

    Projects the quaternion onto the XY-plane heading, returning
    a 3×3 rotation matrix R_heading that, when applied as R_heading.T @ v,
    transforms world vectors into the heading-aligned local frame.

    Args:
        quat_xyzw: (4,) quaternion in [x, y, z, w] format.

    Returns:
        R_heading: (3, 3) rotation matrix representing only the yaw component.
    """
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    # Forward direction = R @ [1, 0, 0] projected onto XY plane
    fwd = R[:, 0].copy()
    fwd[2] = 0.0
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    fwd /= norm
    yaw = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _quat_to_local_orientation(quat_xyzw: np.ndarray) -> np.ndarray:
    """Compute heading-invariant local orientation features.

    Returns a 6-dim vector:
      [sin(yaw_relative), cos(yaw_relative),  ← always (0, 1) since we
       gravity_x, gravity_y, gravity_z,         remove heading ourselves
       root_height_placeholder]                  ← filled by caller

    Actually we return:
      [sin(heading), cos(heading),               ← of ego heading in world
       local_gravity_x, local_gravity_y,         ← R_full.T @ [0,0,-1]
       local_gravity_z, 0]                       ← last dim unused padding

    But since we already remove heading in the transform, we use a
    simpler representation: just the gravity vector in the full body frame
    (captures pitch and roll) + padding to maintain OBS_ROOT_LOCAL=7.

    Args:
        quat_xyzw: (4,) quaternion in xyzw format.

    Returns:
        local_ori: (6,) float array [grav_local(3) + up_local(3)]
    """
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    # Gravity in local frame: R.T @ [0, 0, -9.81] normalized
    gravity_world = np.array([0.0, 0.0, -1.0])
    gravity_local = R.T @ gravity_world  # (3,)
    # Up vector in local frame
    up_world = np.array([0.0, 0.0, 1.0])
    up_local = R.T @ up_world  # (3,)
    return np.concatenate([gravity_local, up_local]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Neural Network
# ═══════════════════════════════════════════════════════════════
class InteractionActor(nn.Module):
    """MLP actor: observation → Δq (joint angle offsets).

    Zero-initialization:
        The final linear layer has zero weight and bias, ensuring
        Δq = 0 at initialization → baseline solo-torque behavior.
        The network learns residual corrections during training.

    Args:
        obs_dim:      Input dimension (default 212).
        output_dim:   Output dimension (default 69 hinge DOFs).
        hidden_dims:  Tuple of hidden layer widths (default (256, 256)).
        activation:   "relu", "elu", or "tanh".
        max_delta:    Maximum Δq magnitude in radians (default 0.05).
                      Output is tanh-bounded to [-max_delta, +max_delta],
                      preventing unbounded growth that destabilizes the
                      PD controller and simulation.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        output_dim: int = N_HINGE_DOFS,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: str = "relu",
        max_delta: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        self.max_delta = max_delta

        # Build MLP
        act_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
        if activation not in act_map:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_map[activation]())
            in_dim = h_dim

        # Output layer — zero-initialized
        output_layer = nn.Linear(in_dim, output_dim)
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) observation tensor.
        Returns:
            delta_q: (batch, output_dim) hinge angle offsets,
                     bounded to [-max_delta, +max_delta] via tanh.
        """
        return torch.tanh(self.net(obs)) * self.max_delta

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# Observation Normalizer
# ═══════════════════════════════════════════════════════════════
class ObservationNormalizer:
    """Running mean/std normalizer (Welford's algorithm).

    Computes statistics from training data and normalizes observations
    to approximately zero mean, unit variance. Can be serialized via
    state_dict() / load_state_dict().
    """

    def __init__(self, obs_dim: int = OBS_DIM, epsilon: float = 1e-8):
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self._M2 = np.zeros(obs_dim, dtype=np.float64)

    def update(self, obs_batch: np.ndarray):
        """Update statistics with a batch of observations.

        Args:
            obs_batch: (N, obs_dim) array.
        """
        for obs in obs_batch:
            self.count += 1
            delta = obs.astype(np.float64) - self.mean
            self.mean += delta / self.count
            delta2 = obs.astype(np.float64) - self.mean
            self._M2 += delta * delta2
        if self.count > 1:
            self.var = self._M2 / (self.count - 1)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations.

        Args:
            obs: (..., obs_dim) array.
        Returns:
            Normalized array with ~zero mean, ~unit variance.
        """
        std = np.sqrt(self.var + self.epsilon).astype(np.float32)
        return ((obs - self.mean.astype(np.float32)) / std).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "_M2": self._M2.copy(),
        }

    def load_state_dict(self, d: Dict[str, Any]):
        self.count = d["count"]
        self.mean = d["mean"].copy()
        self.var = d["var"].copy()
        self._M2 = d["_M2"].copy()


# ═══════════════════════════════════════════════════════════════
# Observation Building
# ═══════════════════════════════════════════════════════════════
def build_observations(
    ref_jq: List[np.ndarray],
    ref_positions: np.ndarray,
    torques_solo: List[np.ndarray],
    w_start: int,
    w_end: int,
    T: int,
) -> np.ndarray:
    """Build ego-relative observations for one window.

    All spatial features are expressed in the ego character's heading-
    aligned local frame (invariant to global translation and yaw).

    Args:
        ref_jq:        [person0_jq, person1_jq], each (T, 76) coords.
        ref_positions: (T, 2, 22, 3) reference body positions.
        torques_solo:  [person0_torq, person1_torq], each (T, 75).
        w_start:       Window start frame (inclusive).
        w_end:         Window end frame (exclusive).
        T:             Total frames in clip.

    Returns:
        obs: (window_size × 2, OBS_DIM) float32 array.
             Row order: [f0_p0, f0_p1, f1_p0, f1_p1, ...].
    """
    window_size = w_end - w_start
    n_persons = 2
    obs = np.zeros((window_size * n_persons, OBS_DIM), dtype=np.float32)

    for f in range(window_size):
        frame = w_start + f
        for p in range(n_persons):
            row = f * n_persons + p
            other = 1 - p
            col = 0

            # Ego root position and heading rotation
            ego_pos = ref_jq[p][frame, :3]       # (3,) world position
            ego_quat = ref_jq[p][frame, 3:7]     # (4,) xyzw quaternion
            R_heading = _quat_to_heading_rot(ego_quat)
            R_inv = R_heading.T                   # world → local heading frame

            # 1. Reference hinge angles (69) — already local/invariant
            obs[row, col:col + OBS_REF_HINGE] = ref_jq[p][frame, 7:76]
            col += OBS_REF_HINGE

            # 2. Other person's body positions in EGO local frame (66)
            other_world = ref_positions[frame, other]  # (22, 3)
            other_rel = (other_world - ego_pos[None, :]) @ R_inv.T  # (22, 3)
            obs[row, col:col + OBS_OTHER_POS] = other_rel.flatten().astype(np.float32)
            col += OBS_OTHER_POS

            # 3. Solo hinge torques (69) — already invariant
            obs[row, col:col + OBS_SOLO_TORQUE] = torques_solo[p][frame, 6:75]
            col += OBS_SOLO_TORQUE

            # 4. Root local state: height(1) + local_orientation(6)
            obs[row, col] = ego_pos[2]  # root height (Z-up, invariant)
            local_ori = _quat_to_local_orientation(ego_quat)  # (6,)
            obs[row, col + 1:col + 7] = local_ori
            col += OBS_ROOT_LOCAL

            # 5. Normalized time in [0, 1]
            obs[row, col] = frame / max(T - 1, 1)
            col += OBS_TIME

    return obs


def build_all_observations(
    ref_jq: List[np.ndarray],
    ref_positions: np.ndarray,
    torques_solo: List[np.ndarray],
    T: int,
) -> np.ndarray:
    """Build observations for ALL frames (inference / save).

    Returns:
        obs: (T × 2, OBS_DIM) float32 array.
    """
    return build_observations(ref_jq, ref_positions, torques_solo, 0, T, T)


# ═══════════════════════════════════════════════════════════════
# Delta Array Mapping
# ═══════════════════════════════════════════════════════════════
def scatter_delta_to_flat(
    delta_q_np: np.ndarray,
    w_start: int,
    w_end: int,
    T: int,
    n_dof: int,
    n_persons: int = 2,
) -> np.ndarray:
    """Scatter actor Δq output into flat (T × n_dof) array.

    Places 69-dim hinge Δq values at the correct positions in
    the flat delta array used by the Warp compose kernel.
    Root DOFs (0–5 per person) remain zero.

    Args:
        delta_q_np: (window_size × n_persons, 69) actor output.
        w_start, w_end: Window frame range.
        T, n_dof:   Trajectory dimensions.
        n_persons:  Number of persons (default 2).

    Returns:
        flat_delta: (T × n_dof,) float32 array.
    """
    window_size = w_end - w_start
    full = np.zeros((T, n_dof), dtype=np.float32)

    for f in range(window_size):
        frame = w_start + f
        for p in range(n_persons):
            src = delta_q_np[f * n_persons + p]        # (69,)
            d_start = p * DOFS_PER_PERSON + 6           # skip 6 root DOFs
            full[frame, d_start:d_start + N_HINGE_DOFS] = src

    return full.flatten().astype(np.float32)


def extract_hinge_grads(
    grad_flat_np: np.ndarray,
    w_start: int,
    w_end: int,
    n_dof: int,
    n_persons: int = 2,
) -> np.ndarray:
    """Extract hinge-DOF gradients from flat Warp gradient array.

    Inverse of scatter_delta_to_flat: reads gradients at hinge
    positions and reshapes to (window × persons, 69).

    Args:
        grad_flat_np: (T × n_dof,) gradient from Warp.
        w_start, w_end: Window frame range.
        n_dof:      Total DOFs.
        n_persons:  Number of persons (default 2).

    Returns:
        hinge_grads: (window_size × n_persons, 69) float32 array.
    """
    window_size = w_end - w_start
    T = len(grad_flat_np) // n_dof
    grad_2d = grad_flat_np.reshape(T, n_dof)
    result = np.zeros((window_size * n_persons, N_HINGE_DOFS),
                      dtype=np.float32)

    for f in range(window_size):
        frame = w_start + f
        for p in range(n_persons):
            d_start = p * DOFS_PER_PERSON + 6
            result[f * n_persons + p] = (
                grad_2d[frame, d_start:d_start + N_HINGE_DOFS]
            )

    return result
