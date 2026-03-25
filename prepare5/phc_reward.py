"""
phc_reward.py — PHC imitation reward function adapted for Newton.

Implements the exact reward function from PHC (humanoid_im.py lines 1524-1554):
    reward = w_pos * exp(-k_pos * mean_pos_err²)
           + w_rot * exp(-k_rot * mean_rot_err²)
           + w_vel * exp(-k_vel * mean_vel_err²)
           + w_ang_vel * exp(-k_ang_vel * mean_ang_vel_err²)

Operates on numpy arrays extracted from Newton state (body_q, body_qd).
Also computes per-body tracking errors for diagnostics.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from prepare5.phc_config import (
    REWARD_WEIGHTS, REWARD_COEFFICIENTS,
    SMPL_TO_NEWTON, N_SMPL_JOINTS, BODIES_PER_PERSON,
)


def extract_body_state(state, person_idx=0):
    """Extract body positions, rotations, velocities from Newton state.

    Args:
        state: Newton state object (has body_q, body_qd)
        person_idx: which person (0-indexed)

    Returns:
        body_pos:     (N_SMPL_JOINTS, 3) body positions
        body_rot:     (N_SMPL_JOINTS, 4) body quaternions (xyzw)
        body_vel:     (N_SMPL_JOINTS, 3) body linear velocities
        body_ang_vel: (N_SMPL_JOINTS, 3) body angular velocities
    """
    body_q = state.body_q.numpy().reshape(-1, 7)   # (n_bodies, 7) = (pos3, quat4)
    body_qd = state.body_qd.numpy().reshape(-1, 6)  # (n_bodies, 6) = (vel3, omega3)

    off = person_idx * BODIES_PER_PERSON

    body_pos = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)
    body_rot = np.zeros((N_SMPL_JOINTS, 4), dtype=np.float32)
    body_vel = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)
    body_ang_vel = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)

    for smpl_j, newton_b in SMPL_TO_NEWTON.items():
        idx = off + newton_b
        body_pos[smpl_j] = body_q[idx, :3]
        body_rot[smpl_j] = body_q[idx, 3:7]
        body_vel[smpl_j] = body_qd[idx, :3]
        body_ang_vel[smpl_j] = body_qd[idx, 3:6]

    return body_pos, body_rot, body_vel, body_ang_vel


def extract_ref_body_state(model, joint_q, joint_qd, person_idx=0, device="cuda:0"):
    """Get reference body positions/rotations via Newton FK.

    Runs forward kinematics on the reference joint_q to get body-space
    positions and rotations, which is what the reward compares against.

    Args:
        model: Newton model
        joint_q: (76,) reference joint coordinates for one frame
        joint_qd: (75,) reference joint velocities for one frame
        person_idx: which person
        device: compute device

    Returns:
        ref_body_pos:     (N_SMPL_JOINTS, 3)
        ref_body_rot:     (N_SMPL_JOINTS, 4)
        ref_body_vel:     (N_SMPL_JOINTS, 3)
        ref_body_ang_vel: (N_SMPL_JOINTS, 3)
    """
    import warp as wp
    import newton

    state = model.state()
    n_coords = model.joint_coord_count
    n_dof = model.joint_dof_count

    # Set joint coordinates
    jq_full = np.zeros(n_coords, dtype=np.float32)
    c = person_idx * 76
    jq_full[c:c + 76] = joint_q.astype(np.float32)
    state.joint_q = wp.array(jq_full, dtype=wp.float32, device=device)

    # Set joint velocities
    jqd_full = np.zeros(n_dof, dtype=np.float32)
    d = person_idx * 75
    jqd_full[d:d + 75] = joint_qd.astype(np.float32)
    state.joint_qd = wp.array(jqd_full, dtype=wp.float32, device=device)

    # Run FK
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    return extract_body_state(state, person_idx)


def quat_angle_diff(q1, q2):
    """Compute angle between two quaternion arrays.

    Args:
        q1: (N, 4) quaternions (xyzw)
        q2: (N, 4) quaternions (xyzw)

    Returns:
        angles: (N,) angle differences in radians
    """
    N = q1.shape[0]
    angles = np.zeros(N, dtype=np.float32)
    for i in range(N):
        # q_err = q1 * q2^-1
        r1 = Rotation.from_quat(q1[i])
        r2 = Rotation.from_quat(q2[i])
        r_err = r1 * r2.inv()
        angles[i] = r_err.magnitude()
    return angles


def compute_imitation_reward(
    body_pos, body_rot, body_vel, body_ang_vel,
    ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel,
    weights=None, coefficients=None,
):
    """Compute PHC-style imitation reward.

    Exactly matches PHC humanoid_im.py compute_imitation_reward():
        r_pos     = exp(-k_pos * mean(||ref_pos - sim_pos||²))
        r_rot     = exp(-k_rot * mean(angle_diff²))
        r_vel     = exp(-k_vel * mean(||ref_vel - sim_vel||²))
        r_ang_vel = exp(-k_ang_vel * mean(||ref_ω - sim_ω||²))
        reward    = w_pos*r_pos + w_rot*r_rot + w_vel*r_vel + w_ang*r_ang

    Args:
        body_pos:         (N, 3) simulated body positions
        body_rot:         (N, 4) simulated body quaternions (xyzw)
        body_vel:         (N, 3) simulated body linear velocities
        body_ang_vel:     (N, 3) simulated body angular velocities
        ref_body_pos:     (N, 3) reference body positions
        ref_body_rot:     (N, 4) reference body quaternions (xyzw)
        ref_body_vel:     (N, 3) reference body linear velocities
        ref_body_ang_vel: (N, 3) reference body angular velocities
        weights:      dict with w_pos, w_rot, w_vel, w_ang_vel
        coefficients: dict with k_pos, k_rot, k_vel, k_ang_vel

    Returns:
        reward: scalar float in [0, 1]
        components: dict with r_pos, r_rot, r_vel, r_ang_vel
    """
    if weights is None:
        weights = REWARD_WEIGHTS
    if coefficients is None:
        coefficients = REWARD_COEFFICIENTS

    w = weights
    k = coefficients

    # Position error: mean squared distance across all bodies
    pos_diff = ref_body_pos - body_pos  # (N, 3)
    pos_dist2 = np.sum(pos_diff ** 2, axis=-1).mean()  # scalar
    r_pos = np.exp(-k['k_pos'] * pos_dist2)

    # Rotation error: mean squared angle diff across all bodies
    angle_diffs = quat_angle_diff(ref_body_rot, body_rot)  # (N,)
    rot_dist2 = np.mean(angle_diffs ** 2)
    r_rot = np.exp(-k['k_rot'] * rot_dist2)

    # Velocity error
    vel_diff = ref_body_vel - body_vel  # (N, 3)
    vel_dist2 = np.sum(vel_diff ** 2, axis=-1).mean()
    r_vel = np.exp(-k['k_vel'] * vel_dist2)

    # Angular velocity error
    ang_vel_diff = ref_body_ang_vel - body_ang_vel  # (N, 3)
    ang_vel_dist2 = np.sum(ang_vel_diff ** 2, axis=-1).mean()
    r_ang_vel = np.exp(-k['k_ang_vel'] * ang_vel_dist2)

    reward = (w['w_pos'] * r_pos + w['w_rot'] * r_rot
              + w['w_vel'] * r_vel + w['w_ang_vel'] * r_ang_vel)

    components = {
        'r_pos': float(r_pos),
        'r_rot': float(r_rot),
        'r_vel': float(r_vel),
        'r_ang_vel': float(r_ang_vel),
        'reward': float(reward),
        'pos_err_m': float(np.sqrt(pos_dist2)),
        'rot_err_rad': float(np.sqrt(rot_dist2)),
    }

    return float(reward), components


def compute_tracking_errors(sim_positions, ref_positions):
    """Compute per-joint position tracking errors.

    Args:
        sim_positions: (T, N_SMPL_JOINTS, 3) simulated positions
        ref_positions: (T, N_SMPL_JOINTS, 3) reference positions

    Returns:
        dict with:
          mpjpe: mean per-joint position error (mm)
          per_joint_mpjpe: (N_SMPL_JOINTS,) per-joint MPJPE (mm)
          per_frame_mpjpe: (T,) per-frame MPJPE (mm)
          max_error: max error across all joints and frames (mm)
    """
    diff = sim_positions - ref_positions  # (T, N, 3)
    per_joint_per_frame = np.linalg.norm(diff, axis=-1)  # (T, N)

    return {
        'mpjpe_mm': float(per_joint_per_frame.mean() * 1000),
        'per_joint_mpjpe_mm': per_joint_per_frame.mean(axis=0) * 1000,
        'per_frame_mpjpe_mm': per_joint_per_frame.mean(axis=1) * 1000,
        'max_error_mm': float(per_joint_per_frame.max() * 1000),
    }
