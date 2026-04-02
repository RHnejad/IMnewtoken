"""
obs_builder.py — Observation vector construction for the PPO RL tracker.

446-dimensional observation vector per person:
  [0:66]    sim body positions, root-relative, yaw-normalised  (22×3)
  [66:154]  sim body rotations as quaternion, root-frame       (22×4)
  [154:220] sim body linear velocities, root frame             (22×3)
  [220:286] sim body angular velocities, root frame            (22×3)
  [286:287] sim root height (z)                                (1)
  [287:290] gravity vector in root frame                       (3)
  [290:356] ref body positions, root-relative                  (22×3)
  [356:444] ref body rotations, root-relative (quaternion)     (22×4)
  [444:446] phase encoding: (sin(2π·t/T), cos(2π·t/T))        (2)
Total: 66+88+66+66+1+3+66+88+2 = 446
"""
import numpy as np
from scipy.spatial.transform import Rotation

from prepare6.rl_config import N_SMPL_JOINTS, OBS_DIM_SOLO


# ── gravity in world frame ──────────────────────────────────────
_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _heading_quat_inv(root_quat_xyzw):
    """Return the inverse of the yaw-only (heading) rotation.

    Extracts yaw from root_quat, returns inverse as (xyzw).
    """
    qn = np.linalg.norm(root_quat_xyzw)
    if qn < 1e-6:
        return np.array([0., 0., 0., 1.], dtype=np.float32)  # identity
    root_quat_xyzw = root_quat_xyzw / qn
    r = Rotation.from_quat(root_quat_xyzw)
    euler = r.as_euler('zyx')   # [yaw, pitch, roll]
    yaw_only = Rotation.from_euler('z', euler[0])
    return yaw_only.inv().as_quat()   # xyzw


def _rotate_vec(quat_xyzw, vecs):
    """Rotate vectors by quaternion.

    Args:
        quat_xyzw: (4,) quaternion
        vecs: (N, 3) vectors
    Returns:
        (N, 3) rotated
    """
    r = Rotation.from_quat(quat_xyzw)
    return r.apply(vecs).astype(np.float32)


def build_obs_single(
    body_pos,      # (22, 3)  sim body positions (world frame)
    body_rot,      # (22, 4)  sim body quaternions xyzw (world frame)
    body_vel,      # (22, 3)  sim body linear velocities (world frame)
    body_ang_vel,  # (22, 3)  sim body angular velocities (world frame)
    ref_body_pos,  # (22, 3)  reference body positions (world frame)
    ref_body_rot,  # (22, 4)  reference body quaternions xyzw
    phase,         # float in [0, 1]  frame_idx / T
):
    """Build 446-dim observation vector for a single person.

    Returns:
        obs: (446,) float32
    """
    N = N_SMPL_JOINTS   # 22

    root_pos = body_pos[0]            # (3,)
    root_quat = body_rot[0]           # (4,) xyzw

    # heading-inverse rotation (yaw only)
    heading_inv_xyzw = _heading_quat_inv(root_quat)
    heading_inv_r = Rotation.from_quat(heading_inv_xyzw)

    # ── sim body positions, root-relative, yaw-normalised ──
    pos_rel = body_pos - root_pos[None, :]            # (22, 3)
    pos_local = heading_inv_r.apply(pos_rel).astype(np.float32)   # (22, 3)

    # ── sim body rotations, heading-relative ──
    root_r_inv = Rotation.from_quat(root_quat).inv()
    rot_local = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        r_rel = root_r_inv * Rotation.from_quat(body_rot[i])
        rot_local[i] = r_rel.as_quat()

    # ── sim velocities and ang-vel, heading-relative ──
    vel_local = heading_inv_r.apply(body_vel).astype(np.float32)
    ang_vel_local = heading_inv_r.apply(body_ang_vel).astype(np.float32)

    # ── root height ──
    root_height = np.array([root_pos[2]], dtype=np.float32)

    # ── gravity in root frame ──
    grav_local = heading_inv_r.apply(_GRAVITY_WORLD[None, :]).astype(np.float32)[0]

    # ── reference positions, root-relative ──
    ref_pos_rel = ref_body_pos - root_pos[None, :]
    ref_pos_local = heading_inv_r.apply(ref_pos_rel).astype(np.float32)

    # ── reference rotations, heading-relative ──
    ref_rot_local = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        r_ref_rel = root_r_inv * Rotation.from_quat(ref_body_rot[i])
        ref_rot_local[i] = r_ref_rel.as_quat()

    # ── phase encoding ──
    phase_enc = np.array([
        np.sin(2 * np.pi * phase),
        np.cos(2 * np.pi * phase),
    ], dtype=np.float32)

    obs = np.concatenate([
        pos_local.ravel(),        # 66
        rot_local.ravel(),        # 88
        vel_local.ravel(),        # 66
        ang_vel_local.ravel(),    # 66
        root_height,              # 1
        grav_local,               # 3
        ref_pos_local.ravel(),    # 66
        ref_rot_local.ravel(),    # 88
        phase_enc,                # 2
    ])
    assert obs.shape == (OBS_DIM_SOLO,), f"Obs dim mismatch: {obs.shape}"
    return obs


def build_obs_batch(
    body_pos_batch,      # (B, 22, 3)
    body_rot_batch,      # (B, 22, 4)
    body_vel_batch,      # (B, 22, 3)
    body_ang_vel_batch,  # (B, 22, 3)
    ref_body_pos_batch,  # (B, 22, 3)
    ref_body_rot_batch,  # (B, 22, 4)
    phases,              # (B,)
):
    """Build observation vectors for a batch of envs.

    Returns:
        obs: (B, 446) float32
    """
    B = body_pos_batch.shape[0]
    obs = np.zeros((B, OBS_DIM_SOLO), dtype=np.float32)
    for i in range(B):
        obs[i] = build_obs_single(
            body_pos_batch[i], body_rot_batch[i],
            body_vel_batch[i], body_ang_vel_batch[i],
            ref_body_pos_batch[i], ref_body_rot_batch[i],
            phases[i],
        )
    return obs
