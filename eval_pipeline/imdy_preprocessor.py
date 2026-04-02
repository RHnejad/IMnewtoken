"""Preprocessing utilities to feed InterHuman clips into ImDy.

This module converts per-person InterHuman positions in Z-up world coordinates
(shape: T x 22 x 3) into the marker-window format expected by ImDy's mkr model:

- mkr:  (N, M, L, 3)
- mvel: (N, M, L, 3)

where L = past_kf + fut_kf + 2 and N = T - L.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

PELVIS_IDX = 0
LEFT_HIP_IDX = 1
RIGHT_HIP_IDX = 2
LEFT_SHOULDER_IDX = 16
RIGHT_SHOULDER_IDX = 17


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions)
    if pos.ndim != 3:
        raise ValueError(f"Expected positions with 3 dims (T, J, 3), got {pos.shape}")
    if pos.shape[-1] != 3:
        raise ValueError(f"Expected last dim = 3 for xyz coordinates, got {pos.shape}")
    if pos.shape[1] < 3:
        raise ValueError(
            "Need at least pelvis and hip joints to estimate heading; got only "
            f"{pos.shape[1]} joints"
        )
    return pos.astype(np.float32, copy=False)


def estimate_heading_from_positions(positions: np.ndarray) -> np.ndarray:
    """Estimate per-frame heading yaw (radians) from InterHuman joint positions.

    Args:
        positions: (T, 22, 3) Z-up coordinates.

    Returns:
        heading: (T,) yaw angle around +Z.
    """
    pos = _validate_positions(positions)

    hip_across = pos[:, RIGHT_HIP_IDX, :2] - pos[:, LEFT_HIP_IDX, :2]
    shoulder_across = None
    if pos.shape[1] > max(LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX):
        shoulder_across = (
            pos[:, RIGHT_SHOULDER_IDX, :2] - pos[:, LEFT_SHOULDER_IDX, :2]
        )

    across = hip_across if shoulder_across is None else 0.5 * (hip_across + shoulder_across)
    across_norm = np.linalg.norm(across, axis=-1, keepdims=True)
    across = across / np.clip(across_norm, 1e-8, None)

    # Forward is perpendicular to left->right across vector on XY plane.
    # forward = z_axis x across
    forward_x = -across[:, 1]
    forward_y = across[:, 0]

    heading = np.arctan2(forward_y, forward_x)
    return np.unwrap(heading).astype(np.float32)


def estimate_linear_velocity_np(data_seq: np.ndarray, dt: float) -> np.ndarray:
    """Numpy equivalent of ImDy's central-difference velocity estimator.

    Args:
        data_seq: (B, T, ..., 3)
        dt: timestep in seconds

    Returns:
        vel_seq: same shape as data_seq
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if data_seq.shape[1] < 2:
        return np.zeros_like(data_seq)

    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, :-2]) / (2.0 * dt)
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt
    return np.concatenate([init_vel, middle_vel, final_vel], axis=1)


def _apply_heading_inverse_xy(windows: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate each marker window by inverse heading around Z."""
    out = windows.copy()
    c = np.cos(-yaw)[:, None, None]
    s = np.sin(-yaw)[:, None, None]

    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def preprocess_for_imdy(
    positions: np.ndarray,
    past_kf: int = 2,
    fut_kf: int = 2,
    treadmill: bool = True,
    remove_heading: bool = False,
    dt: float = 1.0 / 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert InterHuman positions to ImDy marker windows.

    Args:
        positions: (T, 22, 3) Z-up positions for one person.
        past_kf: number of past keyframes (default 2).
        fut_kf: number of future keyframes (default 2).
        treadmill: match ImDy config (`treadmill=true`) by centering XY at root.
        remove_heading: optional heading-invariant rotation around Z.
        dt: seconds between frames (default 1/30).

    Returns:
        mkr_windows: (N, 22, L, 3)
        mvel_windows: (N, 22, L, 3)
        frame_indices: (N,) center-frame indices in original clip
    """
    if past_kf < 0 or fut_kf < 0:
        raise ValueError(f"past_kf and fut_kf must be >=0, got {past_kf}, {fut_kf}")

    pos = _validate_positions(positions)
    window_len = past_kf + fut_kf + 2
    num_windows = pos.shape[0] - window_len

    if num_windows <= 0:
        raise ValueError(
            f"Clip too short for window_len={window_len}: T={pos.shape[0]}"
        )

    starts = np.arange(num_windows, dtype=np.int64)
    frame_indices = starts + past_kf

    windows = np.stack([pos[s : s + window_len] for s in starts], axis=0)  # (N, L, 22, 3)

    if treadmill:
        root_xy = pos[frame_indices, PELVIS_IDX, :2]  # (N, 2)
        windows[..., 0] -= root_xy[:, None, None, 0]
        windows[..., 1] -= root_xy[:, None, None, 1]

    if remove_heading:
        heading = estimate_heading_from_positions(pos)[frame_indices]
        windows = _apply_heading_inverse_xy(windows, heading)

    mvel = estimate_linear_velocity_np(windows, dt=dt)

    # ImDy expects (B, M, L, 3)
    mkr_windows = np.transpose(windows, (0, 2, 1, 3)).astype(np.float32)
    mvel_windows = np.transpose(mvel, (0, 2, 1, 3)).astype(np.float32)

    return mkr_windows, mvel_windows, frame_indices
