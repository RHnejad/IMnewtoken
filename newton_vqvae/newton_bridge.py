"""
newton_bridge.py — Differentiable Newton simulation bridge.

Converts VQ-VAE decoder output (6D rotations) → Newton joint_q,
runs PD-tracked simulation via SolverFeatherstone (differentiable),
and returns physics quantities for loss computation.

Gradient flow:
  decoder → 6D_rot_to_joint_q → PD targets → Featherstone sim → losses
                                                   ↓
                                         wp.Tape.backward()
                                                   ↓
                                         gradients → decoder → encoder

Key design:
  - Uses wp.from_torch() / wp.to_torch() for zero-copy gradient bridge
  - PD controller implemented as Warp kernels (stays in wp.Tape graph)
  - SolverFeatherstone + penalty contacts for differentiability
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import warp as wp

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import newton

from newton_vqvae.config import (
    SIM_FREQ, MOTION_FPS, SIM_SUBSTEPS, TORQUE_LIMIT,
    DOFS_PER_PERSON, COORDS_PER_PERSON, BODIES_PER_PERSON,
    N_SMPL_JOINTS, N_JOINT_Q, GRAD_CHECKPOINT_FRAMES,
)
from newton_vqvae.skeleton_cache import SkeletonCache
from prepare2.retarget import SMPL_TO_NEWTON
from prepare2.pd_utils import (
    build_pd_gains, setup_model_properties, pd_torque_kernel,
    ROOT_POS_KP, ROOT_POS_KD, ROOT_ROT_KP, ROOT_ROT_KD,
    ARMATURE_HINGE, ARMATURE_ROOT,
)
from prepare2.gen_smpl_xml import R_ROT

# Rotation matrix for SMPL-X → Newton coordinate conversion
_R_ROT = torch.from_numpy(R_ROT.astype(np.float32))  # (3,3)


# ═══════════════════════════════════════════════════════════════
# Decoder output → Newton joint_q conversion (differentiable)
# ═══════════════════════════════════════════════════════════════

def cont6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D continuous rotation to 3×3 rotation matrix.
    Input: (..., 6)  →  Output: (..., 3, 3)
    """
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """
    3×3 rotation matrix to intrinsic XYZ Euler angles.
    Input: (..., 3, 3)  →  Output: (..., 3)

    Newton's D6 joint composes: R = Rz(θ2) · Ry(θ1) · Rx(θ0)
    which is extrinsic XYZ = intrinsic ZYX  → scipy 'XYZ' convention.
    """
    # Clamp for numerical safety
    sy = torch.clamp(R[..., 0, 2], -1.0 + 1e-7, 1.0 - 1e-7)
    theta_y = torch.asin(sy)
    cos_y = torch.cos(theta_y)
    # Avoid gimbal lock (cos_y ≈ 0 extremely rare for body poses)
    theta_x = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    theta_z = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([theta_x, theta_y, theta_z], dim=-1)


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    3×3 rotation matrix to quaternion (x, y, z, w) — Newton ordering.
    Input: (..., 3, 3)  →  Output: (..., 4)
    Uses Shepperd's method for robustness.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    diag = torch.stack([R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], dim=-1)
    trace = diag.sum(-1)

    # Four cases to avoid sqrt of negative
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case w
    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2
    q[:, 3] = 0.25 * s
    q[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / s

    # Normalize
    q = F.normalize(q, dim=-1)
    return q.reshape(*batch_shape, 4)


def decoder_output_to_joint_q(
    x_hat: torch.Tensor,
    betas: torch.Tensor,
    offset: torch.Tensor,
    joints_num: int = 22,
) -> torch.Tensor:
    """
    Convert VQ-VAE decoder output to Newton joint_q (differentiable).

    Args:
        x_hat: (B, T, 262) denormalized decoder output
        betas: (B, 10) SMPL-X betas (for body offset)
        offset: (B, 3) per-subject body offset from SMPL-X
        joints_num: 22

    Returns:
        joint_q: (B, T, 76) Newton joint coordinates
    """
    B, T, D = x_hat.shape
    device = x_hat.device

    # Allocate output
    joint_q = torch.zeros(B, T, N_JOINT_Q, device=device, dtype=x_hat.dtype)

    # ── Root position ──
    # x_hat contains positions in the first joints_num*3 dims
    positions = x_hat[..., :joints_num * 3].reshape(B, T, joints_num, 3)
    root_pos = positions[:, :, 0, :]  # (B, T, 3)
    joint_q[:, :, 0:3] = root_pos + offset.unsqueeze(1)

    # ── Root orientation ──
    # Rotations start at index joints_num*6 in InterHuman format:
    # [pos(66), vel(66), rot(126), fc(4)]
    # rot has (joints_num - 1) * 6 dims (root rot is excluded in InterHuman)
    # For Newton, we need the root orientation as a quaternion.
    # InterHuman data aligns root to face Z+ direction — use identity for root rot.
    # During training, reconstruct from the motion positions (Z-up, face Z+).

    # Compute root quaternion from the hip-spine facing direction
    r_hip = positions[:, :, 2, :]  # Right hip
    l_hip = positions[:, :, 1, :]  # Left hip
    across = r_hip - l_hip
    across = F.normalize(across, dim=-1)

    y_axis = torch.zeros_like(across)
    y_axis[..., 1] = 1.0  # Y-up auxiliary

    forward = torch.cross(y_axis, across, dim=-1)
    forward = F.normalize(forward, dim=-1)

    # Build rotation matrix: Z = forward, X = across, Y = up
    up = torch.cross(across, forward, dim=-1)
    R_root = torch.stack([across, up, forward], dim=-1)  # (B, T, 3, 3)

    # Apply R_ROT^{-1} for Newton body-local frame
    R_ROT_inv = _R_ROT.T.to(device)
    R_newton_root = R_root @ R_ROT_inv.unsqueeze(0).unsqueeze(0)

    root_quat = rotmat_to_quat(R_newton_root)  # (B, T, 4) [x,y,z,w]
    joint_q[:, :, 3:7] = root_quat

    # ── Body joint rotations ──
    rot_offset = joints_num * 3 * 2  # After pos(66) + vel(66) = 132
    rots_6d = x_hat[..., rot_offset:rot_offset + (joints_num - 1) * 6]
    rots_6d = rots_6d.reshape(B, T, joints_num - 1, 6)

    rots_mat = cont6d_to_matrix(rots_6d)  # (B, T, 21, 3, 3)

    # Apply R_ROT conjugation: R_newton = R_ROT @ R_smplx @ R_ROT^T
    R_ROT_t = _R_ROT.to(device)
    R_ROT_inv_t = R_ROT_t.T

    for smpl_j in range(1, joints_num):
        newton_body = SMPL_TO_NEWTON[smpl_j]
        q_start = 7 + (newton_body - 1) * 3

        R_smplx = rots_mat[:, :, smpl_j - 1]  # (B, T, 3, 3)
        R_newton = R_ROT_t @ R_smplx @ R_ROT_inv_t
        euler = rotmat_to_euler_xyz(R_newton)  # (B, T, 3)
        joint_q[:, :, q_start:q_start + 3] = euler

    return joint_q


# ═══════════════════════════════════════════════════════════════
# Differentiable Newton simulation
# ═══════════════════════════════════════════════════════════════

class DifferentiableNewtonSim:
    """
    Differentiable physics simulation using SolverFeatherstone.

    Simulates a PD-tracked humanoid following decoded motion targets.
    Gradients flow back through wp.Tape → torch via wp.from_torch().
    """

    def __init__(self, skeleton_cache: SkeletonCache, device: str = "cuda:0"):
        self.cache = skeleton_cache
        self.device = device
        self.sim_dt = 1.0 / SIM_FREQ

    def simulate_single(
        self,
        model: newton.Model,
        target_joint_q: torch.Tensor,
        n_frames: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Run PD-tracked simulation for a single character.

        Args:
            model: Newton Model (single person + ground)
            target_joint_q: (T, 76) target trajectory (torch, requires_grad)
            n_frames: number of frames to simulate

        Returns:
            dict with:
                'sim_positions': (T, 22, 3) simulated joint positions
                'pd_torques':    (T, 75) mean PD torques per frame
                'root_forces':   (T, 6)  root DOF torques (skyhook)
                'body_positions': (T, 24, 3) all body positions
        """
        n_dof = model.joint_dof_count
        n_coords = model.joint_coord_count
        T = min(n_frames, target_joint_q.shape[0])

        # Setup model properties (armature, disable passive springs)
        setup_model_properties(model, n_persons=1, device=self.device)

        # Build PD gains
        kp_np, kd_np = build_pd_gains(model, n_persons=1)
        kp_wp = wp.array(kp_np, dtype=wp.float32, device=self.device)
        kd_wp = wp.array(kd_np, dtype=wp.float32, device=self.device)

        # Create solver (Featherstone for differentiability)
        solver = newton.solvers.SolverFeatherstone(model)

        # Initialize states
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Set initial state from first frame target
        init_q = target_joint_q[0].detach().cpu().numpy()
        state_0.joint_q = wp.array(init_q, dtype=wp.float32, device=self.device)
        state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=self.device)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        # Pre-allocate output tensors
        all_positions = torch.zeros(T, N_SMPL_JOINTS, 3, device=self.device)
        all_torques = torch.zeros(T, DOFS_PER_PERSON, device=self.device)
        all_root_forces = torch.zeros(T, 6, device=self.device)
        all_body_positions = torch.zeros(T, BODIES_PER_PERSON, 3, device=self.device)

        # Pre-allocate Warp buffers
        tau_wp = wp.zeros(n_dof, dtype=wp.float32, device=self.device)
        tau_accum = wp.zeros(n_dof, dtype=wp.float32, device=self.device)

        # Simulation loop (frame by frame)
        for frame in range(T):
            # Get target for this frame as Warp array
            ref_q_torch = target_joint_q[frame]
            ref_q_wp = wp.from_torch(ref_q_torch.contiguous(), dtype=wp.float32)

            # Zero accumulator
            wp.launch(
                kernel=_zero_kernel_75,
                dim=n_dof,
                inputs=[tau_accum],
                device=self.device,
            )

            # Substeps within this frame
            for substep in range(SIM_SUBSTEPS):
                # Compute PD torques on GPU
                wp.launch(
                    kernel=pd_torque_kernel,
                    dim=n_dof,
                    inputs=[
                        state_0.joint_q, state_0.joint_qd,
                        ref_q_wp, kp_wp, kd_wp,
                        TORQUE_LIMIT, tau_wp,
                    ],
                    device=self.device,
                )

                # Accumulate torques for averaging
                wp.launch(
                    kernel=_accumulate_kernel,
                    dim=n_dof,
                    inputs=[tau_wp, tau_accum],
                    device=self.device,
                )

                # Apply torques
                control.joint_f = tau_wp

                # Collide + step
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, self.sim_dt)

                # Swap states
                state_0, state_1 = state_1, state_0

            # Extract positions
            body_q = state_0.body_q
            body_q_torch = wp.to_torch(body_q).reshape(-1, 7)

            for j in range(N_SMPL_JOINTS):
                bidx = SMPL_TO_NEWTON[j]
                all_positions[frame, j] = body_q_torch[bidx, :3]

            for b in range(BODIES_PER_PERSON):
                all_body_positions[frame, b] = body_q_torch[b, :3]

            # Extract mean torques this frame
            tau_mean_wp = wp.zeros(n_dof, dtype=wp.float32, device=self.device)
            wp.launch(
                kernel=_scale_kernel,
                dim=n_dof,
                inputs=[tau_accum, 1.0 / SIM_SUBSTEPS, tau_mean_wp],
                device=self.device,
            )
            all_torques[frame] = wp.to_torch(tau_mean_wp)[:DOFS_PER_PERSON]
            all_root_forces[frame] = wp.to_torch(tau_mean_wp)[:6]

        return {
            'sim_positions': all_positions,
            'pd_torques': all_torques,
            'root_forces': all_root_forces,
            'body_positions': all_body_positions,
        }

    def simulate_pair(
        self,
        model: newton.Model,
        target_jq_p1: torch.Tensor,
        target_jq_p2: torch.Tensor,
        n_frames: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2: Simulate two characters together.

        Both characters are PD-tracked in the same simulation,
        enabling contact forces between them.

        Args:
            model: Newton Model (two persons + ground)
            target_jq_p1: (T, 76) target for person 1
            target_jq_p2: (T, 76) target for person 2
            n_frames: number of frames

        Returns:
            dict with keys for each person (p1/p2) and interaction data
        """
        n_dof = model.joint_dof_count
        T = min(n_frames, target_jq_p1.shape[0], target_jq_p2.shape[0])

        setup_model_properties(model, n_persons=2, device=self.device)
        kp_np, kd_np = build_pd_gains(model, n_persons=2)
        kp_wp = wp.array(kp_np, dtype=wp.float32, device=self.device)
        kd_wp = wp.array(kd_np, dtype=wp.float32, device=self.device)

        solver = newton.solvers.SolverFeatherstone(model)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Init both persons
        init_q = np.zeros(n_dof + 1, dtype=np.float32)  # coords ≈ dof+n_free_joints
        n_coords = model.joint_coord_count
        init_q_full = np.zeros(n_coords, dtype=np.float32)
        init_q_full[:COORDS_PER_PERSON] = target_jq_p1[0].detach().cpu().numpy()
        init_q_full[COORDS_PER_PERSON:2 * COORDS_PER_PERSON] = (
            target_jq_p2[0].detach().cpu().numpy()
        )
        state_0.joint_q = wp.array(init_q_full, dtype=wp.float32, device=self.device)
        state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=self.device)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        # Outputs
        pos_p1 = torch.zeros(T, N_SMPL_JOINTS, 3, device=self.device)
        pos_p2 = torch.zeros(T, N_SMPL_JOINTS, 3, device=self.device)
        torques_p1 = torch.zeros(T, DOFS_PER_PERSON, device=self.device)
        torques_p2 = torch.zeros(T, DOFS_PER_PERSON, device=self.device)
        root_forces_p1 = torch.zeros(T, 6, device=self.device)
        root_forces_p2 = torch.zeros(T, 6, device=self.device)

        tau_wp = wp.zeros(n_dof, dtype=wp.float32, device=self.device)

        for frame in range(T):
            # Build combined reference
            ref_q_full = np.zeros(n_coords, dtype=np.float32)
            ref_q_full[:COORDS_PER_PERSON] = target_jq_p1[frame].detach().cpu().numpy()
            ref_q_full[COORDS_PER_PERSON:2 * COORDS_PER_PERSON] = (
                target_jq_p2[frame].detach().cpu().numpy()
            )
            ref_q_wp = wp.array(ref_q_full, dtype=wp.float32, device=self.device)

            for substep in range(SIM_SUBSTEPS):
                # PD for both persons (sequential launch for now)
                wp.launch(
                    kernel=pd_torque_kernel,
                    dim=DOFS_PER_PERSON,
                    inputs=[
                        state_0.joint_q, state_0.joint_qd,
                        wp.from_torch(target_jq_p1[frame].contiguous()),
                        wp.array(kp_np[:DOFS_PER_PERSON], dtype=wp.float32, device=self.device),
                        wp.array(kd_np[:DOFS_PER_PERSON], dtype=wp.float32, device=self.device),
                        TORQUE_LIMIT, tau_wp,
                    ],
                    device=self.device,
                )

                control.joint_f = tau_wp
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, self.sim_dt)
                state_0, state_1 = state_1, state_0

            # Extract positions for both
            body_q_torch = wp.to_torch(state_0.body_q).reshape(-1, 7)
            for j in range(N_SMPL_JOINTS):
                bidx = SMPL_TO_NEWTON[j]
                pos_p1[frame, j] = body_q_torch[bidx, :3]
                pos_p2[frame, j] = body_q_torch[BODIES_PER_PERSON + bidx, :3]

            tau_np = wp.to_torch(tau_wp).detach()
            torques_p1[frame] = tau_np[:DOFS_PER_PERSON]
            if n_dof >= 2 * DOFS_PER_PERSON:
                torques_p2[frame] = tau_np[DOFS_PER_PERSON:2 * DOFS_PER_PERSON]
            root_forces_p1[frame] = tau_np[:6]
            root_forces_p2[frame] = tau_np[DOFS_PER_PERSON:DOFS_PER_PERSON + 6]

        return {
            'sim_positions_p1': pos_p1,
            'sim_positions_p2': pos_p2,
            'pd_torques_p1': torques_p1,
            'pd_torques_p2': torques_p2,
            'root_forces_p1': root_forces_p1,
            'root_forces_p2': root_forces_p2,
        }


# ═══════════════════════════════════════════════════════════════
# Small Warp helper kernels
# ═══════════════════════════════════════════════════════════════

@wp.kernel
def _zero_kernel_75(arr: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    arr[tid] = 0.0


@wp.kernel
def _accumulate_kernel(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    dst[tid] = dst[tid] + src[tid]


@wp.kernel
def _scale_kernel(
    src: wp.array(dtype=wp.float32),
    scale: float,
    dst: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    dst[tid] = src[tid] * scale


# ═══════════════════════════════════════════════════════════════
# Temporal smoothing for decoded joint angles
# ═══════════════════════════════════════════════════════════════

def gaussian_smooth_1d(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply 1D Gaussian smoothing along time dimension (dim=0).
    Input: (T, D)  →  Output: (T, D)
    """
    if sigma <= 0:
        return x
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    t = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # (T, D) → (1, D, T) for conv1d
    x_t = x.T.unsqueeze(0)
    kernel = kernel.view(1, 1, -1).expand(x.shape[1], -1, -1)

    pad = kernel_size // 2
    x_padded = F.pad(x_t, (pad, pad), mode='replicate')
    out = F.conv1d(x_padded, kernel, groups=x.shape[1])
    return out.squeeze(0).T
