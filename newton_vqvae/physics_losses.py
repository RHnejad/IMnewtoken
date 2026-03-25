"""
physics_losses.py — Physics-informed loss functions for VQ-VAE training.

Loss function:
    L_VQ_Newton = α L_FK_MPJPE + β L_Torque + γ L_Skyhook + δ L_SoftFlow + ε L_ZMP

All losses operate on outputs from DifferentiableNewtonSim.
"""
from __future__ import annotations

# ─── SMPL 22-joint segment mass ratios (DeLeva 1996, adapted to SMPL) ───
# Order: pelvis, L_hip, R_hip, spine1, L_knee, R_knee, spine2,
#         L_ankle, R_ankle, spine3, L_foot, R_foot, neck,
#         L_collar, R_collar, head, L_shoulder, R_shoulder,
#         L_elbow, R_elbow, L_wrist, R_wrist
SMPL_SEGMENT_MASS_RATIOS = [
    0.1117,  # 0  pelvis
    0.1416,  # 1  L_hip  (thigh)
    0.1416,  # 2  R_hip  (thigh)
    0.0551,  # 3  spine1 (abdomen)
    0.0433,  # 4  L_knee (shank)
    0.0433,  # 5  R_knee (shank)
    0.0551,  # 6  spine2 (chest lower)
    0.0137,  # 7  L_ankle
    0.0137,  # 8  R_ankle
    0.0551,  # 9  spine3 (chest upper)
    0.0129,  # 10 L_foot
    0.0129,  # 11 R_foot
    0.0286,  # 12 neck
    0.0000,  # 13 L_collar (massless connector)
    0.0000,  # 14 R_collar (massless connector)
    0.0694,  # 15 head
    0.0271,  # 16 L_shoulder (upper arm)
    0.0271,  # 17 R_shoulder (upper arm)
    0.0162,  # 18 L_elbow (forearm)
    0.0162,  # 19 R_elbow (forearm)
    0.0061,  # 20 L_wrist (hand)
    0.0061,  # 21 R_wrist (hand)
]

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# L_FK_MPJPE — Forward kinematics mean per-joint position error
# ═══════════════════════════════════════════════════════════════

def loss_fk_mpjpe(
    sim_positions: torch.Tensor,
    target_positions: torch.Tensor,
    joint_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MPJPE between simulated and target (FK) joint positions.

    Args:
        sim_positions:    (B, T, J, 3) or (T, J, 3) simulated positions
        target_positions: same shape, from decoded motion FK
        joint_weights:    (J,) optional per-joint importance weights

    Returns:
        scalar loss
    """
    diff = sim_positions - target_positions
    per_joint_err = diff.norm(dim=-1)  # (..., J)
    if joint_weights is not None:
        per_joint_err = per_joint_err * joint_weights
    return per_joint_err.mean()


# ═══════════════════════════════════════════════════════════════
# L_Torque — Joint torque regularization
# ═══════════════════════════════════════════════════════════════

def loss_torque(
    pd_torques: torch.Tensor,
    torque_limit: float = 1000.0,
    norm_type: str = "l2",
) -> torch.Tensor:
    """
    Penalize large PD tracking torques (excluding root DOFs 0:6).

    Large torques mean the decoded motion is hard to track physically.
    Clamped at torque_limit to avoid gradient explosion.

    Args:
        pd_torques: (B, T, D) or (T, D) where D = DOFS_PER_PERSON
        torque_limit: max torque for clamping
        norm_type: "l2" (MSE-style) or "l1"

    Returns:
        scalar loss
    """
    # Exclude root DOFs (positions + rotations are skyhook)
    body_torques = pd_torques[..., 6:]

    # Clamp for numerical stability
    body_torques = torch.clamp(body_torques, -torque_limit, torque_limit)

    # Normalize by torque limit
    normed = body_torques / torque_limit

    if norm_type == "l2":
        return (normed ** 2).mean()
    elif norm_type == "l1":
        return normed.abs().mean()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ═══════════════════════════════════════════════════════════════
# L_Skyhook — Root support force penalty (3-tier regularization)
# ═══════════════════════════════════════════════════════════════

def loss_skyhook(
    root_forces: torch.Tensor,
    log_space: bool = True,
    outlier_clip_sigma: float = 3.0,
) -> torch.Tensor:
    """
    Penalize "skyhook" forces — external root DOF torques that
    physically correspond to an invisible hand holding the character.

    3-tier regularization to handle explosion:
    1. Temporal Gaussian smoothing (done upstream in newton_bridge)
    2. Log-space loss: log(1 + ||F||) avoids gradient explosion
    3. Per-sample outlier clipping at N×median

    Args:
        root_forces: (B, T, 6) or (T, 6) — root position + rotation forces
        log_space: use log(1 + ||F||) instead of ||F||²
        outlier_clip_sigma: clip at this many median absolute deviations

    Returns:
        scalar loss
    """
    # Separate translational (DOF 0:3) and rotational (DOF 3:6)
    f_trans = root_forces[..., :3]
    f_rot = root_forces[..., 3:6]

    # Magnitudes
    mag_trans = f_trans.norm(dim=-1)  # (...,)
    mag_rot = f_rot.norm(dim=-1)

    # Outlier clipping using median absolute deviation (MAD)
    if outlier_clip_sigma > 0 and mag_trans.numel() > 1:
        mag_trans = _mad_clip(mag_trans, outlier_clip_sigma)
        mag_rot = _mad_clip(mag_rot, outlier_clip_sigma)

    if log_space:
        loss_trans = torch.log1p(mag_trans).mean()
        loss_rot = torch.log1p(mag_rot).mean()
    else:
        loss_trans = (mag_trans ** 2).mean()
        loss_rot = (mag_rot ** 2).mean()

    # Weight translational more (it's the literal "flying" artifact)
    return loss_trans + 0.5 * loss_rot


def _mad_clip(x: torch.Tensor, n_sigma: float) -> torch.Tensor:
    """Clip outliers at n_sigma × median absolute deviation above the median."""
    median = x.median()
    mad = (x - median).abs().median()
    upper = median + n_sigma * mad * 1.4826  # 1.4826 ≈ 1/Φ^{-1}(0.75) for normal
    return torch.clamp(x, max=upper.item())


# ═══════════════════════════════════════════════════════════════
# L_SoftFlow — Smooth penalty contact for foot-ground interaction
# ═══════════════════════════════════════════════════════════════

def loss_softflow(
    body_positions: torch.Tensor,
    foot_indices: Tuple[int, ...] = (7, 8, 10, 11),
    ground_height: float = 0.0,
    penetration_weight: float = 1.0,
    sliding_weight: float = 0.5,
    contact_threshold: float = 0.02,
    sigmoid_sharpness: float = 100.0,
    up_axis: int = 2,  # Z-up (Newton convention)
) -> torch.Tensor:
    """
    SoftFlow: smooth penalty contacts for foot-ground interaction.

    Three measures to prevent contact force explosion:
    1. Sigmoid contact detection (no hard threshold → smooth gradients)
    2. Gradual penetration penalty (not abrupt --- avoids gradient spikes)
    3. Horizontal velocity penalty scaled by contact probability

    Uses sigmoid approximation instead of hard thresholding,
    ensuring gradient flow even at the contact boundary.

    Args:
        body_positions: (B, T, J, 3) or (T, J, 3) simulated body positions
        foot_indices: SMPL joint indices for feet (ankles + toes)
        ground_height: ground plane height along up_axis
        penetration_weight: weight for penetration loss
        sliding_weight: weight for sliding loss
        contact_threshold: height below which contact is detected
        sigmoid_sharpness: steepness of sigmoid contact function
        up_axis: which axis is 'up' (0=X, 1=Y, 2=Z). Newton uses Z-up.

    Returns:
        scalar loss
    """
    # Extract foot positions
    # foot_indices: L_Ankle(7), R_Ankle(8), L_Foot(10), R_Foot(11)
    feet = body_positions[..., foot_indices, :]  # (..., 4, 3)

    # Height above ground (Z-up for Newton)
    feet_h = feet[..., up_axis]  # (..., 4)

    # Horizontal axes (the two that aren't up)
    horiz_axes = [i for i in range(3) if i != up_axis]

    # ── Penetration loss ──
    # Smooth penalty: sigmoid(-h) * |h| for h < 0
    penetration_depth = ground_height - feet_h  # positive when below ground
    penetration_mask = torch.sigmoid(sigmoid_sharpness * penetration_depth)
    loss_pen = (penetration_mask * F.relu(penetration_depth)).mean()

    # ── Sliding loss ──
    # Contact probability: high when foot is near ground
    contact_prob = torch.sigmoid(
        sigmoid_sharpness * (contact_threshold - (feet_h - ground_height))
    )

    # Horizontal velocity
    if feet.shape[-3] > 1:  # need at least 2 time steps
        feet_vel = feet[..., 1:, :, :] - feet[..., :-1, :, :]
        horiz_vel = feet_vel[..., horiz_axes]  # horizontal components
        horiz_speed = horiz_vel.norm(dim=-1)  # (..., T-1, 4)

        # Use contact_prob from t-1 (contact at start of step)
        cp = contact_prob[..., :-1, :]
        loss_slide = (cp * horiz_speed).mean()
    else:
        loss_slide = torch.tensor(0.0, device=body_positions.device)

    return penetration_weight * loss_pen + sliding_weight * loss_slide


# ═══════════════════════════════════════════════════════════════
# L_ZMP — Zero Moment Point stability
# ═══════════════════════════════════════════════════════════════

def loss_zmp(
    body_positions: torch.Tensor,
    root_forces: torch.Tensor,
    foot_indices: Tuple[int, ...] = (7, 8, 10, 11),
    support_margin: float = 0.05,
    up_axis: int = 2,  # Z-up (Newton convention)
) -> torch.Tensor:
    """
    Penalize ZMP outside the support polygon.

    Simplified version: penalize CoM horizontal projection being
    outside the bounding box of the feet (axis-aligned).

    Args:
        body_positions: (..., J, 3) body positions
        root_forces: (..., 6) root forces (for detecting stance)
        foot_indices: foot joint indices
        support_margin: margin around support polygon
        up_axis: which axis is 'up' (2=Z for Newton)

    Returns:
        scalar loss
    """
    horiz_axes = [i for i in range(3) if i != up_axis]

    # CoM approximation = root position (body 0)
    com_h = body_positions[..., 0, horiz_axes]  # (..., 2)

    # Foot support horizontal
    feet_h = body_positions[..., foot_indices, :]
    feet_h = feet_h[..., horiz_axes]  # (..., 4, 2)

    # Bounding box of feet (AA)
    bb_min = feet_h.min(dim=-2).values - support_margin  # (..., 2)
    bb_max = feet_h.max(dim=-2).values + support_margin  # (..., 2)

    # Penalize CoM outside bounding box
    below = F.relu(bb_min - com_h)
    above = F.relu(com_h - bb_max)
    out_of_support = (below + above).norm(dim=-1)

    return out_of_support.mean()


# ═══════════════════════════════════════════════════════════════
# L_ContactBudget — Contact force explosion prevention
# ═══════════════════════════════════════════════════════════════

def loss_contact_force_budget(
    pd_torques: torch.Tensor,
    body_positions: torch.Tensor,
    foot_indices: Tuple[int, ...] = (7, 8, 10, 11),
    force_budget: float = 500.0,
    up_axis: int = 2,
) -> torch.Tensor:
    """
    Penalize when estimated contact forces exceed a safety budget.

    This addresses the core problem: when a foot with a narrow contact
    area (cube edge) hits hard ground, contact forces can spike to
    thousands of Newtons in a single timestep, destabilizing the sim.

    Proxy: use the vertical component of ankle/toe PD torques as an
    indicator of contact force magnitude. When these exceed the budget,
    apply a quadratic penalty.

    Args:
        pd_torques: (..., D) PD torques
        body_positions: (..., J, 3) body positions
        foot_indices: SMPL foot joint indices
        force_budget: max acceptable force (N) per foot joint
        up_axis: up axis index (2 = Z)

    Returns:
        scalar loss
    """
    # Estimate vertical contact force from foot joint torques
    # Foot DOF indices in Newton: ankles are around DOF 18-20 and 21-23
    # Use root translational torque as proxy for total contact reaction
    vertical_root_force = pd_torques[..., up_axis].abs()  # root vertical
    over_budget = F.relu(vertical_root_force - force_budget)
    return (over_budget / force_budget).pow(2).mean()


# ═══════════════════════════════════════════════════════════════
# Combined physics loss
# ═══════════════════════════════════════════════════════════════

class PhysicsLoss(nn.Module):
    """
    Combined physics loss module.

    L_VQ_Newton = α L_FK_MPJPE + β L_Torque + γ L_Skyhook + δ L_SoftFlow + ε L_ZMP
    """

    def __init__(self, config):
        super().__init__()
        self.alpha = config.physics_weights.alpha
        self.beta = config.physics_weights.beta
        self.gamma = config.physics_weights.gamma
        self.delta = config.physics_weights.delta
        self.epsilon = config.physics_weights.epsilon
        self.torque_limit = config.torque_limit_loss

        # SMPL foot indices: L_Ankle=7, R_Ankle=8, L_Foot=10, R_Foot=11
        self.foot_indices = (7, 8, 10, 11)

    def forward(
        self,
        sim_result: Dict[str, torch.Tensor],
        target_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all physics losses.

        Args:
            sim_result: output dict from DifferentiableNewtonSim
            target_positions: (B, T, J, 3) FK-derived target positions

        Returns:
            total_loss: scalar
            loss_dict: individual losses for logging
        """
        sim_pos = sim_result['sim_positions']
        torques = sim_result['pd_torques']
        root_f = sim_result['root_forces']
        body_pos = sim_result['body_positions']

        l_fk = loss_fk_mpjpe(sim_pos, target_positions)
        l_tau = loss_torque(torques, torque_limit=self.torque_limit)
        l_sky = loss_skyhook(root_f)
        l_soft = loss_softflow(body_pos, foot_indices=self.foot_indices, up_axis=2)
        l_zmp = loss_zmp(body_pos, root_f, foot_indices=self.foot_indices, up_axis=2)
        l_contact = loss_contact_force_budget(torques, body_pos,
                                               foot_indices=self.foot_indices, up_axis=2)

        total = (
            self.alpha * l_fk
            + self.beta * l_tau
            + self.gamma * l_sky
            + self.delta * l_soft
            + self.epsilon * l_zmp
            + 0.1 * l_contact  # Contact force budget (safety)
        )

        loss_dict = {
            'l_fk_mpjpe': l_fk.detach(),
            'l_torque': l_tau.detach(),
            'l_skyhook': l_sky.detach(),
            'l_softflow': l_soft.detach(),
            'l_zmp': l_zmp.detach(),
            'l_contact_budget': l_contact.detach(),
            'l_physics_total': total.detach(),
        }

        return total, loss_dict


# ═══════════════════════════════════════════════════════════════
# Physics loss scheduler (curriculum)
# ═══════════════════════════════════════════════════════════════

class PhysicsLossScheduler:
    """
    Gradually ramp physics losses during training.

    Schedule:
        - Epoch 0 to warmup_epochs: physics_weight = 0 (kinematic only)
        - Epoch warmup_epochs to warmup + ramp_epochs: linear ramp 0 → 1
        - After: physics_weight = 1.0

    Usage:
        scheduler = PhysicsLossScheduler(warmup_epochs=5, ramp_epochs=10)
        for epoch in range(max_epoch):
            w = scheduler.get_weight(epoch)
            loss = kinematic_loss + w * physics_loss
    """

    def __init__(self, warmup_epochs: int = 5, ramp_epochs: int = 10):
        self.warmup = warmup_epochs
        self.ramp = ramp_epochs

    def get_weight(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 0.0
        elif epoch < self.warmup + self.ramp:
            return (epoch - self.warmup) / self.ramp
        else:
            return 1.0
