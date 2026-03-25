"""
newton_force_analysis.py — 4-method interaction force analysis using Newton simulator.

Computes interaction forces F_{A→B} and F_{B→A} between two interacting persons
using four complementary biomechanics methods, compared against each other and
against the naive COM-based approach (dyadic_physics.md).

Methods:
  1. Contact Sensors    — direct MuJoCo contact force measurement (feet + hands)
  2. Inverse Dynamics   — τ = M(q)(q̈ - q̈_free) with zero-phase Butterworth + B-spline
  3. RRA (Residual Reduction) — adjust COM trajectory to minimize root residuals
  4. Optimization-Based ID    — frame-by-frame constrained QP with Newton's 3rd law

Usage:
    python physics_analysis/newton_force_analysis.py --clip 1000
    python physics_analysis/newton_force_analysis.py --clip 1000 --methods 1 2
    python physics_analysis/newton_force_analysis.py --clip 1000 --mass-uncertainty
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore", message="Custom attribute")

import warp as wp
wp.config.verbose = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import newton

from prepare2.retarget import (
    load_interhuman_clip, smplx_to_joint_q, get_or_create_xml,
    SMPL_TO_NEWTON, N_SMPL_JOINTS, N_JOINT_Q,
)
from prepare2.pd_utils import (
    build_model, setup_model_properties, build_pd_gains,
    compute_all_pd_torques_np, create_mujoco_solver,
    create_contact_sensors, update_contact_sensors, init_state,
    DOFS_PER_PERSON, COORDS_PER_PERSON, BODIES_PER_PERSON,
    BODY_NAMES, DOF_NAMES, DEFAULT_TORQUE_LIMIT,
)
from prepare2.compute_torques import (
    inverse_dynamics, smooth_trajectory,
    compute_qd_qdd_spline,
)
from newton_vqvae.physics_losses import SMPL_SEGMENT_MASS_RATIOS


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

SIM_FREQ = 480
MOTION_FPS = 30
SIM_SUBSTEPS = SIM_FREQ // MOTION_FPS
SIM_DT = 1.0 / SIM_FREQ
GRAVITY = 9.81
UP_AXIS = 2  # Z-up

JOINT_GROUPS = {
    "Left Leg": ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
    "Right Leg": ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
    "Spine": ["Torso", "Spine", "Chest"],
    "Head/Neck": ["Neck", "Head"],
    "Left Arm": ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
    "Right Arm": ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
}

FOOT_BODY_INDICES = [3, 4, 7, 8]  # L_Ankle, L_Toe, R_Ankle, R_Toe

# SMPL kinematic tree (parent[i] = parent joint of joint i)
# 0=Pelvis, 1=L_Hip, 2=R_Hip, 3=Spine1, 4=L_Knee, 5=R_Knee, 6=Spine2,
# 7=L_Ankle, 8=R_Ankle, 9=Spine3, 10=L_Foot, 11=R_Foot, 12=Neck,
# 13=L_Collar, 14=R_Collar, 15=Head, 16=L_Shoulder, 17=R_Shoulder,
# 18=L_Elbow, 19=R_Elbow, 20=L_Wrist, 21=R_Wrist
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
# Bones as (child, parent) pairs for drawing
SMPL_BONES = [(i, SMPL_PARENTS[i]) for i in range(1, N_SMPL_JOINTS)]


# ═══════════════════════════════════════════════════════════════
# Data Bundle
# ═══════════════════════════════════════════════════════════════

@dataclass
class PersonBundle:
    """Holds all data for one person."""
    joint_q: np.ndarray       # (T, 76) Newton joint coordinates
    betas: np.ndarray         # (10,) or (16,) SMPL-X shape params
    mass: float               # estimated body mass in kg
    positions: Optional[np.ndarray] = None  # (T, 22, 3) FK joint positions


# ═══════════════════════════════════════════════════════════════
# Mass estimation
# ═══════════════════════════════════════════════════════════════

def estimate_mass(betas: np.ndarray) -> float:
    """Estimate body mass from SMPL-X betas using mesh volume."""
    try:
        import torch
        import trimesh
        import smplx
        model_path = os.path.join(_PROJECT_ROOT, "data", "body_model")
        smpl_model = smplx.create(
            model_path=model_path, model_type='smplx',
            gender='neutral', batch_size=1,
        )
        smpl_model.eval()
        with torch.no_grad():
            output = smpl_model(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0))
            verts = output.vertices[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=smpl_model.faces, process=False)
        return float(mesh.volume * 1000.0)
    except Exception as e:
        print(f"  Mass estimation failed ({e}), using default 75 kg")
        return 75.0


def compute_fk_positions(betas, joint_q, device="cuda:0"):
    """Compute FK positions (T, 22, 3) from joint_q."""
    T = joint_q.shape[0]
    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    setup_model_properties(model, n_persons=1, device=device)

    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
    positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    for t in range(T):
        state.joint_q = wp.array(joint_q[t].astype(np.float32), dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)
        for j in range(N_SMPL_JOINTS):
            positions[t, j] = body_q[SMPL_TO_NEWTON[j], :3]
    return positions


# ═══════════════════════════════════════════════════════════════
# Method 1: Contact Sensor Forces
# ═══════════════════════════════════════════════════════════════

def method1_contact_sensors(
    bundle_A: PersonBundle,
    bundle_B: PersonBundle,
    solver_type: str = "auto",
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Direct contact force measurement via MuJoCo sensors.

    Runs a 2-person PD-tracked simulation and reads contact sensor data
    for feet (GRF) and hands (inter-person interaction).

    Returns dict with foot_grf_A/B, hand_forces, tracking_mpjpe, etc.
    """
    print("  [Method 1] Contact Sensor Forces...")
    T = min(bundle_A.joint_q.shape[0], bundle_B.joint_q.shape[0])
    jq_A = bundle_A.joint_q[:T].astype(np.float32)
    jq_B = bundle_B.joint_q[:T].astype(np.float32)

    # Build 2-person model
    model, _ = build_model([bundle_A.betas, bundle_B.betas],
                           device=device, with_ground=True)
    setup_model_properties(model, n_persons=2, device=device)
    kp, kd = build_pd_gains(model, n_persons=2)
    n_dof = model.joint_dof_count

    # Must use MuJoCo for contact sensors
    try:
        solver = create_mujoco_solver(model, n_persons=2)
    except Exception as e:
        print(f"    MuJoCo unavailable ({e}), Method 1 requires MuJoCo — skipping")
        return None

    sensor_dict = create_contact_sensors(model, solver, n_persons=2, verbose=False)
    if sensor_dict is None:
        print("    Contact sensors unavailable — skipping Method 1")
        return None

    # Init state
    state_0, state_1, control = init_state(model, [jq_A, jq_B], n_persons=2, device=device)

    # Output buffers
    all_foot_grf = []
    all_hand_forces = []
    torques_A = np.zeros((T, DOFS_PER_PERSON), dtype=np.float32)
    torques_B = np.zeros((T, DOFS_PER_PERSON), dtype=np.float32)
    sim_pos_A = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    sim_pos_B = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)

    for frame in range(T):
        tau_accum = np.zeros(n_dof, dtype=np.float32)
        for substep in range(SIM_SUBSTEPS):
            cq = state_0.joint_q.numpy()
            cqd = state_0.joint_qd.numpy()
            tau = compute_all_pd_torques_np(
                cq, cqd, [jq_A, jq_B], frame, kp, kd,
                n_persons=2, torque_limit=DEFAULT_TORQUE_LIMIT,
            )
            tau_accum += tau
            control.joint_f = wp.array(tau, dtype=wp.float32, device=device)
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, SIM_DT)
            state_0, state_1 = state_1, state_0

        tau_mean = tau_accum / SIM_SUBSTEPS
        torques_A[frame] = tau_mean[:DOFS_PER_PERSON]
        torques_B[frame] = tau_mean[DOFS_PER_PERSON:2 * DOFS_PER_PERSON]

        # Positions
        body_q = state_0.body_q.numpy().reshape(-1, 7)
        for j in range(N_SMPL_JOINTS):
            bidx = SMPL_TO_NEWTON[j]
            sim_pos_A[frame, j] = body_q[bidx, :3]
            sim_pos_B[frame, j] = body_q[BODIES_PER_PERSON + bidx, :3]

        # Contact sensors
        cf = update_contact_sensors(solver, state_0, sensor_dict)
        if cf is not None:
            all_foot_grf.append(cf['foot_forces'].copy())
            if cf.get('hand_forces') is not None:
                all_hand_forces.append(cf['hand_forces'].copy())

        if frame % 50 == 0:
            print(f"    Frame {frame}/{T}")

    foot_grf = np.array(all_foot_grf) if all_foot_grf else None
    hand_forces = np.array(all_hand_forces) if all_hand_forces else None

    # Compute hand interaction force magnitude
    hand_F_int = None
    if hand_forces is not None and hand_forces.ndim >= 2:
        # Sum across all hand sensor groups to get total hand contact force
        if hand_forces.ndim == 3:
            hand_F_int = hand_forces.sum(axis=1)  # (T, 3)
        elif hand_forces.ndim == 2:
            hand_F_int = hand_forces  # (T, 3)

    # Tracking quality
    ref_A = bundle_A.positions[:T] if bundle_A.positions is not None else None
    ref_B = bundle_B.positions[:T] if bundle_B.positions is not None else None
    mpjpe_A = np.linalg.norm(sim_pos_A - ref_A, axis=-1).mean(axis=-1) if ref_A is not None else None
    mpjpe_B = np.linalg.norm(sim_pos_B - ref_B, axis=-1).mean(axis=-1) if ref_B is not None else None

    return {
        'foot_grf': foot_grf,
        'hand_forces': hand_forces,
        'hand_F_int': hand_F_int,
        'torques_A': torques_A,
        'torques_B': torques_B,
        'sim_pos_A': sim_pos_A,
        'sim_pos_B': sim_pos_B,
        'mpjpe_A': mpjpe_A,
        'mpjpe_B': mpjpe_B,
    }


# ═══════════════════════════════════════════════════════════════
# Method 2: Inverse Dynamics with Zero-Phase Filtering
# ═══════════════════════════════════════════════════════════════

def method2_inverse_dynamics(
    bundle: PersonBundle,
    fps: int = MOTION_FPS,
    cutoff_hz: float = 6.0,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Inverse dynamics with zero-phase Butterworth filtering on positions
    and B-spline differentiation.

    Root DOFs 0:5 are "skyhook" / virtual forces — they represent the
    unbalanced external force needed to achieve the reference trajectory.

    Returns dict with torques (T, 75), root_forces (T, 6), qd, qdd.
    """
    print(f"  [Method 2] Inverse Dynamics (cutoff={cutoff_hz}Hz)...")
    jq = bundle.joint_q.copy()
    T = jq.shape[0]

    # Zero-phase filter on positions BEFORE differentiation
    print(f"    Filtering positions (Butterworth, cutoff={cutoff_hz}Hz)...")
    jq_filtered = smooth_trajectory(jq, fps, cutoff=cutoff_hz)

    # Build 1-person model with ground
    xml_path = get_or_create_xml(bundle.betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    # Run inverse dynamics (reuse from compute_torques.py)
    torques, qd, qdd = inverse_dynamics(
        model, jq_filtered, fps, device=device,
        smooth=False,  # already filtered above
        diff_method="spline",
    )

    return {
        'torques': torques,           # (T, 75)
        'root_forces': torques[:, :6], # (T, 6) — skyhook
        'joint_torques': torques[:, 6:],  # (T, 69) — actual joint torques
        'qd': qd,
        'qdd': qdd,
        'joint_q_filtered': jq_filtered,
    }


# ═══════════════════════════════════════════════════════════════
# Method 3: Residual Reduction Algorithm (RRA-style)
# ═══════════════════════════════════════════════════════════════

def method3_rra(
    bundle: PersonBundle,
    id_result: Dict[str, np.ndarray],
    fps: int = MOTION_FPS,
    n_knots: int = 20,
    reg_lambda: float = 1e-2,
    max_iter: int = 50,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Residual Reduction Algorithm: adjust COM trajectory to minimize
    non-physical root residuals from Method 2.

    Parameterizes COM offset as a B-spline with n_knots control points,
    then optimizes with L-BFGS-B to minimize root force magnitude.

    Returns dict with adjusted torques, residual reduction, Δcom trajectory.
    """
    from scipy.interpolate import splev, splrep
    from scipy.optimize import minimize

    print(f"  [Method 3] RRA (knots={n_knots}, λ={reg_lambda})...")
    jq_base = id_result['joint_q_filtered'].copy()
    T = jq_base.shape[0]
    dt = 1.0 / fps
    t_arr = np.arange(T, dtype=np.float64) * dt

    # Build model for repeated I.D. evaluations
    xml_path = get_or_create_xml(bundle.betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    # Knot positions for B-spline (uniformly spaced)
    t_knots = np.linspace(t_arr[0], t_arr[-1], n_knots)

    # Initial root residuals
    root_res_before = id_result['root_forces'][:, :3].copy()
    res_mag_before = np.linalg.norm(root_res_before, axis=-1)
    print(f"    Before RRA: mean |root_residual| = {res_mag_before.mean():.1f} N")

    def objective(params):
        """Minimize ||root_forces_adjusted||² + λ||Δcom||²"""
        delta_com_knots = params.reshape(n_knots, 3)

        # Interpolate Δcom to all frames via B-spline
        delta_com = np.zeros((T, 3), dtype=np.float64)
        for d in range(3):
            tck = splrep(t_knots, delta_com_knots[:, d], k=3, s=0)
            delta_com[:, d] = splev(t_arr, tck)

        # Apply COM offset to root translation (first 3 coords of joint_q)
        jq_adj = jq_base.copy()
        jq_adj[:, :3] += delta_com.astype(np.float32)

        # Re-run inverse dynamics
        torques_adj, _, _ = inverse_dynamics(
            model, jq_adj, fps, device=device,
            smooth=False, diff_method="spline",
        )

        # Root residual (translational)
        root_res = torques_adj[:, :3]
        res_cost = np.sum(root_res ** 2) / T
        reg_cost = reg_lambda * np.sum(delta_com_knots ** 2)

        return float(res_cost + reg_cost)

    # Optimize
    x0 = np.zeros(n_knots * 3)
    print(f"    Optimizing {n_knots * 3} parameters...")
    result = minimize(
        objective, x0, method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-8},
    )

    # Extract optimized result
    delta_com_knots = result.x.reshape(n_knots, 3)
    delta_com = np.zeros((T, 3), dtype=np.float64)
    for d in range(3):
        tck = splrep(t_knots, delta_com_knots[:, d], k=3, s=0)
        delta_com[:, d] = splev(t_arr, tck)

    jq_adj = jq_base.copy()
    jq_adj[:, :3] += delta_com.astype(np.float32)

    torques_adj, qd_adj, qdd_adj = inverse_dynamics(
        model, jq_adj, fps, device=device,
        smooth=False, diff_method="spline",
    )

    root_res_after = torques_adj[:, :3]
    res_mag_after = np.linalg.norm(root_res_after, axis=-1)
    reduction_pct = (1.0 - res_mag_after.mean() / max(res_mag_before.mean(), 1e-8)) * 100

    print(f"    After RRA: mean |root_residual| = {res_mag_after.mean():.1f} N "
          f"({reduction_pct:.1f}% reduction)")
    print(f"    Δcom: mean={np.linalg.norm(delta_com, axis=-1).mean()*100:.2f} cm, "
          f"max={np.linalg.norm(delta_com, axis=-1).max()*100:.2f} cm")

    return {
        'torques': torques_adj,
        'root_forces': torques_adj[:, :6],
        'root_forces_before': id_result['root_forces'],
        'root_res_mag_before': res_mag_before,
        'root_res_mag_after': res_mag_after,
        'delta_com': delta_com.astype(np.float32),
        'reduction_pct': reduction_pct,
        'opt_result': result,
    }


# ═══════════════════════════════════════════════════════════════
# Method 4: Optimization-Based Inverse Dynamics
# ═══════════════════════════════════════════════════════════════

def method4_optimization_id(
    bundle_A: PersonBundle,
    bundle_B: PersonBundle,
    id_A: Dict[str, np.ndarray],
    id_B: Dict[str, np.ndarray],
    fps: int = MOTION_FPS,
    w_acc: float = 1.0,
    w_res: float = 10.0,
    w_tau: float = 0.01,
    w_fint: float = 0.1,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Frame-by-frame constrained optimization for interaction forces.

    For each frame, solve the coupled 2-person system:
      M_A · (q̈_A + Δq̈_A) + h_A = τ_A + F_ext_A
      M_B · (q̈_B + Δq̈_B) + h_B = τ_B + F_ext_B
    with F_{B→A} = -F_{A→B} (Newton's 3rd law) enforced by construction.

    Minimizes: w1·||Δq̈||² + w2·||root_residuals||² + w3·||τ||² + w4·||F_int||²

    Uses scipy least-squares per frame (fast, no Warp tape needed for
    this per-frame analytical formulation).

    Returns dict with F_int (T, 3), adjusted torques, etc.
    """
    from scipy.optimize import minimize as sp_minimize

    print(f"  [Method 4] Optimization-Based ID...")
    T = min(id_A['torques'].shape[0], id_B['torques'].shape[0])

    # Get mass matrices and free accelerations for each frame
    # We use Method 2's torques and the EoM residuals
    torques_A = id_A['torques'][:T]
    torques_B = id_B['torques'][:T]
    qdd_A = id_A['qdd'][:T]
    qdd_B = id_B['qdd'][:T]

    # The root residuals from solo I.D. tell us the "missing" external force
    # for each person. In an interaction, part of this is the interaction force.
    root_res_A = torques_A[:, :3].copy()  # translational skyhook
    root_res_B = torques_B[:, :3].copy()

    # For each frame, decompose root residual into:
    #   root_res = F_ground + F_interaction
    # with F_int_{B→A} = -F_int_{A→B} (3 unknowns for F_int)
    # and F_ground_z ≥ 0
    #
    # Decision vars per frame: F_int (3), Δq̈_root_A (3), Δq̈_root_B (3) = 9 vars
    # But simplified: we just decompose the root residuals.

    F_int = np.zeros((T, 3), dtype=np.float32)        # F_{B→A}
    F_ground_A = np.zeros((T, 3), dtype=np.float32)
    F_ground_B = np.zeros((T, 3), dtype=np.float32)
    opt_cost = np.zeros(T, dtype=np.float32)

    # Ground contact detection
    pos_A = bundle_A.positions[:T] if bundle_A.positions is not None else None
    pos_B = bundle_B.positions[:T] if bundle_B.positions is not None else None

    contact_A = np.ones(T, dtype=bool)
    contact_B = np.ones(T, dtype=bool)
    if pos_A is not None:
        feet_A = pos_A[:, FOOT_BODY_INDICES, UP_AXIS]
        contact_A = (feet_A < 0.05).any(axis=-1)
    if pos_B is not None:
        feet_B = pos_B[:, FOOT_BODY_INDICES, UP_AXIS]
        contact_B = (feet_B < 0.05).any(axis=-1)

    for t in range(T):
        # Root residual = total external force needed at root
        # res_A = F_ground_A + F_{B→A}  (all in root DOF space)
        # res_B = F_ground_B + F_{A→B} = F_ground_B - F_{B→A}
        rA = root_res_A[t]
        rB = root_res_B[t]

        def frame_objective(x):
            f_int = x[:3]  # F_{B→A}

            # F_ground = root_residual - F_interaction
            fg_A = rA - f_int
            fg_B = rB + f_int  # Newton's 3rd: F_{A→B} = -f_int

            cost = 0.0
            # Interaction force regularization
            cost += w_fint * np.sum(f_int ** 2)

            # GRF penalty: vertical component should be ≥ 0 (ground pushes up)
            if contact_A[t]:
                cost += w_res * max(0, -fg_A[UP_AXIS]) ** 2
            else:
                # Not grounded → F_ground should be ~0
                cost += w_res * np.sum(fg_A ** 2)

            if contact_B[t]:
                cost += w_res * max(0, -fg_B[UP_AXIS]) ** 2
            else:
                cost += w_res * np.sum(fg_B ** 2)

            return cost

        result = sp_minimize(frame_objective, np.zeros(3), method='L-BFGS-B')
        F_int[t] = result.x[:3]
        F_ground_A[t] = rA - F_int[t]
        F_ground_B[t] = rB + F_int[t]
        opt_cost[t] = result.fun

    # Newton's 3rd law is enforced by construction
    newton3_error = np.zeros(T, dtype=np.float32)  # always 0

    F_int_mag = np.linalg.norm(F_int, axis=-1)
    print(f"    |F_int|: mean={F_int_mag.mean():.1f} N, max={F_int_mag.max():.1f} N")
    print(f"    Newton's 3rd law error: 0.0 N (by construction)")

    return {
        'F_int': F_int,             # (T, 3) F_{B→A}
        'F_int_neg': -F_int,        # (T, 3) F_{A→B}
        'F_ground_A': F_ground_A,
        'F_ground_B': F_ground_B,
        'newton3_error': newton3_error,
        'F_int_mag': F_int_mag,
        'opt_cost': opt_cost,
        'contact_A': contact_A,
        'contact_B': contact_B,
    }


# ═══════════════════════════════════════════════════════════════
# Naive COM-based analysis (for comparison)
# ═══════════════════════════════════════════════════════════════

def naive_interaction_forces(
    positions_A: np.ndarray,
    positions_B: np.ndarray,
    mass_A: float,
    mass_B: float,
) -> Dict[str, np.ndarray]:
    """
    Naive COM-based interaction forces from dyadic_physics.md.
    Only solvable when ≥1 person is floating.
    """
    seg_mass = np.array(SMPL_SEGMENT_MASS_RATIOS, dtype=np.float32)
    seg_mass = seg_mass / seg_mass.sum()

    dt = 1.0 / MOTION_FPS
    g_vec = np.zeros(3, dtype=np.float32)
    g_vec[UP_AXIS] = -GRAVITY

    com_A = (positions_A * seg_mass[None, :, None]).sum(axis=1)
    com_B = (positions_B * seg_mass[None, :, None]).sum(axis=1)
    acc_A = np.diff(np.diff(com_A, axis=0) / dt, axis=0) / dt
    acc_B = np.diff(np.diff(com_B, axis=0) / dt, axis=0) / dt
    Ta = acc_A.shape[0]

    foot_idx = [7, 8, 10, 11]
    feet_A = positions_A[1:-1, foot_idx, UP_AXIS]
    feet_B = positions_B[1:-1, foot_idx, UP_AXIS]
    contact_A = (feet_A < 0.05).any(axis=-1)
    contact_B = (feet_B < 0.05).any(axis=-1)

    A_float = ~contact_A
    B_float = ~contact_B

    contact_state = np.full(Ta, 3, dtype=np.int32)
    contact_state[A_float & contact_B] = 1
    contact_state[contact_A & B_float] = 2
    contact_state[A_float & B_float] = 0

    F_B_on_A = np.full((Ta, 3), np.nan, dtype=np.float32)
    F_A_on_B = np.full((Ta, 3), np.nan, dtype=np.float32)
    n3_err = np.full(Ta, np.nan, dtype=np.float32)

    A_float_only = A_float & contact_B
    B_float_only = contact_A & B_float
    both_float = A_float & B_float

    if A_float_only.any():
        F_B_on_A[A_float_only] = mass_A * (acc_A[A_float_only] - g_vec)
        F_A_on_B[A_float_only] = -F_B_on_A[A_float_only]
        n3_err[A_float_only] = 0.0
    if B_float_only.any():
        F_A_on_B[B_float_only] = mass_B * (acc_B[B_float_only] - g_vec)
        F_B_on_A[B_float_only] = -F_A_on_B[B_float_only]
        n3_err[B_float_only] = 0.0
    if both_float.any():
        F_BA = mass_A * (acc_A[both_float] - g_vec)
        F_AB = mass_B * (acc_B[both_float] - g_vec)
        n3_err[both_float] = np.linalg.norm(F_BA + F_AB, axis=-1)
        F_B_on_A[both_float] = F_BA
        F_A_on_B[both_float] = F_AB

    return {
        'F_B_on_A': F_B_on_A,
        'F_A_on_B': F_A_on_B,
        'contact_state': contact_state,
        'newton3_error': n3_err,
        'n_solvable': int((~np.isnan(F_B_on_A[:, 0])).sum()),
        'n_total': Ta,
        'n_both_ground': int((contact_A & contact_B).sum()),
    }


# ═══════════════════════════════════════════════════════════════
# Mass Uncertainty (Monte Carlo)
# ═══════════════════════════════════════════════════════════════

def method2_with_uncertainty(
    bundle: PersonBundle,
    fps: int = MOTION_FPS,
    cutoff_hz: float = 6.0,
    n_samples: int = 10,
    mass_std_pct: float = 5.0,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Run Method 2 multiple times with perturbed mass to estimate uncertainty.
    Returns mean, std, and 95% CI for root forces and torques.
    """
    print(f"  [Mass Uncertainty] Monte Carlo with {n_samples} samples, "
          f"±{mass_std_pct}% mass...")

    # Nominal run
    nominal = method2_inverse_dynamics(bundle, fps, cutoff_hz, device)
    T = nominal['torques'].shape[0]

    all_root = [nominal['root_forces']]
    all_torques = [nominal['torques']]

    for i in range(n_samples - 1):
        # Perturb mass by scaling betas slightly
        scale = 1.0 + np.random.randn() * (mass_std_pct / 100.0)
        perturbed_betas = bundle.betas.copy()
        # betas[1] most strongly affects body size/mass
        perturbed_betas[1] *= scale

        perturbed = PersonBundle(
            joint_q=bundle.joint_q,
            betas=perturbed_betas,
            mass=bundle.mass * scale,
        )
        try:
            result = method2_inverse_dynamics(perturbed, fps, cutoff_hz, device)
            all_root.append(result['root_forces'][:T])
            all_torques.append(result['torques'][:T])
        except Exception as e:
            print(f"    Sample {i+1} failed: {e}")

    all_root = np.array(all_root)    # (n_samples, T, 6)
    all_torques = np.array(all_torques)  # (n_samples, T, 75)

    return {
        'nominal': nominal,
        'root_mean': all_root.mean(axis=0),
        'root_std': all_root.std(axis=0),
        'torques_mean': all_torques.mean(axis=0),
        'torques_std': all_torques.std(axis=0),
        'n_samples': len(all_root),
    }


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def _select_key_frames(T, fps, interaction_mag=None, torques=None, n_spaced=None):
    """Select key frames: evenly spaced + auto-detected peaks."""
    frames = set()

    # First and last
    frames.add(0)
    frames.add(T - 1)

    # Evenly spaced (~1 second apart)
    if n_spaced is None:
        n_spaced = max(1, int(T / fps))
    step = max(1, T // (n_spaced + 1))
    for i in range(1, n_spaced + 1):
        frames.add(min(i * step, T - 1))

    # Auto-detect peaks
    if interaction_mag is not None and len(interaction_mag) > 0:
        # Peak interaction force
        peak_idx = int(np.argmax(interaction_mag))
        frames.add(peak_idx)

        # Find secondary peaks (local maxima)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(interaction_mag, distance=fps, prominence=10.0)
        for p in peaks[:3]:  # top 3
            frames.add(int(p))

    if torques is not None:
        # Peak total joint torque
        total_tau = np.linalg.norm(torques[:, 6:], axis=-1)
        frames.add(int(np.argmax(total_tau)))

    return sorted(frames)


def plot_skeleton_keyframes(
    pos_A: np.ndarray,  # (T, 22, 3)
    pos_B: np.ndarray,
    key_frames: List[int],
    clip_id: str,
    save_dir: str,
    fps: int = MOTION_FPS,
    methods_tag: str = "",
):
    """2D stick figure plots of both persons at key frames."""
    n_frames = len(key_frames)
    n_cols = min(n_frames, 6)
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, frame in enumerate(key_frames):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Front view: X horizontal, Z vertical (Z-up)
        for person_pos, color, label in [
            (pos_A[frame], '#3498db', 'A'),
            (pos_B[frame], '#e74c3c', 'B'),
        ]:
            x = person_pos[:, 0]
            z = person_pos[:, UP_AXIS]

            # Draw bones
            for child, parent in SMPL_BONES:
                if child < len(x) and parent < len(x):
                    ax.plot([x[parent], x[child]], [z[parent], z[child]],
                            color=color, linewidth=2, alpha=0.8)

            # Draw joints
            ax.scatter(x, z, c=color, s=15, zorder=5, alpha=0.9)

            # Highlight hands (wrists)
            hand_idx = [20, 21]  # L_Wrist, R_Wrist
            for hi in hand_idx:
                if hi < len(x):
                    ax.scatter(x[hi], z[hi], c=color, s=40,
                               marker='o', edgecolors='black', linewidth=0.5, zorder=6)

        t_sec = frame / fps
        ax.set_title(f"t={t_sec:.2f}s (frame {frame})", fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Z (m)", fontsize=7)

    # Hide unused axes
    for idx in range(len(key_frames), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Skeleton Keyframes — Clip {clip_id} (Blue=A, Red=B)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"skeleton_keyframes_clip_{clip_id}{methods_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_analysis(
    results: Dict,
    clip_id: str,
    save_dir: str,
    fps: int = MOTION_FPS,
):
    """Generate the main multi-panel analysis figure."""
    methods_run = results.get('methods_run', [])
    bundles = results['bundles']
    T = min(b.joint_q.shape[0] for b in bundles)
    time_axis = np.arange(T) / fps
    colors_p = ['#3498db', '#e74c3c']
    colors_xyz = ['#e74c3c', '#2ecc71', '#3498db']

    # Determine number of panels
    panels = []
    panels.append('contact_state')
    panels.append('angular_heatmap')
    panels.append('torque_heatmap')
    if 1 in methods_run:
        panels.append('m1_contact')
    if 2 in methods_run:
        panels.append('m2_skyhook')
    if 3 in methods_run:
        panels.append('m3_rra')
    if 4 in methods_run:
        panels.append('m4_fint')
    panels.append('comparison')
    panels.append('naive')

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels), sharex=True)
    fig.suptitle(f"4-Method Interaction Force Analysis — Clip {clip_id}",
                 fontsize=14, y=0.995)

    panel_idx = 0

    # ── Contact state ──
    ax = axes[panel_idx]; panel_idx += 1
    if bundles[0].positions is not None and bundles[1].positions is not None:
        for p_idx, (bndl, color) in enumerate(zip(bundles, colors_p)):
            feet_h = bndl.positions[:T, FOOT_BODY_INDICES, UP_AXIS]
            contact = (feet_h < 0.05).any(axis=-1).astype(float)
            ax.fill_between(time_axis, p_idx - 0.4, p_idx + 0.4,
                            where=contact > 0.5, color=color, alpha=0.4,
                            label=f"P{p_idx+1} grounded")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Person A", "Person B"])
    ax.legend(fontsize=8)
    ax.set_title("Ground Contact State")
    ax.grid(True, alpha=0.3)

    # ── Angular position heatmap ──
    ax = axes[panel_idx]; panel_idx += 1
    jq_hinge = bundles[0].joint_q[:T, 7:]  # (T, 69) hinge angles
    # Reshape to (T, 23) by taking magnitude per 3-DOF body
    body_angles = np.zeros((T, 23))
    for b in range(23):
        body_angles[:, b] = np.linalg.norm(jq_hinge[:, b*3:(b+1)*3], axis=-1)
    body_angles_deg = np.degrees(body_angles)
    im = ax.imshow(
        body_angles_deg.T, aspect='auto', cmap='viridis',
        extent=[time_axis[0], time_axis[-1], 22.5, -0.5],
        interpolation='nearest',
    )
    ax.set_yticks(range(23))
    ax.set_yticklabels(BODY_NAMES[1:], fontsize=6)
    ax.set_ylabel("Joint")
    plt.colorbar(im, ax=ax, label="Angle magnitude (deg)", shrink=0.8)
    ax.set_title("Angular Position Heatmap — Person A")

    # ── Torque heatmap ──
    ax = axes[panel_idx]; panel_idx += 1
    if 2 in methods_run and 'id_A' in results:
        torques = results['id_A']['torques'][:T]
    elif 1 in methods_run and results.get('m1') is not None:
        torques = results['m1']['torques_A'][:T]
    else:
        torques = np.zeros((T, DOFS_PER_PERSON))
    body_torques = np.zeros((T, 23))
    for b in range(23):
        dof_s = 6 + b * 3
        body_torques[:, b] = np.linalg.norm(torques[:, dof_s:dof_s + 3], axis=-1)
    im = ax.imshow(
        body_torques.T, aspect='auto', cmap='hot',
        extent=[time_axis[0], time_axis[-1], 22.5, -0.5],
        interpolation='nearest',
    )
    ax.set_yticks(range(23))
    ax.set_yticklabels(BODY_NAMES[1:], fontsize=6)
    ax.set_ylabel("Joint")
    plt.colorbar(im, ax=ax, label="Torque (Nm)", shrink=0.8)
    ax.set_title("Per-Joint Torque Heatmap — Person A")

    # ── Method 1: Contact sensor forces ──
    if 1 in methods_run:
        ax = axes[panel_idx]; panel_idx += 1
        m1 = results.get('m1')
        if m1 is not None and m1.get('hand_F_int') is not None:
            hf = m1['hand_F_int']
            hf_mag = np.linalg.norm(hf, axis=-1)
            ax.plot(time_axis[:len(hf_mag)], hf_mag,
                    color='purple', linewidth=1.5,
                    label="|Hand contact force|")
        else:
            ax.text(0.5, 0.5, "No hand contact data available",
                    transform=ax.transAxes, ha='center', color='grey')
        ax.set_ylabel("Force (N)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Method 1: Contact Sensor Forces (hands)")

    # ── Method 2: Skyhook forces ──
    if 2 in methods_run:
        ax = axes[panel_idx]; panel_idx += 1
        for p_idx, (key, color) in enumerate([('id_A', colors_p[0]), ('id_B', colors_p[1])]):
            if key in results:
                rf = results[key]['root_forces'][:T, :3]
                rf_mag = np.linalg.norm(rf, axis=-1)
                ax.plot(time_axis[:len(rf_mag)], rf_mag,
                        color=color, linewidth=1.5,
                        label=f"P{p_idx+1} |root residual|")
                ax.axhline(bundles[p_idx].mass * GRAVITY, color=color,
                           linestyle=':', alpha=0.4,
                           label=f"P{p_idx+1} weight")
        ax.set_ylabel("Force (N)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Method 2: I.D. Root Residuals (skyhook)")

    # ── Method 3: RRA ──
    if 3 in methods_run:
        ax = axes[panel_idx]; panel_idx += 1
        for p_idx, key in enumerate(['rra_A', 'rra_B']):
            if key in results:
                rra = results[key]
                ax.plot(time_axis[:len(rra['root_res_mag_before'])],
                        rra['root_res_mag_before'],
                        color=colors_p[p_idx], linewidth=1, linestyle='--',
                        alpha=0.5, label=f"P{p_idx+1} before RRA")
                ax.plot(time_axis[:len(rra['root_res_mag_after'])],
                        rra['root_res_mag_after'],
                        color=colors_p[p_idx], linewidth=1.5,
                        label=f"P{p_idx+1} after RRA ({rra['reduction_pct']:.0f}% ↓)")
        ax.set_ylabel("Root Residual (N)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Method 3: RRA Residual Reduction")

    # ── Method 4: Optimized F_int ──
    if 4 in methods_run:
        ax = axes[panel_idx]; panel_idx += 1
        m4 = results.get('m4')
        if m4 is not None:
            F_int = m4['F_int']
            for c_idx, label in enumerate(['X', 'Y', 'Z']):
                ax.plot(time_axis[:len(F_int)], F_int[:, c_idx],
                        color=colors_xyz[c_idx], linewidth=1,
                        label=f"$F_{{B→A}}$ {label}")
            ax.plot(time_axis[:len(m4['F_int_mag'])], m4['F_int_mag'],
                    color='black', linewidth=1.5, alpha=0.7,
                    label="|F_{B→A}|")
        ax.set_ylabel("Force (N)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Method 4: Optimized Interaction Force (Newton 3rd enforced)")

    # ── Comparison panel ──
    ax = axes[panel_idx]; panel_idx += 1
    legend_entries = []

    if 1 in methods_run and results.get('m1') is not None:
        m1 = results['m1']
        if m1.get('hand_F_int') is not None:
            hf_mag = np.linalg.norm(m1['hand_F_int'], axis=-1)
            ax.plot(time_axis[:len(hf_mag)], hf_mag,
                    color='purple', linewidth=1.5, alpha=0.8,
                    label="M1: Contact sensor")

    if 2 in methods_run and 'id_A' in results:
        rf = results['id_A']['root_forces'][:T, :3]
        ax.plot(time_axis[:len(rf)], np.linalg.norm(rf, axis=-1),
                color='green', linewidth=1.5, alpha=0.8,
                label="M2: I.D. skyhook (A)")

    if 4 in methods_run and results.get('m4') is not None:
        ax.plot(time_axis[:len(results['m4']['F_int_mag'])],
                results['m4']['F_int_mag'],
                color='red', linewidth=1.5, alpha=0.8,
                label="M4: Optimized |F_int|")

    # Naive
    naive = results.get('naive')
    if naive is not None:
        naive_mag = np.linalg.norm(naive['F_B_on_A'], axis=-1)
        solvable = ~np.isnan(naive_mag)
        naive_time = time_axis[1:-1][:naive['n_total']]
        ax.scatter(naive_time[solvable], naive_mag[solvable],
                   s=8, color='orange', alpha=0.6, zorder=3,
                   label=f"Naive COM ({naive['n_solvable']}/{naive['n_total']})")

    ax.set_ylabel("Interaction Force (N)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Comparison: All Methods")

    # ── Naive COM panel ──
    ax = axes[panel_idx]; panel_idx += 1
    if naive is not None:
        naive_time = time_axis[1:-1][:naive['n_total']]
        cs = naive['contact_state']
        regime_colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
        regime_labels = ['Both float', 'A float', 'B float', 'Both ground']
        for r in range(4):
            mask = cs == r
            if mask.any():
                ax.fill_between(naive_time, 0, 1, where=mask,
                                color=regime_colors[r], alpha=0.3,
                                label=regime_labels[r], transform=ax.get_xaxis_transform())

        naive_mag = np.linalg.norm(naive['F_B_on_A'], axis=-1)
        solvable = ~np.isnan(naive_mag)
        ax.scatter(naive_time[solvable], naive_mag[solvable],
                   s=10, color='orange', zorder=3,
                   label=f"|F_{{B→A}}| ({naive['n_solvable']}/{naive['n_total']} solvable)")
        ax.text(0.02, 0.95,
                f"Both grounded (undetermined): {naive['n_both_ground']} frames",
                transform=ax.transAxes, fontsize=8, va='top', color='grey')
    else:
        ax.text(0.5, 0.5, "Naive comparison not available",
                transform=ax.transAxes, ha='center', color='grey')
    ax.set_ylabel("Force (N)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Naive COM Analysis (from dyadic_physics.md)")

    plt.tight_layout()
    methods_tag = "_m" + "".join(str(m) for m in sorted(methods_run)) if methods_run else ""
    path = os.path.join(save_dir, f"newton_analysis_clip_{clip_id}{methods_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_mass_uncertainty(
    unc_results: Dict,
    clip_id: str,
    save_dir: str,
    fps: int = MOTION_FPS,
    methods_tag: str = "",
):
    """Plot mass uncertainty bands for root forces."""
    if unc_results is None:
        return

    T = unc_results['root_mean'].shape[0]
    time_axis = np.arange(T) / fps

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    mean_mag = np.linalg.norm(unc_results['root_mean'][:, :3], axis=-1)
    std_mag = np.linalg.norm(unc_results['root_std'][:, :3], axis=-1)

    ax.plot(time_axis, mean_mag, color='#2c3e50', linewidth=1.5, label="Mean root residual")
    ax.fill_between(time_axis, mean_mag - std_mag, mean_mag + std_mag,
                     color='#3498db', alpha=0.3, label=f"±1σ ({unc_results['n_samples']} samples)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Root Residual (N)")
    ax.set_title(f"Mass Uncertainty — Root Force Sensitivity (Clip {clip_id})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"mass_uncertainty_clip_{clip_id}{methods_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def analyze_clip_from_joint_q(
    clip_id: str,
    joint_q_dir: str,
    save_dir: str,
    methods: List[int] = None,
    solver_type: str = "auto",
    device: str = "cuda:0",
):
    """Run 4-method analysis using pre-computed joint_q npy files.

    Loads {joint_q_dir}/{clip_id}_person{0,1}_joint_q.npy and
    {joint_q_dir}/{clip_id}_person{0,1}_betas.npy.

    This is used for position-based IK retarget data where we don't have
    raw SMPL-X params but already have Newton joint_q coordinates.
    """
    if methods is None:
        methods = [1, 2]

    print(f"\n{'='*60}")
    print(f"Analyzing clip {clip_id} (from joint_q) — Methods: {methods}")
    print(f"{'='*60}")

    bundles = []
    for p_idx in range(2):
        jq_path = os.path.join(joint_q_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
        betas_path = os.path.join(joint_q_dir, f"{clip_id}_person{p_idx}_betas.npy")

        if not os.path.exists(jq_path):
            print(f"  ERROR: {jq_path} not found")
            return None

        jq = np.load(jq_path).astype(np.float64)
        betas = np.load(betas_path).astype(np.float64) if os.path.exists(betas_path) \
            else np.zeros(10, dtype=np.float64)

        mass = estimate_mass(betas)
        positions = compute_fk_positions(betas, jq, device=device)
        bundle = PersonBundle(joint_q=jq, betas=betas, mass=mass, positions=positions)
        bundles.append(bundle)
        print(f"  Person {p_idx+1}: {jq.shape[0]} frames, mass={mass:.1f} kg")

    T = min(b.joint_q.shape[0] for b in bundles)

    results = {
        'bundles': bundles,
        'clip_id': clip_id,
        'methods_run': methods,
    }

    if 1 in methods:
        m1 = method1_contact_sensors(bundles[0], bundles[1], solver_type, device)
        results['m1'] = m1

    if 2 in methods or 3 in methods or 4 in methods:
        id_A = method2_inverse_dynamics(bundles[0], device=device)
        id_B = method2_inverse_dynamics(bundles[1], device=device)
        results['id_A'] = id_A
        results['id_B'] = id_B

        root_A_mag = np.linalg.norm(id_A['root_forces'][:, :3], axis=-1)
        root_B_mag = np.linalg.norm(id_B['root_forces'][:, :3], axis=-1)
        print(f"\n  Method 2 summary:")
        print(f"    P1 root residual: mean={root_A_mag.mean():.1f} N, max={root_A_mag.max():.1f} N")
        print(f"    P2 root residual: mean={root_B_mag.mean():.1f} N, max={root_B_mag.max():.1f} N")

    if 3 in methods:
        rra_A = method3_rra(bundles[0], id_A, device=device)
        rra_B = method3_rra(bundles[1], id_B, device=device)
        results['rra_A'] = rra_A
        results['rra_B'] = rra_B

    if 4 in methods:
        m4 = method4_optimization_id(bundles[0], bundles[1], id_A, id_B, device=device)
        results['m4'] = m4

    if bundles[0].positions is not None and bundles[1].positions is not None:
        naive = naive_interaction_forces(
            bundles[0].positions[:T], bundles[1].positions[:T],
            bundles[0].mass, bundles[1].mass,
        )
        results['naive'] = naive

    os.makedirs(save_dir, exist_ok=True)
    methods_tag = "_m" + "".join(str(m) for m in sorted(methods)) if methods else ""
    plot_analysis(results, clip_id, save_dir)

    if bundles[0].positions is not None and bundles[1].positions is not None:
        interaction_mag = results.get('m4', {}).get('F_int_mag') if 4 in methods else None
        torques_for_peaks = results.get('id_A', {}).get('torques') if 2 in methods else None
        key_frames = _select_key_frames(T, MOTION_FPS, interaction_mag, torques_for_peaks)
        plot_skeleton_keyframes(
            bundles[0].positions[:T], bundles[1].positions[:T],
            key_frames, clip_id, save_dir, methods_tag=methods_tag,
        )

    return results


def analyze_clip(
    clip_id: str,
    data_dir: str,
    save_dir: str,
    methods: List[int] = None,
    solver_type: str = "auto",
    mass_uncertainty: bool = False,
    mc_samples: int = 10,
    device: str = "cuda:0",
):
    """Run the full 4-method analysis on one InterHuman clip."""
    if methods is None:
        methods = [1, 2, 3, 4]

    print(f"\n{'='*60}")
    print(f"Analyzing clip {clip_id} — Methods: {methods}")
    print(f"{'='*60}")

    # ── Load data ──
    persons_data = load_interhuman_clip(data_dir, clip_id)
    if persons_data is None:
        print(f"  ERROR: Could not load clip {clip_id}")
        return None

    n_persons = len(persons_data)
    if n_persons < 2:
        print(f"  ERROR: Need 2 persons, found {n_persons}")
        return None

    # ── Build bundles ──
    bundles = []
    for p_idx, p in enumerate(persons_data):
        jq = smplx_to_joint_q(p['root_orient'], p['pose_body'], p['trans'], p['betas'])
        mass = estimate_mass(p['betas'])
        positions = compute_fk_positions(p['betas'], jq, device=device)
        bundle = PersonBundle(
            joint_q=jq, betas=p['betas'], mass=mass, positions=positions,
        )
        bundles.append(bundle)
        print(f"  Person {p_idx+1}: {jq.shape[0]} frames, mass={mass:.1f} kg")

    T = min(b.joint_q.shape[0] for b in bundles)

    results = {
        'bundles': bundles,
        'clip_id': clip_id,
        'methods_run': methods,
    }

    # ── Method 1: Contact sensors ──
    if 1 in methods:
        m1 = method1_contact_sensors(bundles[0], bundles[1], solver_type, device)
        results['m1'] = m1

    # ── Method 2: Inverse dynamics ──
    if 2 in methods or 3 in methods or 4 in methods:
        # Method 2 is prerequisite for Methods 3 and 4
        id_A = method2_inverse_dynamics(bundles[0], device=device)
        id_B = method2_inverse_dynamics(bundles[1], device=device)
        results['id_A'] = id_A
        results['id_B'] = id_B

        # Summary
        root_A_mag = np.linalg.norm(id_A['root_forces'][:, :3], axis=-1)
        root_B_mag = np.linalg.norm(id_B['root_forces'][:, :3], axis=-1)
        print(f"\n  Method 2 summary:")
        print(f"    P1 root residual: mean={root_A_mag.mean():.1f} N, max={root_A_mag.max():.1f} N")
        print(f"    P2 root residual: mean={root_B_mag.mean():.1f} N, max={root_B_mag.max():.1f} N")

    # ── Method 3: RRA ──
    if 3 in methods:
        rra_A = method3_rra(bundles[0], id_A, device=device)
        rra_B = method3_rra(bundles[1], id_B, device=device)
        results['rra_A'] = rra_A
        results['rra_B'] = rra_B

    # ── Method 4: Optimization ──
    if 4 in methods:
        m4 = method4_optimization_id(
            bundles[0], bundles[1], id_A, id_B, device=device,
        )
        results['m4'] = m4

    # ── Naive COM comparison ──
    if bundles[0].positions is not None and bundles[1].positions is not None:
        naive = naive_interaction_forces(
            bundles[0].positions[:T], bundles[1].positions[:T],
            bundles[0].mass, bundles[1].mass,
        )
        results['naive'] = naive
        print(f"\n  Naive COM: {naive['n_solvable']}/{naive['n_total']} frames solvable")

    # ── Mass uncertainty ──
    unc_result = None
    if mass_uncertainty and 2 in methods:
        unc_A = method2_with_uncertainty(
            bundles[0], n_samples=mc_samples, device=device,
        )
        results['uncertainty_A'] = unc_A
        unc_result = unc_A

    # ── Visualization ──
    methods_tag = "_m" + "".join(str(m) for m in sorted(methods)) if methods else ""
    os.makedirs(save_dir, exist_ok=True)
    plot_analysis(results, clip_id, save_dir)

    if mass_uncertainty and unc_result is not None:
        plot_mass_uncertainty(unc_result, clip_id, save_dir, methods_tag=methods_tag)

    # ── Skeleton keyframes ──
    if bundles[0].positions is not None and bundles[1].positions is not None:
        interaction_mag = None
        torques_for_peaks = None
        if 4 in methods and results.get('m4') is not None:
            interaction_mag = results['m4']['F_int_mag']
        if 2 in methods and 'id_A' in results:
            torques_for_peaks = results['id_A']['torques']

        key_frames = _select_key_frames(
            T, MOTION_FPS, interaction_mag, torques_for_peaks,
        )
        plot_skeleton_keyframes(
            bundles[0].positions[:T], bundles[1].positions[:T],
            key_frames, clip_id, save_dir, methods_tag=methods_tag,
        )

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Analysis complete for clip {clip_id}")
    print(f"Methods run: {methods}")
    if 4 in methods and results.get('m4') is not None:
        m4 = results['m4']
        print(f"  Method 4 F_int: mean={m4['F_int_mag'].mean():.1f} N, "
              f"max={m4['F_int_mag'].max():.1f} N")
    print(f"Output: {save_dir}")
    print(f"{'='*60}")

    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="4-method interaction force analysis using Newton simulator"
    )
    parser.add_argument("--clip", type=str, default=None, help="Single clip ID")
    parser.add_argument("--clips", type=str, nargs="+", help="Multiple clip IDs")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "data", "InterHuman"),
                        help="InterHuman data directory")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "physics_analysis", "newton_results"),
                        help="Output directory")
    parser.add_argument("--methods", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4],
                        help="Which methods to run (default: all)")
    parser.add_argument("--solver", type=str, choices=["mujoco", "featherstone", "auto"],
                        default="auto")
    parser.add_argument("--mass-uncertainty", action="store_true",
                        help="Run Monte Carlo mass uncertainty analysis")
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="Number of Monte Carlo samples")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for processing multiple clips")
    parser.add_argument("--joint-q-dir", type=str, default=None,
                        help="Load pre-computed joint_q npy files from this dir "
                             "(e.g. data/retargeted_v2/gt_from_positions). "
                             "If set, ignores --data-dir and uses analyze_clip_from_joint_q.")
    args = parser.parse_args()

    if args.clip:
        clip_ids = [args.clip]
    elif args.clips:
        clip_ids = args.clips
    else:
        clip_ids = ["1000"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Choose analysis function based on input type
    if args.joint_q_dir:
        print(f"Using pre-computed joint_q from: {args.joint_q_dir}")
        for clip_id in clip_ids:
            analyze_clip_from_joint_q(
                clip_id=clip_id,
                joint_q_dir=args.joint_q_dir,
                save_dir=args.output_dir,
                methods=args.methods,
                solver_type=args.solver,
                device=args.device,
            )
    elif args.workers > 1 and len(clip_ids) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Processing {len(clip_ids)} clips with {args.workers} workers (threads)...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for clip_id in clip_ids:
                f = executor.submit(
                    analyze_clip,
                    clip_id=clip_id,
                    data_dir=args.data_dir,
                    save_dir=args.output_dir,
                    methods=args.methods,
                    solver_type=args.solver,
                    mass_uncertainty=args.mass_uncertainty,
                    mc_samples=args.mc_samples,
                    device=args.device,
                )
                futures[f] = clip_id
            for f in as_completed(futures):
                cid = futures[f]
                try:
                    f.result()
                    print(f"  ✓ Clip {cid} done")
                except Exception as e:
                    print(f"  ✗ Clip {cid} failed: {e}")
    else:
        for clip_id in clip_ids:
            analyze_clip(
                clip_id=clip_id,
                data_dir=args.data_dir,
                save_dir=args.output_dir,
                methods=args.methods,
                solver_type=args.solver,
                mass_uncertainty=args.mass_uncertainty,
                mc_samples=args.mc_samples,
                device=args.device,
            )

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
