"""
run_full_analysis.py — Unified Newton physics analysis pipeline.

For each clip, generates:
  1. Newton MP4 video (GT vs Generated, SMPL bodies + stick figures)
  2. Per-clip torque comparison (histograms, heatmaps, bar charts, time series)
  3. Inter-person force analysis (horizontal force, axial force, CoM distance)
  4. Root PD force analysis (per-person, GT vs Gen) — PD torques are physically grounded
  5. Contact state + skeleton keyframes at key moments
  6. Raw data (.npz) for downstream analysis

Uses the CORRECT prepare4 pipeline:
  - IK from positions_zup (not broken rot6d)
  - FPS = 30 (correct InterMask rate)
  - PD forward simulation (not inverse dynamics) for physically-grounded torques
  - No skyhook / root residual artefacts — gravity + contacts handled by solver

Output structure:
    output/newton_analysis/
    ├── clip_1129_hit/
    │   ├── newton_video.mp4              (kinematic GT vs Gen)
    │   ├── newton_video_torque.mp4       (+ ID & PD torque sim, --torque-video)
    │   ├── torque_comparison.png         (6-panel)
    │   ├── forces.png                    (4-panel)
    │   ├── root_residuals.png            (root PD forces / contact state)
    │   ├── interaction_forces.png        (Newton 3rd law decomposition)
    │   ├── foot_sole_acceleration.png    (sole proxy diagnostics)
    │   ├── skeleton_keyframes.png
    │   ├── data.npz
    │   └── summary.txt
    ├── clip_1147_pull/
    │   └── ...
    ├── summary.txt
    └── eval_alignment_audit.txt

Usage:
    conda run -n mimickit --no-capture-output python prepare4/run_full_analysis.py

    # Custom clips:
    conda run -n mimickit --no-capture-output python prepare4/run_full_analysis.py \
        --clips "1129 hit" "1147 pull" "1187 kick"

    # Skip video generation (plots only):
    conda run -n mimickit --no-capture-output python prepare4/run_full_analysis.py --no-video

    # Generate torque-driven videos (ID + PD sim alongside kinematic):
    conda run -n mimickit --no-capture-output python prepare4/run_full_analysis.py --torque-video
"""
import os
import sys
import subprocess
import pickle
import argparse
import time as _time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter

# ── Constants ──
FPS = 30
DT = 1.0 / FPS
BODY_MASS = 75.0  # kg
GRAVITY = 9.81
UP_AXIS = 2  # Z-up
EVAL_MAX_GT_LENGTH = 300  # InterHumanDataset.max_gt_length in eval.py path

BODY_GROUPS = {
    "L_Leg":       slice(6, 18),
    "R_Leg":       slice(18, 30),
    "Spine/Torso": slice(30, 45),
    "L_Arm":       slice(45, 60),
    "R_Arm":       slice(60, 75),
}

GROUP_COLORS = {
    "L_Leg": "#1f77b4",
    "R_Leg": "#ff7f0e",
    "Spine/Torso": "#2ca02c",
    "L_Arm": "#d62728",
    "R_Arm": "#9467bd",
}

BODY_NAMES_JOINTS = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]

FOOT_ANKLE_INDICES = [7, 8]      # L_Ankle, R_Ankle
FOOT_FOREFOOT_INDICES = [10, 11]  # L_Foot/Toe, R_Foot/Toe

# Contact detection thresholds for sole proxy kinematics
CONTACT_Z_THRESHOLD = 0.035   # meters above sequence-local floor
CONTACT_VZ_THRESHOLD = 0.80   # m/s vertical speed for stance-like contact

DEFAULT_CLIPS = [
    ("1129", "hit"),
    ("1147", "pull"),
    ("1187", "kick"),
    ("1006", "sword"),
    ("1441", "strike"),
    ("3017", "punch"),
]


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_gt_persons(clip_id):
    """Load GT persons from InterHuman dataset (SMPL-X params)."""
    from prepare4.retarget import load_interhuman_pkl
    data_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    return load_interhuman_pkl(data_dir, str(clip_id))


def load_gen_persons(clip_id):
    """Load generated persons from InterMask output."""
    path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman",
                        f"{clip_id}.pkl")
    if not os.path.isfile(path):
        return None, None
    with open(path, "rb") as f:
        raw = pickle.load(f)

    text = raw.get("text", f"Clip {clip_id}")
    results = []
    for pkey in ["person1", "person2"]:
        if pkey not in raw:
            return None, text
        p = raw[pkey]
        d = {k: p[k].astype(np.float64) for k in
             ["root_orient", "pose_body", "trans", "betas"]}
        if "positions_zup" in p:
            d["positions_zup"] = p["positions_zup"].astype(np.float64)
        results.append(d)
    return results, text


def load_positions_zup(clip_id, source):
    """Load positions_zup for both persons from either GT or Generated."""
    if source == "generated":
        path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman",
                            f"{clip_id}.pkl")
        if not os.path.isfile(path):
            return None, None
        with open(path, "rb") as f:
            raw = pickle.load(f)
        p1 = raw["person1"].get("positions_zup")
        p2 = raw["person2"].get("positions_zup")
        return p1, p2
    else:
        from data.utils import trans_matrix as TRANS_MATRIX_TORCH
        INV_TRANS = np.linalg.inv(TRANS_MATRIX_TORCH.numpy().astype(np.float64))
        gt_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
        positions = []
        for pidx in [1, 2]:
            npy_path = os.path.join(gt_dir, "motions_processed",
                                    f"person{pidx}", f"{clip_id}.npy")
            if not os.path.isfile(npy_path):
                return None, None
            raw = np.load(npy_path).astype(np.float64)
            pos_yup = raw[:, :66].reshape(-1, 22, 3)
            pos_zup = np.einsum("mn,...n->...m", INV_TRANS, pos_yup)
            positions.append(pos_zup)
        return positions[0], positions[1]


def load_motion_text(clip_id):
    """Load the annotation text for a clip."""
    annots_path = os.path.join(PROJECT_ROOT, "data", "InterHuman",
                               "annots", f"{clip_id}.txt")
    if os.path.isfile(annots_path):
        with open(annots_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            return lines[0]
    return f"Clip {clip_id}"


def get_eval_temporal_info(clip_id):
    """Compute eval.py-equivalent temporal lengths for one InterHuman clip.

    Eval dataset path applies:
      - crop GT to at most 300 frames
      - process_motion_np uses motion[:-1], so effective GT motion_lens = crop_len - 1
      - transformer ids_length = motion_lens // 4
      - decoded generation length = ids_length * 4
    """
    p1 = os.path.join(PROJECT_ROOT, "data", "InterHuman", "motions_processed",
                      "person1", f"{clip_id}.npy")
    if not os.path.isfile(p1):
        return None

    raw_len = int(np.load(p1, mmap_mode="r").shape[0])
    eval_crop_len = min(raw_len, EVAL_MAX_GT_LENGTH)
    eval_motion_lens = max(eval_crop_len - 1, 0)
    eval_ids_len = eval_motion_lens // 4
    eval_gen_len = eval_ids_len * 4
    return {
        "raw_len": raw_len,
        "eval_crop_len": eval_crop_len,
        "eval_motion_lens": eval_motion_lens,
        "eval_ids_len": eval_ids_len,
        "eval_gen_len": eval_gen_len,
    }


# ═══════════════════════════════════════════════════════════════
# Physics Computation
# ═══════════════════════════════════════════════════════════════

def pd_forward_torques(joint_q, betas, dt=DT, device="cuda:0", verbose=False,
                       settle_frames=15):
    """Compute physically-grounded torques via PD tracking in forward simulation.

    Runs a Newton/MuJoCo forward sim with a ground plane.  At every physics
    substep (480 Hz), explicit PD torques are computed via the GPU kernel
    ``pd_torque_kernel`` and written to ``control.joint_f``.  The torques
    are accumulated across substeps and averaged per frame.

    Because the simulation includes a ground plane, gravity, and contacts,
    the resulting torques are physically valid — no "skyhook" root residuals.

    A static settle phase holds the initial pose for ``settle_frames``
    before motion begins, allowing the physics engine to establish
    ground contacts and equilibrium (prevents falling at start).

    Args:
        joint_q: (T, 76) reference joint coordinates
        betas: (10,) SMPL-X shape parameters
        dt: timestep (1/fps)
        device: CUDA device
        verbose: print progress
        settle_frames: number of frames to hold initial pose for settling
                       (default 15 ≈ 0.5s at 30fps)

    Returns:
        torques: (T, 75) average PD torques per frame
        sim_jq:  (T, 76) simulated joint coordinates
    """
    import warp as wp
    import newton
    from prepare4.dynamics import (
        set_segment_masses, _setup_model_for_id,
        PD_GAINS, ROOT_POS_KP, ROOT_POS_KD, ROOT_ROT_KP, ROOT_ROT_KD,
        DEFAULT_TORQUE_LIMIT, BODY_NAMES as DYN_BODY_NAMES,
        N_JOINT_Q, N_JOINT_QD,
    )
    from prepare4.gen_xml import get_or_create_xml
    from prepare2.pd_utils import pd_torque_kernel, accumulate_torque_kernel

    T = joint_q.shape[0]
    fps = round(1.0 / dt)
    sim_freq = 480
    sim_steps = sim_freq // fps
    dt_sim = 1.0 / sim_freq
    torque_limit = DEFAULT_TORQUE_LIMIT

    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    set_segment_masses(model, total_mass=BODY_MASS, verbose=False)
    _setup_model_for_id(model, device=device)

    n_dof = model.joint_dof_count

    kp_np = np.zeros(n_dof, dtype=np.float32)
    kd_np = np.zeros(n_dof, dtype=np.float32)
    kp_np[:3] = ROOT_POS_KP;    kd_np[:3] = ROOT_POS_KD
    kp_np[3:6] = ROOT_ROT_KP;   kd_np[3:6] = ROOT_ROT_KD
    for b_idx, name in enumerate(DYN_BODY_NAMES[1:]):
        s = 6 + b_idx * 3
        kp_val, kd_val = PD_GAINS.get(name, (100, 10))
        kp_np[s:s + 3] = kp_val
        kd_np[s:s + 3] = kd_val

    kp_gpu = wp.array(kp_np, dtype=wp.float32, device=device)
    kd_gpu = wp.array(kd_np, dtype=wp.float32, device=device)

    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=450, nconmax=150,
        impratio=10, iterations=100, ls_iterations=50,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    jq_f32 = joint_q.astype(np.float32)
    state_0.joint_q = wp.array(jq_f32[0], dtype=wp.float32, device=device)
    state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    all_ref_gpu = wp.array(jq_f32.ravel(), dtype=wp.float32, device=device)
    ref_q_gpu = wp.zeros(N_JOINT_Q, dtype=wp.float32, device=device)

    tau_substep = wp.zeros(n_dof, dtype=wp.float32, device=device)
    tau_accum = wp.zeros(n_dof, dtype=wp.float32, device=device)
    control.joint_f = tau_substep

    torques_out = np.zeros((T, n_dof), dtype=np.float32)
    sim_jq = np.zeros((T, N_JOINT_Q), dtype=np.float32)
    sim_jq[0] = jq_f32[0]

    inv_steps = 1.0 / float(sim_steps)

    if verbose:
        print(f"  PD forward sim: {T} frames, {sim_steps} substeps/frame, "
              f"{sim_freq} Hz, settle={settle_frames} frames...")

    # ── Settle phase: hold first frame to establish contacts ──
    if settle_frames > 0:
        # Target = frame 0
        wp.copy(ref_q_gpu, all_ref_gpu,
                src_offset=0, count=N_JOINT_Q)

        for sf in range(settle_frames):
            tau_accum.zero_()
            for sub in range(sim_steps):
                wp.launch(
                    pd_torque_kernel, dim=n_dof,
                    inputs=[state_0.joint_q, state_0.joint_qd,
                            ref_q_gpu, kp_gpu, kd_gpu, torque_limit,
                            tau_substep],
                    device=device,
                )
                # Don't accumulate — settle torques are discarded
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

        if verbose:
            cq = state_0.joint_q.numpy()
            ref0 = jq_f32[0]
            settle_pos_err = np.linalg.norm(cq[:3] - ref0[:3])
            settle_hinge_err = (np.abs(cq[7:] - ref0[7:]).mean()
                                * 180 / np.pi)
            print(f"    Settle done ({settle_frames} frames): "
                  f"pos_err={settle_pos_err * 100:.1f}cm "
                  f"hinge_err={settle_hinge_err:.1f}°")

    for t in range(1, T):
        wp.copy(ref_q_gpu, all_ref_gpu,
                src_offset=t * N_JOINT_Q, count=N_JOINT_Q)

        tau_accum.zero_()

        for sub in range(sim_steps):
            wp.launch(
                pd_torque_kernel, dim=n_dof,
                inputs=[state_0.joint_q, state_0.joint_qd,
                        ref_q_gpu, kp_gpu, kd_gpu, torque_limit,
                        tau_substep],
                device=device,
            )

            wp.launch(
                accumulate_torque_kernel, dim=n_dof,
                inputs=[tau_substep, tau_accum],
                device=device,
            )

            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt_sim)
            state_0, state_1 = state_1, state_0

        torques_out[t] = tau_accum.numpy() * inv_steps
        sim_jq[t] = state_0.joint_q.numpy()

        if verbose and t % 50 == 0:
            ref = jq_f32[t]
            pos_err = np.linalg.norm(sim_jq[t, :3] - ref[:3])
            hinge_err = np.abs(sim_jq[t, 7:] - ref[7:]).mean() * 180 / np.pi
            tau_h = torques_out[t, 6:]
            print(f"    Frame {t}/{T}: pos_err={pos_err * 100:.1f}cm "
                  f"hinge_err={hinge_err:.1f}° "
                  f"|τ_hinge| mean={np.abs(tau_h).mean():.1f} Nm")

    if verbose:
        print(f"  Done. |τ_hinge| mean={np.abs(torques_out[:, 6:]).mean():.1f} Nm, "
              f"|root| mean={np.linalg.norm(torques_out[:, :3], axis=-1).mean():.0f} N")

    return torques_out, sim_jq


def compute_torques_for_person(person_data, source, device="cuda:0"):
    """Compute torques for one person via PD forward simulation.

    Runs a full Newton/MuJoCo forward sim with ground contacts and PD
    tracking.  The applied PD torques are recorded — these are physically
    grounded (gravity + contacts handled by the solver), eliminating the
    "skyhook" root residuals that plague pure inverse dynamics.

    For GT: rotation_retarget (direct SMPL-X → Newton), downsample 60→30fps.
    For Generated: ik_retarget from positions_zup (the correct approach).

    Returns (torques, joint_q) tuple or (None, None).
    """
    from prepare4.retarget import rotation_retarget, ik_retarget

    if source == "generated":
        if "positions_zup" not in person_data:
            return None, None
        joint_q, _ = ik_retarget(
            person_data["positions_zup"], person_data["betas"],
            ik_iters=50, device=device, sequential=True,
        )
    else:
        joint_q = rotation_retarget(
            person_data["root_orient"], person_data["pose_body"],
            person_data["trans"], person_data["betas"],
        )
        joint_q = joint_q[::2]  # 60fps → 30fps

    if joint_q.shape[0] < 11:
        return None, None

    torques, sim_jq = pd_forward_torques(
        joint_q, person_data["betas"],
        dt=DT, device=device, verbose=False,
    )
    return torques, joint_q


def compute_com_acceleration(positions_zup):
    """Compute center-of-mass acceleration from mean joint position (≈CoM proxy).
    Returns: (T, 3) acceleration in m/s²."""
    com = positions_zup.mean(axis=1)
    T = com.shape[0]
    if T < 11:
        return np.zeros_like(com)
    accel = np.zeros_like(com)
    for d in range(3):
        win = min(11, T if T % 2 == 1 else T - 1)
        accel[:, d] = savgol_filter(com[:, d], win, 5, deriv=2, delta=DT)
    return accel


def compute_interaction_force(positions_p1, positions_p2, mass=BODY_MASS):
    """Estimate inter-person forces from CoM accelerations.

    Returns dict with time-series of force magnitudes, directions,
    and CoM distance."""
    com1 = positions_p1.mean(axis=1)
    com2 = positions_p2.mean(axis=1)
    T = min(com1.shape[0], com2.shape[0])
    com1, com2 = com1[:T], com2[:T]

    accel1 = compute_com_acceleration(positions_p1[:T])
    accel2 = compute_com_acceleration(positions_p2[:T])

    force1 = mass * accel1
    force2 = mass * accel2

    gravity = np.array([0.0, 0.0, -GRAVITY])
    ext_force1 = force1 - mass * gravity
    ext_force2 = force2 - mass * gravity

    delta = com2 - com1
    dist = np.clip(np.linalg.norm(delta, axis=1, keepdims=True), 0.01, None)
    direction = delta / dist

    return {
        "force_mag_p1": np.linalg.norm(ext_force1, axis=1),
        "force_mag_p2": np.linalg.norm(ext_force2, axis=1),
        "force_along_p1": np.sum(ext_force1 * direction, axis=1),
        "force_along_p2": np.sum(ext_force2 * (-direction), axis=1),
        "horiz_force_p1": np.linalg.norm(ext_force1[:, :2], axis=1),
        "horiz_force_p2": np.linalg.norm(ext_force2[:, :2], axis=1),
        "com_dist": dist.squeeze(),
    }


def _savgol_win(T, target=11, min_win=5):
    """Pick a safe odd SavGol window for a sequence length."""
    if T < min_win:
        return None
    win = min(target, T if T % 2 == 1 else T - 1)
    if win < min_win:
        return None
    if win % 2 == 0:
        win -= 1
    return max(win, min_win)


def _smooth_derivative(x, deriv):
    """SavGol derivative for (T, D) signals; returns zeros for short clips."""
    T, D = x.shape
    win = _savgol_win(T)
    if win is None:
        return np.zeros_like(x)
    poly = min(5, win - 1)
    out = np.zeros_like(x)
    for d in range(D):
        out[:, d] = savgol_filter(x[:, d], win, poly, deriv=deriv, delta=DT)
    return out


def compute_foot_sole_kinematics(positions):
    """Build a foot-sole proxy from ankle and forefoot keypoints.

    The 22-joint skeleton does not contain a true sole center keypoint.
    We approximate sole points as midpoints on each foot segment:
      sole_left  = 0.5 * (L_ankle + L_forefoot)
      sole_right = 0.5 * (R_ankle + R_forefoot)
    and define sole COM as the mean of both soles.
    """
    left_ankle = positions[:, FOOT_ANKLE_INDICES[0], :]
    right_ankle = positions[:, FOOT_ANKLE_INDICES[1], :]
    left_fore = positions[:, FOOT_FOREFOOT_INDICES[0], :]
    right_fore = positions[:, FOOT_FOREFOOT_INDICES[1], :]

    sole_left = 0.5 * (left_ankle + left_fore)
    sole_right = 0.5 * (right_ankle + right_fore)
    sole_com = 0.5 * (sole_left + sole_right)

    kin = {
        "sole_left": sole_left,
        "sole_right": sole_right,
        "sole_com": sole_com,
        "sole_left_vel": _smooth_derivative(sole_left, deriv=1),
        "sole_right_vel": _smooth_derivative(sole_right, deriv=1),
        "sole_com_vel": _smooth_derivative(sole_com, deriv=1),
        "sole_left_acc": _smooth_derivative(sole_left, deriv=2),
        "sole_right_acc": _smooth_derivative(sole_right, deriv=2),
        "sole_com_acc": _smooth_derivative(sole_com, deriv=2),
    }
    return kin


def detect_foot_contact(positions, z_threshold=CONTACT_Z_THRESHOLD,
                        vz_threshold=CONTACT_VZ_THRESHOLD):
    """Detect stance contact from sole proxy height + vertical speed.

    Returns (T,) boolean indicating whether either foot is in stance-like
    ground contact.
    """
    kin = compute_foot_sole_kinematics(positions)
    sole_z = np.stack([
        kin["sole_left"][:, UP_AXIS],
        kin["sole_right"][:, UP_AXIS],
    ], axis=1)
    sole_vz = np.stack([
        kin["sole_left_vel"][:, UP_AXIS],
        kin["sole_right_vel"][:, UP_AXIS],
    ], axis=1)

    floor_ref = sole_z.min(axis=0, keepdims=True)
    sole_rel = sole_z - floor_ref

    stance_like = (sole_rel < z_threshold) & (np.abs(sole_vz) < vz_threshold)
    return stance_like.any(axis=1)


def compute_contact_state(pos_p1, pos_p2):
    """Compute 4-state contact regime for each frame.
    0=both_float, 1=A_float_only, 2=B_float_only, 3=both_grounded."""
    T = min(pos_p1.shape[0], pos_p2.shape[0])
    contact_A = detect_foot_contact(pos_p1[:T])
    contact_B = detect_foot_contact(pos_p2[:T])

    state = np.full(T, 3, dtype=np.int32)
    state[(~contact_A) & contact_B] = 1
    state[contact_A & (~contact_B)] = 2
    state[(~contact_A) & (~contact_B)] = 0
    return state, contact_A, contact_B


def decompose_interaction_forces(torques_A, torques_B, pos_p1, pos_p2,
                                 mass=BODY_MASS):
    """Decompose root PD forces into ground reaction + interaction forces.

    For each frame, root force (DOFs 0:3) = F_ground + F_interaction.
    Newton's 3rd law: F_{B→A} = -F_{A→B} (3 unknowns).

    Contact state determines solvability:
      - A floating: F_ground_A ≈ 0, so F_{B→A} = root_force_A
      - B floating: F_ground_B ≈ 0, so F_{A→B} = root_force_B
      - Both floating: overdetermined (2 equations, 3 unknowns) — average
      - Both grounded: underdetermined — use optimization with GRF ≥ 0

    Args:
        torques_A, torques_B: (T_A, 75), (T_B, 75) from PD forward simulation
        pos_p1, pos_p2: (T, 22, 3) joint positions
        mass: body mass in kg

    Returns dict with F_int (T, 3), F_ground_A/B, contact_state, etc.
    """
    from scipy.optimize import minimize as sp_minimize

    T = min(torques_A.shape[0], torques_B.shape[0],
            pos_p1.shape[0], pos_p2.shape[0])

    root_res_A = torques_A[:T, :3].copy()
    root_res_B = torques_B[:T, :3].copy()

    contact_state, contact_A, contact_B = compute_contact_state(
        pos_p1[:T], pos_p2[:T])

    F_int = np.zeros((T, 3), dtype=np.float64)  # F_{B→A}
    F_ground_A = np.zeros((T, 3), dtype=np.float64)
    F_ground_B = np.zeros((T, 3), dtype=np.float64)

    w_fint = 0.1  # regularization for interaction force
    w_res = 10.0  # penalty for GRF violations

    for t in range(T):
        rA = root_res_A[t]
        rB = root_res_B[t]

        def frame_objective(x):
            f_int = x[:3]
            fg_A = rA - f_int
            fg_B = rB + f_int

            cost = w_fint * np.sum(f_int ** 2)

            if contact_A[t]:
                cost += w_res * max(0, -fg_A[UP_AXIS]) ** 2
            else:
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

    F_int_mag = np.linalg.norm(F_int, axis=-1)

    # Direction of interaction: project onto inter-person axis
    com1 = pos_p1[:T].mean(axis=1)
    com2 = pos_p2[:T].mean(axis=1)
    delta = com2 - com1
    dist = np.clip(np.linalg.norm(delta, axis=1, keepdims=True), 0.01, None)
    direction = delta / dist

    F_int_along = np.sum(F_int * direction, axis=1)

    return {
        "F_int": F_int,
        "F_int_mag": F_int_mag,
        "F_int_along": F_int_along,
        "F_ground_A": F_ground_A,
        "F_ground_B": F_ground_B,
        "contact_state": contact_state,
        "contact_A": contact_A,
        "contact_B": contact_B,
        "com_dist": dist.squeeze(),
        "T": T,
    }


def plot_interaction_forces(decomp_gt, decomp_gen, text, clip_id, out_dir):
    """Plot the decomposed interaction forces: F_{B→A} from root PD forces."""
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(f'Interaction Force (from Root Residual Decomposition) — Clip {clip_id}\n"{text}"',
                 fontsize=13, fontweight='bold')

    T_gt = decomp_gt["T"]
    T_gen = decomp_gen["T"]
    t_gt = np.arange(T_gt) / FPS
    t_gen = np.arange(T_gen) / FPS
    axis_colors = ['#e74c3c', '#2ecc71', '#3498db']
    axis_names = ['X', 'Y', 'Z']

    # Panel 1: GT interaction force by axis
    ax = axes[0, 0]
    for d in range(3):
        ax.plot(t_gt, decomp_gt["F_int"][:, d], color=axis_colors[d],
                linewidth=1.0, alpha=0.8, label=f"$F_{{B→A}}$ {axis_names[d]}")
    ax.plot(t_gt, decomp_gt["F_int_mag"], color="black", linewidth=1.5,
            alpha=0.7, label="|F_{B→A}|")
    ax.set_ylabel("Force (N)")
    ax.set_title("GT — Interaction Force F_{B→A}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

    # Panel 2: Gen interaction force by axis
    ax = axes[0, 1]
    for d in range(3):
        ax.plot(t_gen, decomp_gen["F_int"][:, d], color=axis_colors[d],
                linewidth=1.0, alpha=0.8, label=f"$F_{{B→A}}$ {axis_names[d]}")
    ax.plot(t_gen, decomp_gen["F_int_mag"], color="black", linewidth=1.5,
            alpha=0.7, label="|F_{B→A}|")
    ax.set_ylabel("Force (N)")
    ax.set_title("Gen — Interaction Force F_{B→A}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

    # Panel 3: GT vs Gen |F_int| overlaid
    ax = axes[1, 0]
    ax.plot(t_gt, decomp_gt["F_int_mag"], color="steelblue", linewidth=1.5,
            label="GT |F_{B→A}|")
    ax.plot(t_gen, decomp_gen["F_int_mag"], color="orangered", linewidth=1.5,
            linestyle="--", label="Gen |F_{B→A}|")
    ax.set_ylabel("Force (N)")
    ax.set_title("|Interaction Force| — GT vs Gen")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Force along inter-person axis
    ax = axes[1, 1]
    ax.plot(t_gt, decomp_gt["F_int_along"], color="steelblue", linewidth=1.5,
            label="GT F along axis")
    ax.plot(t_gen, decomp_gen["F_int_along"], color="orangered", linewidth=1.5,
            linestyle="--", label="Gen F along axis")
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel("Force (N)")
    ax.set_title("Interaction Force Along Inter-Person Axis (+push, -pull)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 5: Contact state + interaction force
    ax = axes[2, 0]
    regime_colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
    regime_labels = ['Both float', 'A float', 'B float', 'Both ground']
    for r in range(4):
        mask = decomp_gt["contact_state"] == r
        if mask.any():
            ax.fill_between(t_gt, 0, decomp_gt["F_int_mag"].max() * 1.1,
                            where=mask, color=regime_colors[r], alpha=0.15,
                            label=f"GT: {regime_labels[r]}")
    ax.plot(t_gt, decomp_gt["F_int_mag"], color="steelblue", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("GT |F_int| with Contact State")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 6: Summary bar chart
    ax = axes[2, 1]
    metrics = {
        '|F_int|\nmean': (decomp_gt["F_int_mag"].mean(), decomp_gen["F_int_mag"].mean()),
        '|F_int|\nP95': (np.percentile(decomp_gt["F_int_mag"], 95),
                         np.percentile(decomp_gen["F_int_mag"], 95)),
        '|F_along|\nmean': (np.abs(decomp_gt["F_int_along"]).mean(),
                            np.abs(decomp_gen["F_int_along"]).mean()),
        'CoM dist\nmean': (decomp_gt["com_dist"].mean(), decomp_gen["com_dist"].mean()),
    }
    x_bar = np.arange(len(metrics))
    w = 0.35
    gt_bar = [v[0] for v in metrics.values()]
    gen_bar = [v[1] for v in metrics.values()]
    ax.bar(x_bar - w/2, gt_bar, w, label="GT", color="steelblue", alpha=0.8)
    ax.bar(x_bar + w/2, gen_bar, w, label="Gen", color="orangered", alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax.set_title("Interaction Force Summary (Newton 3rd enforced)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (gt_v, gen_v) in enumerate(metrics.values()):
        ratio = gen_v / max(gt_v, 0.01)
        ax.text(i, max(gt_v, gen_v) * 1.05, f"×{ratio:.2f}",
                ha="center", fontsize=8, color="gray")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "interaction_forces.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_foot_sole_acceleration(gt_pos1, gt_pos2, gen_pos1, gen_pos2,
                                decomp_gt, decomp_gen, text, clip_id, out_dir):
    """Plot foot-sole COM acceleration with contact and estimated GRF_z.

    Focuses on directionality of vertical velocity/acceleration around
    contact-to-flight transitions.
    """
    if gt_pos1 is None or gt_pos2 is None or gen_pos1 is None or gen_pos2 is None:
        return

    def _person_series(pos, decomp_fg, contact_mask):
        kin = compute_foot_sole_kinematics(pos)
        T = min(pos.shape[0], decomp_fg.shape[0], contact_mask.shape[0])
        t = np.arange(T) / FPS
        contact = contact_mask[:T]
        liftoff = np.zeros(T, dtype=bool)
        if T > 1:
            liftoff[1:] = contact[:-1] & (~contact[1:])

        az = kin["sole_com_acc"][:T, UP_AXIS]
        vz = kin["sole_com_vel"][:T, UP_AXIS]
        # Frames where foot is airborne and moving upward with upward accel.
        airborne_up = (~contact) & (vz > 0.0) & (az > 0.0)
        # Ballistic-like airborne motion: upward/downward velocity with gravity-like downward accel.
        airborne_gravity_like = (~contact) & (az < -1.0)

        return {
            "t": t,
            "az": az,
            "ax": kin["sole_com_acc"][:T, 0],
            "ay": kin["sole_com_acc"][:T, 1],
            "a_mag": np.linalg.norm(kin["sole_com_acc"][:T], axis=-1),
            "z": kin["sole_com"][:T, UP_AXIS],
            "vz": vz,
            "fgz": decomp_fg[:T, UP_AXIS],
            "contact": contact,
            "liftoff": liftoff,
            "airborne_up": airborne_up,
            "airborne_gravity_like": airborne_gravity_like,
        }

    gt_p1 = _person_series(gt_pos1, decomp_gt["F_ground_A"], decomp_gt["contact_A"])
    gt_p2 = _person_series(gt_pos2, decomp_gt["F_ground_B"], decomp_gt["contact_B"])
    gen_p1 = _person_series(gen_pos1, decomp_gen["F_ground_A"], decomp_gen["contact_A"])
    gen_p2 = _person_series(gen_pos2, decomp_gen["F_ground_B"], decomp_gen["contact_B"])

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(f'Foot-Sole COM Acceleration vs Contact/GRF — Clip {clip_id}\n"{text}"',
                 fontsize=13, fontweight='bold')

    def _draw(ax, s, title, color):
        t = s["t"]
        # Shade detected contact frames
        ax.fill_between(t, -40, 40, where=s["contact"], color="#7f8c8d", alpha=0.12,
                        label="Detected contact")
        ax.plot(t, s["az"], color=color, linewidth=1.2, label="sole COM a_z (m/s²)")
        ax.plot(t, s["fgz"] / BODY_MASS, color="#2ecc71", linewidth=1.2,
                linestyle="--", label="F_ground,z / m (m/s²)")
        ax.plot(t, s["vz"], color="#9b59b6", linewidth=0.9, alpha=0.7,
                label="sole COM v_z (m/s)")
        # Mark liftoff transitions for directional sanity checks.
        if s["liftoff"].any():
            t_lift = t[s["liftoff"]]
            ax.vlines(t_lift, -40, 40, colors="#f39c12", linewidth=0.6,
                  alpha=0.35, label="Liftoff events")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylim(-40, 40)
        ax.set_title(title)
        ax.set_ylabel("a_z, Fz/m, v_z")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    _draw(axes[0, 0], gt_p1, "GT P1 — sole COM acceleration", "#1f77b4")
    _draw(axes[0, 1], gt_p2, "GT P2 — sole COM acceleration", "#d62728")
    _draw(axes[1, 0], gen_p1, "Gen P1 — sole COM acceleration", "#1f77b4")
    _draw(axes[1, 1], gen_p2, "Gen P2 — sole COM acceleration", "#d62728")

    # Direction diagnostics in (v_z, a_z) space
    ax = axes[2, 0]
    for label, s, c, m in [
        ("GT P1", gt_p1, "#1f77b4", "o"),
        ("GT P2", gt_p2, "#d62728", "o"),
        ("Gen P1", gen_p1, "#1f77b4", "x"),
        ("Gen P2", gen_p2, "#d62728", "x"),
    ]:
        step = max(1, len(s["vz"]) // 500)
        idx = np.arange(0, len(s["vz"]), step)
        ax.scatter(s["vz"][idx], s["az"][idx], s=10, marker=m,
                   color=c, alpha=0.22, label=f"{label} (subsampled)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("sole COM v_z (m/s)")
    ax.set_ylabel("sole COM a_z (m/s²)")
    ax.set_title("Direction Check: Vertical Velocity vs Acceleration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    def _liftoff_up_ratio(s):
        m = s["liftoff"]
        if m.sum() == 0:
            return 0.0
        return 100.0 * np.mean((s["vz"][m] > 0.0) & (s["az"][m] > 0.0))

    def _airborne_up_ratio(s):
        m = ~s["contact"]
        if m.sum() == 0:
            return 0.0
        return 100.0 * np.mean(s["airborne_up"][m])

    def _airborne_gravity_like_ratio(s):
        m = ~s["contact"]
        if m.sum() == 0:
            return 0.0
        return 100.0 * np.mean(s["airborne_gravity_like"][m])

    metrics = {
        "GT P1 contact%": gt_p1["contact"].mean() * 100,
        "GT P2 contact%": gt_p2["contact"].mean() * 100,
        "Gen P1 contact%": gen_p1["contact"].mean() * 100,
        "Gen P2 contact%": gen_p2["contact"].mean() * 100,
        "GT liftoff (vz>0, az>0)%": 0.5 * (_liftoff_up_ratio(gt_p1) + _liftoff_up_ratio(gt_p2)),
        "Gen liftoff (vz>0, az>0)%": 0.5 * (_liftoff_up_ratio(gen_p1) + _liftoff_up_ratio(gen_p2)),
        "GT airborne (vz>0, az>0)%": 0.5 * (_airborne_up_ratio(gt_p1) + _airborne_up_ratio(gt_p2)),
        "Gen airborne (vz>0, az>0)%": 0.5 * (_airborne_up_ratio(gen_p1) + _airborne_up_ratio(gen_p2)),
        "GT airborne (az<-1)%": 0.5 * (_airborne_gravity_like_ratio(gt_p1) + _airborne_gravity_like_ratio(gt_p2)),
        "Gen airborne (az<-1)%": 0.5 * (_airborne_gravity_like_ratio(gen_p1) + _airborne_gravity_like_ratio(gen_p2)),
    }
    y = np.arange(len(metrics))
    vals = list(metrics.values())
    colors = ["#95a5a6"] * 4 + ["#2980b9", "#e67e22", "#2980b9", "#e67e22", "#2980b9", "#e67e22"]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(list(metrics.keys()), fontsize=8)
    ax.set_title("Directionality Diagnostics")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "foot_sole_acceleration.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Plotting — Torque Comparison (6 panels)
# ═══════════════════════════════════════════════════════════════

def plot_torque_comparison(gt_torques, gen_torques, text, clip_id, out_dir):
    """Generate 6-panel torque comparison figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f'Torque Analysis — Clip {clip_id}\n"{text}"',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Overlaid histograms
    ax1 = fig.add_subplot(gs[0, 0])
    gt_h = gt_torques[:, 6:].ravel()
    gen_h = gen_torques[:, 6:].ravel()
    bins = np.linspace(-100, 100, 101)
    ax1.hist(gt_h, bins=bins, density=True, alpha=0.5, color="steelblue",
             label=f"GT (N={gt_torques.shape[0]})")
    ax1.hist(gen_h, bins=bins, density=True, alpha=0.5, color="orangered",
             label=f"Gen (N={gen_torques.shape[0]})")
    ax1.set_xlabel("Torque (Nm)")
    ax1.set_ylabel("Density")
    ax1.set_title("All Hinge Torques: Distribution")
    ax1.legend(fontsize=9)
    ax1.axvline(0, color="k", linewidth=0.5)
    for p, ls in [(95, "--"), (99, ":")]:
        ax1.axvline(np.percentile(np.abs(gt_h), p), color="steelblue",
                    linestyle=ls, alpha=0.6)
        ax1.axvline(np.percentile(np.abs(gen_h), p), color="orangered",
                    linestyle=ls, alpha=0.6)
    ax1.grid(alpha=0.3)

    # Panel 2: Per-group bar chart (mean + P95)
    ax2 = fig.add_subplot(gs[0, 1])
    group_names = list(BODY_GROUPS.keys())
    x = np.arange(len(group_names))
    width = 0.2
    for offset, (metric, label_suffix, alpha) in enumerate([
        ("mean", "mean", 0.9), ("p95", "P95", 0.6)
    ]):
        gt_vals, gen_vals = [], []
        for name in group_names:
            sl = BODY_GROUPS[name]
            gt_v = np.abs(gt_torques[:, sl]).ravel()
            gen_v = np.abs(gen_torques[:, sl]).ravel()
            if metric == "mean":
                gt_vals.append(gt_v.mean())
                gen_vals.append(gen_v.mean())
            else:
                gt_vals.append(np.percentile(gt_v, 95))
                gen_vals.append(np.percentile(gen_v, 95))
        ax2.bar(x - width - width/2 + offset * width, gt_vals, width,
                label=f"GT {label_suffix}", color="steelblue", alpha=alpha)
        ax2.bar(x + width/2 + offset * width, gen_vals, width,
                label=f"Gen {label_suffix}", color="orangered", alpha=alpha)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("|τ| (Nm)")
    ax2.set_title("Torque by Body Group")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Per-body ratio heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    n_bodies = len(BODY_NAMES_JOINTS)
    pcts = [50, 75, 90, 95]
    ratio_map = np.zeros((n_bodies, len(pcts)))
    for b in range(n_bodies):
        dof_start = 6 + b * 3
        gt_body = np.abs(gt_torques[:, dof_start:dof_start + 3]).ravel()
        gen_body = np.abs(gen_torques[:, dof_start:dof_start + 3]).ravel()
        for j, p in enumerate(pcts):
            ratio_map[b, j] = np.percentile(gen_body, p) / max(np.percentile(gt_body, p), 0.1)
    im = ax3.imshow(ratio_map, aspect="auto", cmap="RdYlGn_r", vmin=0.5, vmax=3.0)
    ax3.set_xticks(range(len(pcts)))
    ax3.set_xticklabels([f"P{p}" for p in pcts])
    ax3.set_yticks(range(n_bodies))
    ax3.set_yticklabels(BODY_NAMES_JOINTS, fontsize=7)
    ax3.set_title("Gen/GT Ratio per Body")
    for i in range(n_bodies):
        for j in range(len(pcts)):
            v = ratio_map[i, j]
            color = "black" if 0.7 < v < 2.0 else "white"
            ax3.text(j, i, f"{v:.1f}", ha="center", va="center",
                     fontsize=6, color=color)
    fig.colorbar(im, ax=ax3, shrink=0.7, label="Gen/GT")

    # Panel 4: Per-body |τ| percentile heatmap (GT + Gen side by side)
    ax4 = fig.add_subplot(gs[1, 1])
    pcts_abs = [50, 75, 90, 95]
    heatmap = np.zeros((n_bodies, len(pcts_abs) * 2))
    xtick_labels = []
    for j, p in enumerate(pcts_abs):
        xtick_labels.extend([f"GT\nP{p}", f"Gen\nP{p}"])
        for b in range(n_bodies):
            dof_start = 6 + b * 3
            heatmap[b, j * 2] = np.percentile(np.abs(gt_torques[:, dof_start:dof_start+3]).ravel(), p)
            heatmap[b, j * 2 + 1] = np.percentile(np.abs(gen_torques[:, dof_start:dof_start+3]).ravel(), p)
    im2 = ax4.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax4.set_xticks(range(len(pcts_abs) * 2))
    ax4.set_xticklabels(xtick_labels, fontsize=6)
    ax4.set_yticks(range(n_bodies))
    ax4.set_yticklabels(BODY_NAMES_JOINTS, fontsize=7)
    ax4.set_title("|τ| Percentiles (Nm)")
    for i in range(n_bodies):
        for j in range(len(pcts_abs) * 2):
            v = heatmap[i, j]
            color = "black" if v < heatmap.max() * 0.65 else "white"
            ax4.text(j, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=5, color=color)
    fig.colorbar(im2, ax=ax4, shrink=0.7, label="Nm")

    # Panels 5+6: Torque time series (overlaid GT vs Gen)
    time_gt = np.arange(gt_torques.shape[0]) / FPS
    time_gen = np.arange(gen_torques.shape[0]) / FPS

    ax5 = fig.add_subplot(gs[2, 0])
    for name, sl in list(BODY_GROUPS.items())[:3]:
        c = GROUP_COLORS[name]
        ax5.plot(time_gt, np.abs(gt_torques[:, sl]).mean(axis=1),
                 color=c, linewidth=1.0, alpha=0.8, label=name)
        ax5.plot(time_gen, np.abs(gen_torques[:, sl]).mean(axis=1),
                 color=c, linewidth=1.0, alpha=0.4, linestyle="--")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("|τ| mean (Nm)")
    ax5.set_title("Legs & Spine (solid=GT, dashed=Gen)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    for name, sl in list(BODY_GROUPS.items())[3:]:
        c = GROUP_COLORS[name]
        ax6.plot(time_gt, np.abs(gt_torques[:, sl]).mean(axis=1),
                 color=c, linewidth=1.0, alpha=0.8, label=name)
        ax6.plot(time_gen, np.abs(gen_torques[:, sl]).mean(axis=1),
                 color=c, linewidth=1.0, alpha=0.4, linestyle="--")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("|τ| mean (Nm)")
    ax6.set_title("Arms (solid=GT, dashed=Gen)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    path = os.path.join(out_dir, "torque_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Plotting — Inter-Person Forces (4 panels)
# ═══════════════════════════════════════════════════════════════

def plot_forces(forces_gt, forces_gen, text, clip_id, out_dir):
    """Generate 4-panel inter-person force comparison figure."""
    if forces_gt is None or forces_gen is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Inter-Person Forces — Clip {clip_id}\n"{text}"',
                 fontsize=13, fontweight='bold')

    T_gt = len(forces_gt["force_mag_p1"])
    T_gen = len(forces_gen["force_mag_p1"])
    t_gt = np.arange(T_gt) / FPS
    t_gen = np.arange(T_gen) / FPS

    # Panel A: Horizontal force
    ax = axes[0, 0]
    ax.plot(t_gt, forces_gt["horiz_force_p1"], color="#1f77b4", linewidth=1.0,
            alpha=0.8, label='GT P1')
    ax.plot(t_gt, forces_gt["horiz_force_p2"], color="#d62728", linewidth=1.0,
            alpha=0.8, label='GT P2')
    ax.plot(t_gen, forces_gen["horiz_force_p1"], color="#1f77b4", linewidth=1.0,
            alpha=0.4, linestyle="--", label='Gen P1')
    ax.plot(t_gen, forces_gen["horiz_force_p2"], color="#d62728", linewidth=1.0,
            alpha=0.4, linestyle="--", label='Gen P2')
    ax.set_ylabel("Force (N)")
    ax.set_title("Horizontal Force (solid=GT, dashed=Gen)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel B: Force along inter-person axis
    ax = axes[0, 1]
    ax.plot(t_gt, forces_gt["force_along_p1"], color="#2ca02c", linewidth=1.0,
            alpha=0.8, label='GT P1')
    ax.plot(t_gt, forces_gt["force_along_p2"], color="#ff7f0e", linewidth=1.0,
            alpha=0.8, label='GT P2')
    ax.plot(t_gen, forces_gen["force_along_p1"], color="#2ca02c", linewidth=1.0,
            alpha=0.4, linestyle="--", label='Gen P1')
    ax.plot(t_gen, forces_gen["force_along_p2"], color="#ff7f0e", linewidth=1.0,
            alpha=0.4, linestyle="--", label='Gen P2')
    ax.set_ylabel("Force (N)")
    ax.set_title("Force Along Inter-Person Axis")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)

    # Panel C: CoM distance
    ax = axes[1, 0]
    ax.plot(t_gt, forces_gt["com_dist"], color="#9467bd", linewidth=1.5,
            label='GT')
    ax.plot(t_gen, forces_gen["com_dist"], color="#9467bd", linewidth=1.5,
            linestyle="--", label='Gen')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("CoM Distance Between Persons")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel D: Summary bar chart
    ax = axes[1, 1]
    metrics = {
        'Horiz F\nmean': (
            np.mean([forces_gt["horiz_force_p1"].mean(),
                     forces_gt["horiz_force_p2"].mean()]),
            np.mean([forces_gen["horiz_force_p1"].mean(),
                     forces_gen["horiz_force_p2"].mean()]),
        ),
        'Horiz F\nP95': (
            np.mean([np.percentile(forces_gt["horiz_force_p1"], 95),
                     np.percentile(forces_gt["horiz_force_p2"], 95)]),
            np.mean([np.percentile(forces_gen["horiz_force_p1"], 95),
                     np.percentile(forces_gen["horiz_force_p2"], 95)]),
        ),
        'CoM dist\nmean': (
            forces_gt["com_dist"].mean(),
            forces_gen["com_dist"].mean(),
        ),
    }
    x_bar = np.arange(len(metrics))
    w = 0.35
    gt_bar = [v[0] for v in metrics.values()]
    gen_bar = [v[1] for v in metrics.values()]
    ax.bar(x_bar - w/2, gt_bar, w, label="GT", color="steelblue", alpha=0.8)
    ax.bar(x_bar + w/2, gen_bar, w, label="Gen", color="orangered", alpha=0.8)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax.set_title("Force Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (gt_v, gen_v) in enumerate(metrics.values()):
        ratio = gen_v / max(gt_v, 0.01)
        ax.text(i, max(gt_v, gen_v) * 1.05, f"×{ratio:.2f}",
                ha="center", fontsize=8, color="gray")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "forces.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Plotting — Root Residuals & Contact State (new, from physics_analysis)
# ═══════════════════════════════════════════════════════════════

def plot_root_residuals(gt_torques_list, gen_torques_list,
                        gt_pos_p1, gt_pos_p2, gen_pos_p1, gen_pos_p2,
                        text, clip_id, out_dir):
    """Root PD force analysis + contact state + torque heatmaps.

    Root forces (DOFs 0:3) are now from PD forward sim — physically grounded
    (gravity + contacts handled by solver), not skyhook residuals."""
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(f'Root PD Forces & Contact Analysis — Clip {clip_id}\n"{text}"',
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Contact State ──
    ax = fig.add_subplot(gs[0, :])
    if gt_pos_p1 is not None and gt_pos_p2 is not None:
        T_gt = min(gt_pos_p1.shape[0], gt_pos_p2.shape[0])
        contact_state_gt, contact_A_gt, contact_B_gt = compute_contact_state(
            gt_pos_p1[:T_gt], gt_pos_p2[:T_gt])
        t_gt = np.arange(T_gt) / FPS

        regime_colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']
        regime_labels = ['Both float', 'A float only', 'B float only', 'Both grounded']
        for r in range(4):
            mask = contact_state_gt == r
            if mask.any():
                ax.fill_between(t_gt, 0, 1, where=mask,
                                color=regime_colors[r], alpha=0.4,
                                label=f"GT: {regime_labels[r]}")

        # Also show Gen contact state (offset vertically)
        if gen_pos_p1 is not None and gen_pos_p2 is not None:
            T_gen = min(gen_pos_p1.shape[0], gen_pos_p2.shape[0])
            cs_gen, _, _ = compute_contact_state(gen_pos_p1[:T_gen], gen_pos_p2[:T_gen])
            t_gen = np.arange(T_gen) / FPS
            for r in range(4):
                mask = cs_gen == r
                if mask.any():
                    ax.fill_between(t_gen, 1.2, 2.2, where=mask,
                                    color=regime_colors[r], alpha=0.3)

        ax.set_yticks([0.5, 1.7])
        ax.set_yticklabels(["GT", "Gen"], fontsize=10)
        ax.set_ylim(-0.1, 2.5)

        # Count frames in each state
        pct_both_g = (contact_state_gt == 3).sum() / T_gt * 100
        pct_solvable = ((contact_state_gt != 3).sum()) / T_gt * 100
        ax.text(0.98, 0.95, f"GT: {pct_both_g:.0f}% both grounded, "
                f"{pct_solvable:.0f}% solvable",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_title("Contact State (shaded = foot on ground)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.2)

    # ── Panel 2: Root PD Force Magnitude (GT) ──
    ax = fig.add_subplot(gs[1, 0])
    for pidx, torques in enumerate(gt_torques_list):
        root_force = torques[:, :3]  # translational root forces
        root_mag = np.linalg.norm(root_force, axis=-1)
        t = np.arange(len(root_mag)) / FPS
        color = "#3498db" if pidx == 0 else "#e74c3c"
        ax.plot(t, root_mag, color=color, linewidth=1.2, alpha=0.8,
                label=f"P{pidx+1} |root force|")
    ax.axhline(BODY_MASS * GRAVITY, color="gray", linestyle=":", alpha=0.5,
               label=f"Body weight ({BODY_MASS * GRAVITY:.0f}N)")
    ax.set_ylabel("Force (N)")
    ax.set_title("GT Root PD Forces")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 3: Root PD Force Magnitude (Gen) ──
    ax = fig.add_subplot(gs[1, 1])
    for pidx, torques in enumerate(gen_torques_list):
        root_force = torques[:, :3]
        root_mag = np.linalg.norm(root_force, axis=-1)
        t = np.arange(len(root_mag)) / FPS
        color = "#3498db" if pidx == 0 else "#e74c3c"
        ax.plot(t, root_mag, color=color, linewidth=1.2, alpha=0.8,
                label=f"P{pidx+1} |root force|")
    ax.axhline(BODY_MASS * GRAVITY, color="gray", linestyle=":", alpha=0.5,
               label=f"Body weight ({BODY_MASS * GRAVITY:.0f}N)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Gen Root PD Forces")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 4: Torque Heatmap GT (person 0) ──
    ax = fig.add_subplot(gs[2, 0])
    tau_gt = gt_torques_list[0]
    T_gt = tau_gt.shape[0]
    n_bodies = len(BODY_NAMES_JOINTS)
    body_torques_gt = np.zeros((T_gt, n_bodies))
    for b in range(n_bodies):
        dof_s = 6 + b * 3
        body_torques_gt[:, b] = np.linalg.norm(tau_gt[:, dof_s:dof_s+3], axis=-1)
    time_gt = np.arange(T_gt) / FPS
    im = ax.imshow(body_torques_gt.T, aspect='auto', cmap='hot',
                   extent=[0, time_gt[-1], n_bodies - 0.5, -0.5],
                   interpolation='nearest')
    ax.set_yticks(range(n_bodies))
    ax.set_yticklabels(BODY_NAMES_JOINTS, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_title("GT — Per-Joint |τ| Heatmap (P1)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Nm")

    # ── Panel 5: Torque Heatmap Gen (person 0) ──
    ax = fig.add_subplot(gs[2, 1])
    tau_gen = gen_torques_list[0]
    T_gen = tau_gen.shape[0]
    body_torques_gen = np.zeros((T_gen, n_bodies))
    for b in range(n_bodies):
        dof_s = 6 + b * 3
        body_torques_gen[:, b] = np.linalg.norm(tau_gen[:, dof_s:dof_s+3], axis=-1)
    time_gen = np.arange(T_gen) / FPS
    im = ax.imshow(body_torques_gen.T, aspect='auto', cmap='hot',
                   extent=[0, time_gen[-1], n_bodies - 0.5, -0.5],
                   interpolation='nearest',
                   vmin=0, vmax=max(body_torques_gt.max(), body_torques_gen.max()))
    ax.set_yticks(range(n_bodies))
    ax.set_yticklabels(BODY_NAMES_JOINTS, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_title("Gen — Per-Joint |τ| Heatmap (P1)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Nm")

    # ── Panel 6: Root Residual breakdown by axis (GT P1) ──
    ax = fig.add_subplot(gs[3, 0])
    root_gt = gt_torques_list[0][:, :6]
    t = np.arange(root_gt.shape[0]) / FPS
    axis_names = ['X', 'Y', 'Z']
    axis_colors = ['#e74c3c', '#2ecc71', '#3498db']
    for d in range(3):
        ax.plot(t, root_gt[:, d], color=axis_colors[d], linewidth=1.0,
                alpha=0.8, label=f"F_{axis_names[d]}")
    for d in range(3, 6):
        ax.plot(t, root_gt[:, d], color=axis_colors[d-3], linewidth=1.0,
                alpha=0.4, linestyle="--", label=f"τ_{axis_names[d-3]}")
    ax.set_ylabel("Force (N) / Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("GT P1 Root DOFs (solid=force, dashed=torque)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

    # ── Panel 7: Root Residual breakdown by axis (Gen P1) ──
    ax = fig.add_subplot(gs[3, 1])
    root_gen = gen_torques_list[0][:, :6]
    t = np.arange(root_gen.shape[0]) / FPS
    for d in range(3):
        ax.plot(t, root_gen[:, d], color=axis_colors[d], linewidth=1.0,
                alpha=0.8, label=f"F_{axis_names[d]}")
    for d in range(3, 6):
        ax.plot(t, root_gen[:, d], color=axis_colors[d-3], linewidth=1.0,
                alpha=0.4, linestyle="--", label=f"τ_{axis_names[d-3]}")
    ax.set_ylabel("Force (N) / Torque (Nm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Gen P1 Root DOFs (solid=force, dashed=torque)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)

    # ── Panel 8: Root Residual Summary ──
    ax = fig.add_subplot(gs[4, 0])
    summary_names = ["GT P1", "GT P2", "Gen P1", "Gen P2"]
    summary_means = []
    summary_p95 = []
    all_torques = gt_torques_list + gen_torques_list
    for torques in all_torques:
        root_mag = np.linalg.norm(torques[:, :3], axis=-1)
        summary_means.append(root_mag.mean())
        summary_p95.append(np.percentile(root_mag, 95))
    x_s = np.arange(len(summary_names))
    colors_s = ["steelblue", "steelblue", "orangered", "orangered"]
    ax.bar(x_s - 0.15, summary_means, 0.3, label="Mean", alpha=0.9,
           color=colors_s)
    ax.bar(x_s + 0.15, summary_p95, 0.3, label="P95", alpha=0.6,
           color=colors_s)
    ax.set_xticks(x_s)
    ax.set_xticklabels(summary_names)
    ax.set_ylabel("|Root Residual| (N)")
    ax.set_title("Root Residual Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i in range(len(summary_names)):
        ax.text(i, summary_p95[i] * 1.03, f"{summary_means[i]:.0f}N",
                ha="center", fontsize=8, color="gray")

    # ── Panel 9: Hinge torque summary comparison ──
    ax = fig.add_subplot(gs[4, 1])
    gt_avg_hinge = np.mean([np.abs(t[:, 6:]).mean() for t in gt_torques_list])
    gen_avg_hinge = np.mean([np.abs(t[:, 6:]).mean() for t in gen_torques_list])
    gt_p95_hinge = np.mean([np.percentile(np.abs(t[:, 6:]), 95) for t in gt_torques_list])
    gen_p95_hinge = np.mean([np.percentile(np.abs(t[:, 6:]), 95) for t in gen_torques_list])
    bar_x = np.arange(2)
    ax.bar(bar_x - 0.15, [gt_avg_hinge, gt_p95_hinge], 0.3,
           label="GT", color="steelblue", alpha=0.8)
    ax.bar(bar_x + 0.15, [gen_avg_hinge, gen_p95_hinge], 0.3,
           label="Gen", color="orangered", alpha=0.8)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["Mean |τ_hinge|", "P95 |τ_hinge|"])
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Overall Hinge Torque Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ratio_mean = gen_avg_hinge / max(gt_avg_hinge, 0.1)
    ratio_p95 = gen_p95_hinge / max(gt_p95_hinge, 0.1)
    ax.text(0, max(gt_avg_hinge, gen_avg_hinge) * 1.08,
            f"×{ratio_mean:.2f}", ha="center", fontsize=9, color="gray")
    ax.text(1, max(gt_p95_hinge, gen_p95_hinge) * 1.08,
            f"×{ratio_p95:.2f}", ha="center", fontsize=9, color="gray")

    path = os.path.join(out_dir, "root_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Plotting — Skeleton Keyframes
# ═══════════════════════════════════════════════════════════════

def select_key_frames(T, torques=None, forces=None, n_extra=4):
    """Select interesting frames: first, last, evenly spaced, peak torque/force."""
    from scipy.signal import find_peaks
    frames = {0, T - 1}

    # Evenly spaced
    step = max(1, T // (n_extra + 1))
    for i in range(1, n_extra + 1):
        frames.add(min(i * step, T - 1))

    # Peak torque moment
    if torques is not None:
        total_tau = np.linalg.norm(torques[:, 6:], axis=-1)
        frames.add(int(np.argmax(total_tau)))
        try:
            peaks, _ = find_peaks(total_tau, distance=FPS, prominence=5.0)
            for p in peaks[:2]:
                frames.add(int(p))
        except Exception:
            pass

    # Peak interaction force
    if forces is not None and "horiz_force_p1" in forces:
        total_f = forces["horiz_force_p1"] + forces["horiz_force_p2"]
        frames.add(int(np.argmax(total_f[:T])))

    return sorted(f for f in frames if 0 <= f < T)


def plot_skeleton_keyframes(gt_pos_p1, gt_pos_p2, gen_pos_p1, gen_pos_p2,
                            key_frames, text, clip_id, out_dir):
    """2D skeleton keyframes at selected moments (front view, X vs Z)."""
    n_frames = len(key_frames)
    n_cols = min(n_frames, 6)
    n_rows = max(1, (n_frames + n_cols - 1) // n_cols) * 2  # GT row + Gen row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f'Skeleton Keyframes — Clip {clip_id}\n"{text}"',
                 fontsize=13, fontweight='bold')

    if n_rows == 1:
        axes = axes[np.newaxis, :] if n_cols > 1 else np.array([[axes]])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_pair in range(n_rows // 2):
        for col, frame in enumerate(key_frames):
            if col >= n_cols:
                break
            # GT row
            gt_row = row_pair * 2
            gen_row = row_pair * 2 + 1
            if gt_row >= n_rows or gen_row >= n_rows:
                break

            for ax_row, pos_p1, pos_p2, source_label in [
                (gt_row, gt_pos_p1, gt_pos_p2, "GT"),
                (gen_row, gen_pos_p1, gen_pos_p2, "Gen"),
            ]:
                ax = axes[ax_row, col]
                T = min(pos_p1.shape[0], pos_p2.shape[0])
                if frame >= T:
                    ax.set_visible(False)
                    continue

                for pos, color, plabel in [
                    (pos_p1[frame], '#3498db', 'P1'),
                    (pos_p2[frame], '#e74c3c', 'P2'),
                ]:
                    x = pos[:, 0]
                    z = pos[:, UP_AXIS]
                    for child, parent in SMPL_BONES:
                        if child < 22 and parent < 22:
                            ax.plot([x[parent], x[child]], [z[parent], z[child]],
                                    color=color, linewidth=2, alpha=0.8)
                    ax.scatter(x, z, c=color, s=15, zorder=5, alpha=0.9)

                t_sec = frame / FPS
                ax.set_title(f"{source_label} t={t_sec:.2f}s (f{frame})", fontsize=8)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2)

    # Hide unused axes
    for i in range(n_rows):
        for j in range(n_cols):
            if j >= len(key_frames):
                axes[i, j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "skeleton_keyframes.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# Newton MP4 Video Generation
# ═══════════════════════════════════════════════════════════════

def generate_newton_video(clip_id, mp4_path, device="cuda:0",
                          id_torque=False, pd_torque=False):
    """Generate Newton MP4 by calling view_gt_vs_gen.py as a subprocess.

    Args:
        id_torque: pass --id-torque to add inverse-dynamics torque sim
        pd_torque: pass --pd-torque to add PD tracking torque sim
    """
    script = os.path.join(PROJECT_ROOT, "prepare4", "view_gt_vs_gen.py")
    cmd = [
        sys.executable, script,
        "--clip", str(clip_id),
        "--save-mp4", mp4_path,
        "--cam-preset", "side",
        "--device", device,
    ]
    if id_torque:
        cmd.append("--id-torque")
    if pd_torque:
        cmd.append("--pd-torque")

    mode_desc = "kinematic"
    if id_torque and pd_torque:
        mode_desc = "kinematic + ID + PD torque"
    elif id_torque:
        mode_desc = "kinematic + ID torque"
    elif pd_torque:
        mode_desc = "kinematic + PD torque"
    print(f"    Generating Newton video ({mode_desc}): {mp4_path}")

    timeout = 600
    if id_torque or pd_torque:
        timeout = 1800

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=PROJECT_ROOT, timeout=timeout,
    )
    if result.returncode != 0:
        print(f"    ERROR generating video:")
        print(result.stderr[-500:] if result.stderr else "No stderr")
        return False
    for line in result.stdout.split('\n'):
        if any(k in line for k in ['Auto y_offset', 'Camera:', 'frames',
                                    'Saved:', 'Pre-simulating', 'Done.']):
            print(f"      {line.strip()}")
    return True


# ═══════════════════════════════════════════════════════════════
# Per-Clip Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

def process_clip(clip_id, label, out_base, device="cuda:0",
                 generate_video=True, generate_torque_video=False,
                 eval_aligned=True):
    """Run full analysis for one clip."""
    clip_dir = os.path.join(out_base, f"clip_{clip_id}_{label}")
    os.makedirs(clip_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Clip {clip_id} — {label}")
    print(f" Output: {clip_dir}")
    print(f"{'='*70}")

    text = load_motion_text(clip_id)
    print(f"  Description: {text}")

    # ── Load data ──
    gen_persons, gen_text = load_gen_persons(clip_id)
    if gen_persons is None:
        print(f"  SKIP: No generated data for clip {clip_id}")
        return None
    if gen_text:
        text = gen_text

    gt_persons = load_gt_persons(clip_id)
    if gt_persons is None:
        print(f"  SKIP: No GT data for clip {clip_id}")
        return None

    eval_info = get_eval_temporal_info(clip_id) if eval_aligned else None
    if eval_info is not None:
        print(f"  Eval alignment: raw={eval_info['raw_len']} "
              f"crop={eval_info['eval_crop_len']} "
              f"motion_lens={eval_info['eval_motion_lens']} "
              f"ids={eval_info['eval_ids_len']} "
              f"gen_expected={eval_info['eval_gen_len']}")

    # ── Compute torques ──
    print(f"  Computing GT torques...")
    gt_torques_list = []
    gt_jq_list = []
    for i, p in enumerate(gt_persons):
        torques, jq = compute_torques_for_person(p, "gt", device)
        if torques is None:
            print(f"    GT P{i+1}: FAILED")
            return None
        gt_torques_list.append(torques)
        gt_jq_list.append(jq)
        print(f"    P{i+1}: {torques.shape[0]} frames, "
              f"|τ_hinge| mean={np.abs(torques[:, 6:]).mean():.1f} Nm, "
              f"|root| mean={np.linalg.norm(torques[:, :3], axis=-1).mean():.0f} N")

    print(f"  Computing Generated torques...")
    gen_torques_list = []
    gen_jq_list = []
    for i, p in enumerate(gen_persons):
        torques, jq = compute_torques_for_person(p, "generated", device)
        if torques is None:
            print(f"    Gen P{i+1}: FAILED")
            return None
        gen_torques_list.append(torques)
        gen_jq_list.append(jq)
        print(f"    P{i+1}: {torques.shape[0]} frames, "
              f"|τ_hinge| mean={np.abs(torques[:, 6:]).mean():.1f} Nm, "
              f"|root| mean={np.linalg.norm(torques[:, :3], axis=-1).mean():.0f} N")

    # Average for plotting
    T_gt = min(t.shape[0] for t in gt_torques_list)
    T_gen = min(t.shape[0] for t in gen_torques_list)
    gt_torques_avg = np.mean([t[:T_gt] for t in gt_torques_list], axis=0)
    gen_torques_avg = np.mean([t[:T_gen] for t in gen_torques_list], axis=0)

    # ── Load positions for force analysis ──
    print(f"  Loading positions...")
    print(f"    Foot points: ankle idx={FOOT_ANKLE_INDICES}, forefoot idx={FOOT_FOREFOOT_INDICES}")
    print(f"    Sole point used for contact/accel = midpoint(ankle, forefoot)")
    gt_pos1, gt_pos2 = load_positions_zup(clip_id, "gt")
    gen_pos1, gen_pos2 = load_positions_zup(clip_id, "generated")

    # Enforce eval.py temporal policy for GT path to keep comparisons fair.
    if eval_info is not None:
        gt_keep = eval_info["eval_motion_lens"]
        if gt_keep > 0:
            gt_torques_list = [t[:gt_keep] for t in gt_torques_list]
            gt_jq_list = [q[:gt_keep] for q in gt_jq_list]
            if gt_pos1 is not None:
                gt_pos1 = gt_pos1[:gt_keep]
                gt_pos2 = gt_pos2[:gt_keep]
            T_gt = min(t.shape[0] for t in gt_torques_list)
            gt_torques_avg = np.mean([t[:T_gt] for t in gt_torques_list], axis=0)

    forces_gt, forces_gen = None, None
    if gt_pos1 is not None and gen_pos1 is not None:
        forces_gt = compute_interaction_force(gt_pos1, gt_pos2)
        forces_gen = compute_interaction_force(gen_pos1, gen_pos2)

    if eval_info is not None and gen_pos1 is not None:
        gen_actual = int(min(gen_pos1.shape[0], gen_pos2.shape[0]))
        print(f"  Eval alignment check: gen_actual={gen_actual}, "
              f"gen_expected(ids*4)={eval_info['eval_gen_len']}")

    # ── Decompose interaction forces from root PD forces ──
    decomp_gt, decomp_gen = None, None
    if gt_pos1 is not None and gen_pos1 is not None:
        print(f"  Decomposing interaction forces (Newton's 3rd law)...")
        decomp_gt = decompose_interaction_forces(
            gt_torques_list[0], gt_torques_list[1], gt_pos1, gt_pos2)
        decomp_gen = decompose_interaction_forces(
            gen_torques_list[0], gen_torques_list[1], gen_pos1, gen_pos2)
        print(f"    GT: |F_int| mean={decomp_gt['F_int_mag'].mean():.1f} N, "
              f"P95={np.percentile(decomp_gt['F_int_mag'], 95):.1f} N")
        print(f"    Gen: |F_int| mean={decomp_gen['F_int_mag'].mean():.1f} N, "
              f"P95={np.percentile(decomp_gen['F_int_mag'], 95):.1f} N")

        # Diagnostic: frames where contact is detected but solved GRFz is near-zero.
        grf_eps = 50.0  # N
        gt_miss_p1 = np.mean(decomp_gt["contact_A"] & (decomp_gt["F_ground_A"][:, UP_AXIS] < grf_eps)) * 100
        gt_miss_p2 = np.mean(decomp_gt["contact_B"] & (decomp_gt["F_ground_B"][:, UP_AXIS] < grf_eps)) * 100
        gen_miss_p1 = np.mean(decomp_gen["contact_A"] & (decomp_gen["F_ground_A"][:, UP_AXIS] < grf_eps)) * 100
        gen_miss_p2 = np.mean(decomp_gen["contact_B"] & (decomp_gen["F_ground_B"][:, UP_AXIS] < grf_eps)) * 100
        print(f"    Contact but low GRFz (<{grf_eps:.0f}N): GT P1={gt_miss_p1:.1f}%, GT P2={gt_miss_p2:.1f}%")
        print(f"                                 Gen P1={gen_miss_p1:.1f}%, Gen P2={gen_miss_p2:.1f}%")

    # ── Generate plots ──
    print(f"  Generating plots...")

    # 1. Torque comparison (6 panels)
    plot_torque_comparison(gt_torques_avg, gen_torques_avg, text, clip_id, clip_dir)

    # 2. Inter-person forces (4 panels — CoM-based)
    plot_forces(forces_gt, forces_gen, text, clip_id, clip_dir)

    # 3. Root residuals + contact state + heatmaps (10 panels)
    plot_root_residuals(gt_torques_list, gen_torques_list,
                        gt_pos1, gt_pos2, gen_pos1, gen_pos2,
                        text, clip_id, clip_dir)

    # 4. Interaction force decomposition (6 panels — Newton's 3rd law)
    if decomp_gt is not None:
        plot_interaction_forces(decomp_gt, decomp_gen, text, clip_id, clip_dir)

    # 5. Foot-sole acceleration diagnostics
    if decomp_gt is not None:
        plot_foot_sole_acceleration(gt_pos1, gt_pos2, gen_pos1, gen_pos2,
                                    decomp_gt, decomp_gen, text, clip_id, clip_dir)

    # 6. Skeleton keyframes
    if gt_pos1 is not None and gen_pos1 is not None:
        key_frames = select_key_frames(
            min(gt_pos1.shape[0], gen_pos1.shape[0]),
            torques=gt_torques_avg,
            forces=forces_gt,
        )
        plot_skeleton_keyframes(gt_pos1, gt_pos2, gen_pos1, gen_pos2,
                                key_frames, text, clip_id, clip_dir)

    # 7. Newton MP4 video (kinematic only)
    if generate_video:
        mp4_path = os.path.join(clip_dir, "newton_video.mp4")
        generate_newton_video(clip_id, mp4_path, device)

    # 8. Torque-driven Newton MP4 videos
    if generate_torque_video:
        mp4_torque = os.path.join(clip_dir, "newton_video_torque.mp4")
        generate_newton_video(clip_id, mp4_torque, device,
                              id_torque=False, pd_torque=True)

    # 9. Save raw data
    data = {
        'clip_id': clip_id,
        'label': label,
        'text': text,
        'gt_torques_p1': gt_torques_list[0],
        'gt_torques_p2': gt_torques_list[1],
        'gen_torques_p1': gen_torques_list[0],
        'gen_torques_p2': gen_torques_list[1],
    }
    if gt_pos1 is not None:
        data['gt_positions_p1'] = gt_pos1
        data['gt_positions_p2'] = gt_pos2
    if gen_pos1 is not None:
        data['gen_positions_p1'] = gen_pos1
        data['gen_positions_p2'] = gen_pos2
    if decomp_gt is not None:
        data['gt_F_int'] = decomp_gt["F_int"]
        data['gt_F_ground_A'] = decomp_gt["F_ground_A"]
        data['gt_F_ground_B'] = decomp_gt["F_ground_B"]
        data['gen_F_int'] = decomp_gen["F_int"]
        data['gen_F_ground_A'] = decomp_gen["F_ground_A"]
        data['gen_F_ground_B'] = decomp_gen["F_ground_B"]

        gt_sole1 = compute_foot_sole_kinematics(gt_pos1)
        gt_sole2 = compute_foot_sole_kinematics(gt_pos2)
        gen_sole1 = compute_foot_sole_kinematics(gen_pos1)
        gen_sole2 = compute_foot_sole_kinematics(gen_pos2)
        data['gt_sole_com_acc_p1'] = gt_sole1["sole_com_acc"]
        data['gt_sole_com_acc_p2'] = gt_sole2["sole_com_acc"]
        data['gen_sole_com_acc_p1'] = gen_sole1["sole_com_acc"]
        data['gen_sole_com_acc_p2'] = gen_sole2["sole_com_acc"]
        data['gt_contact_p1'] = decomp_gt["contact_A"].astype(np.int8)
        data['gt_contact_p2'] = decomp_gt["contact_B"].astype(np.int8)
        data['gen_contact_p1'] = decomp_gen["contact_A"].astype(np.int8)
        data['gen_contact_p2'] = decomp_gen["contact_B"].astype(np.int8)
    if eval_info is not None:
        data['eval_raw_len'] = np.array([eval_info['raw_len']], dtype=np.int32)
        data['eval_crop_len'] = np.array([eval_info['eval_crop_len']], dtype=np.int32)
        data['eval_motion_lens'] = np.array([eval_info['eval_motion_lens']], dtype=np.int32)
        data['eval_ids_len'] = np.array([eval_info['eval_ids_len']], dtype=np.int32)
        data['eval_gen_expected_len'] = np.array([eval_info['eval_gen_len']], dtype=np.int32)
    np.savez(os.path.join(clip_dir, "data.npz"), **data)
    print(f"    Saved: data.npz")

    # ── Per-clip summary statistics ──
    summary = {
        'clip_id': clip_id,
        'label': label,
        'text': text,
        'gt_frames': T_gt,
        'gen_frames': T_gen,
        'gt_duration_s': T_gt / FPS,
        'gen_duration_s': T_gen / FPS,
        'gt_mean_hinge_torque': float(np.abs(gt_torques_avg[:, 6:]).mean()),
        'gen_mean_hinge_torque': float(np.abs(gen_torques_avg[:, 6:]).mean()),
        'gt_p95_hinge_torque': float(np.percentile(np.abs(gt_torques_avg[:, 6:]), 95)),
        'gen_p95_hinge_torque': float(np.percentile(np.abs(gen_torques_avg[:, 6:]), 95)),
        'torque_ratio_mean': float(np.abs(gen_torques_avg[:, 6:]).mean() /
                                   max(np.abs(gt_torques_avg[:, 6:]).mean(), 0.1)),
        'gt_mean_root_residual': float(np.mean([
            np.linalg.norm(t[:, :3], axis=-1).mean() for t in gt_torques_list])),
        'gen_mean_root_residual': float(np.mean([
            np.linalg.norm(t[:, :3], axis=-1).mean() for t in gen_torques_list])),
    }
    if forces_gt is not None:
        summary['gt_mean_horiz_force'] = float(np.mean([
            forces_gt["horiz_force_p1"].mean(), forces_gt["horiz_force_p2"].mean()]))
        summary['gen_mean_horiz_force'] = float(np.mean([
            forces_gen["horiz_force_p1"].mean(), forces_gen["horiz_force_p2"].mean()]))
    if decomp_gt is not None:
        summary['gt_mean_F_int'] = float(decomp_gt["F_int_mag"].mean())
        summary['gen_mean_F_int'] = float(decomp_gen["F_int_mag"].mean())
        summary['gt_p95_F_int'] = float(np.percentile(decomp_gt["F_int_mag"], 95))
        summary['gen_p95_F_int'] = float(np.percentile(decomp_gen["F_int_mag"], 95))
        summary['F_int_ratio_mean'] = float(decomp_gen["F_int_mag"].mean() /
                                            max(decomp_gt["F_int_mag"].mean(), 0.1))

        grf_eps = 50.0
        summary['gt_contact_low_grfz_pct_p1'] = float(
            np.mean(decomp_gt["contact_A"] & (decomp_gt["F_ground_A"][:, UP_AXIS] < grf_eps)) * 100)
        summary['gt_contact_low_grfz_pct_p2'] = float(
            np.mean(decomp_gt["contact_B"] & (decomp_gt["F_ground_B"][:, UP_AXIS] < grf_eps)) * 100)
        summary['gen_contact_low_grfz_pct_p1'] = float(
            np.mean(decomp_gen["contact_A"] & (decomp_gen["F_ground_A"][:, UP_AXIS] < grf_eps)) * 100)
        summary['gen_contact_low_grfz_pct_p2'] = float(
            np.mean(decomp_gen["contact_B"] & (decomp_gen["F_ground_B"][:, UP_AXIS] < grf_eps)) * 100)
        summary['foot_sole_proxy'] = "midpoint(ankle[7/8], forefoot[10/11])"
    if eval_info is not None:
        summary['eval_aligned'] = True
        summary['eval_raw_len'] = int(eval_info['raw_len'])
        summary['eval_crop_len'] = int(eval_info['eval_crop_len'])
        summary['eval_motion_lens'] = int(eval_info['eval_motion_lens'])
        summary['eval_ids_len'] = int(eval_info['eval_ids_len'])
        summary['eval_gen_expected_len'] = int(eval_info['eval_gen_len'])
        if gen_pos1 is not None:
            summary['gen_actual_len'] = int(min(gen_pos1.shape[0], gen_pos2.shape[0]))

    # Write per-clip summary
    with open(os.path.join(clip_dir, "summary.txt"), "w") as f:
        f.write(f"Clip {clip_id} — {label}\n")
        f.write(f"Description: {text}\n")
        f.write(f"{'='*60}\n\n")
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.3f}\n")
            else:
                f.write(f"  {k}: {v}\n")
    print(f"    Saved: summary.txt")

    return summary


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Newton physics analysis: videos + torques + forces")
    parser.add_argument("--clips", type=str, nargs="*", default=None,
                        help='Clip specs as "ID label" pairs. '
                             'Default: 1129 hit, 1147 pull, etc.')
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output/newton_analysis/)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip Newton MP4 video generation")
    parser.add_argument("--torque-video", action="store_true",
                        help="Generate torque-driven Newton video "
                             "(ID + PD sim side-by-side with kinematic)")
    parser.add_argument("--no-eval-aligned", action="store_true",
                        help="Disable eval.py temporal alignment (default: aligned)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    out_base = args.output_dir or os.path.join(PROJECT_ROOT, "output", "newton_analysis")
    os.makedirs(out_base, exist_ok=True)

    # Parse clip specs
    if args.clips:
        clips = []
        i = 0
        while i < len(args.clips):
            clip_id = args.clips[i]
            if i + 1 < len(args.clips) and not args.clips[i+1].isdigit():
                label = args.clips[i+1]
                i += 2
            else:
                label = load_motion_text(clip_id).split()[0] if load_motion_text(clip_id) else "clip"
                i += 1
            clips.append((clip_id, label))
    else:
        clips = DEFAULT_CLIPS

    print(f"Newton Physics Analysis Pipeline")
    print(f"  Output:  {out_base}")
    print(f"  Clips:   {len(clips)}")
    print(f"  Video:   {'yes' if not args.no_video else 'no'}")
    print(f"  Torque video: {'yes' if args.torque_video else 'no'}")
    print(f"  Device:  {args.device}")
    print(f"  Eval-aligned lengths: {'yes' if not args.no_eval_aligned else 'no'}")
    print(f"  Pipeline: prepare4 (IK from positions, FPS=30, angle normalize)")
    print()

    all_summaries = []
    t_start = _time.time()
    for clip_id, label in clips:
        summary = process_clip(clip_id, label, out_base,
                               device=args.device,
                               generate_video=not args.no_video,
                               generate_torque_video=args.torque_video,
                               eval_aligned=(not args.no_eval_aligned))
        if summary:
            all_summaries.append(summary)

    # ── Global summary ──
    elapsed = _time.time() - t_start
    print(f"\n{'='*70}")
    print(f" ANALYSIS COMPLETE — {len(all_summaries)}/{len(clips)} clips")
    print(f"{'='*70}\n")

    if all_summaries:
        # Print summary table
        print(f"{'Clip':>6s} {'Label':>8s} {'GT τ':>7s} {'Gen τ':>7s} "
              f"{'Ratio':>6s} {'GT Root':>8s} {'Gen Root':>8s} {'Frames':>8s}")
        print("-" * 65)
        for s in all_summaries:
            print(f"{s['clip_id']:>6s} {s['label']:>8s} "
                  f"{s['gt_mean_hinge_torque']:>7.1f} "
                  f"{s['gen_mean_hinge_torque']:>7.1f} "
                  f"{s['torque_ratio_mean']:>6.2f}x "
                  f"{s['gt_mean_root_residual']:>8.0f} "
                  f"{s['gen_mean_root_residual']:>8.0f} "
                  f"{s['gt_frames']:>3d}/{s['gen_frames']:<3d}")

        # Write global summary
        summary_path = os.path.join(out_base, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Newton Physics Analysis — Summary\n")
            f.write(f"Pipeline: prepare4 (IK from positions, FPS=30)\n")
            f.write(f"Eval-aligned lengths: {'yes' if not args.no_eval_aligned else 'no'}\n")
            f.write(f"Date: {_time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"{'Clip':>6s} {'Label':>8s} {'GT τ':>7s} {'Gen τ':>7s} "
                    f"{'Ratio':>6s} {'GT Root':>8s} {'Gen Root':>8s}\n")
            f.write("-" * 60 + "\n")
            for s in all_summaries:
                f.write(f"{s['clip_id']:>6s} {s['label']:>8s} "
                        f"{s['gt_mean_hinge_torque']:>7.1f} "
                        f"{s['gen_mean_hinge_torque']:>7.1f} "
                        f"{s['torque_ratio_mean']:>6.2f}x "
                        f"{s['gt_mean_root_residual']:>8.0f} "
                        f"{s['gen_mean_root_residual']:>8.0f}\n")
            f.write(f"\nOverall mean torque ratio: "
                    f"{np.mean([s['torque_ratio_mean'] for s in all_summaries]):.2f}x\n")
        print(f"\nGlobal summary: {summary_path}")

        # Write eval-alignment audit for fair-comparison traceability.
        audit_path = os.path.join(out_base, "eval_alignment_audit.txt")
        with open(audit_path, "w") as f:
            f.write("Eval Alignment Audit\n")
            f.write(f"Date: {_time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Eval-aligned mode: {'yes' if not args.no_eval_aligned else 'no'}\n")
            f.write(f"{'='*88}\n")
            f.write("clip,label,gt_frames,gen_frames,eval_motion_lens,eval_gen_expected_len,gen_actual_len,"
                    "gt_match,gen_match\n")

            gt_ok = []
            gen_ok = []
            for s in all_summaries:
                e_mlen = int(s.get('eval_motion_lens', -1))
                e_gen = int(s.get('eval_gen_expected_len', -1))
                g_act = int(s.get('gen_actual_len', -1))
                gt_match = (int(s['gt_frames']) == e_mlen) if e_mlen >= 0 else False
                gen_match = (g_act == e_gen) if (e_gen >= 0 and g_act >= 0) else False
                gt_ok.append(gt_match)
                gen_ok.append(gen_match)
                f.write(f"{s['clip_id']},{s['label']},{int(s['gt_frames'])},{int(s['gen_frames'])},"
                        f"{e_mlen},{e_gen},{g_act},{int(gt_match)},{int(gen_match)}\n")

            all_ok = bool(gt_ok) and bool(gen_ok) and all(gt_ok) and all(gen_ok)
            f.write(f"\nall_gt_match={all_ok and all(gt_ok)}\n")
            f.write(f"all_gen_match={all_ok and all(gen_ok)}\n")
            f.write(f"ready_for_torque_vq_phase={int(all_ok)}\n")
        print(f"Eval alignment audit: {audit_path}")

    print(f"\nOutput directory: {out_base}")


if __name__ == "__main__":
    main()
