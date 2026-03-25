"""
Shared PD control utilities for Newton physics simulation.

This module centralizes all PD-related constants, model building,
gain setup, and torque computation that were previously duplicated
across compute_torques.py, batch_sim_solo.py, and simulate_torques.py.

Usage:
    from prepare2.pd_utils import (
        build_model, setup_pd_gains, compute_pd_torques_np,
        BODY_GAINS, DOFS_PER_PERSON, COORDS_PER_PERSON,
        DEFAULT_SIM_FREQ, DEFAULT_FPS,
    )
"""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import newton

from prepare2.retarget import get_or_create_xml, SMPL_TO_NEWTON, N_SMPL_JOINTS

# Lazy import — newton.sensors may not exist in older builds
try:
    from newton import Contacts
    from newton.sensors import SensorContact
    _HAS_SENSORS = True
except ImportError:
    _HAS_SENSORS = False


# ═══════════════════════════════════════════════════════════════
# Constants (single source of truth)
# ═══════════════════════════════════════════════════════════════

# DOFs per person: 6 root (3 pos + 3 rot) + 23*3 hinges = 75
DOFS_PER_PERSON = 75
# Joint coordinates per person: 3 pos + 4 quat + 23*3 hinges = 76
COORDS_PER_PERSON = 76
# Number of bodies per person (Pelvis + 23 joints)
BODIES_PER_PERSON = 24
# Number of non-root bodies
N_JOINT_BODIES = 23

DEFAULT_SIM_FREQ = 480    # Hz physics
DEFAULT_FPS = 30          # motion playback rate
DEFAULT_TORQUE_LIMIT = 1000.0  # Nm clamp

# Armature values (regularization for thin SMPL-X limbs)
ARMATURE_HINGE = 0.5
ARMATURE_ROOT = 5.0

# Root PD gains
ROOT_POS_KP = 2000.0   # N/m
ROOT_POS_KD = 400.0
ROOT_ROT_KP = 1000.0   # Nm/rad
ROOT_ROT_KD = 200.0

# Per-body PD gains (kp Nm/rad, kd Nms/rad)
# Tuned for SMPL-X model with 3-hinge compound joints.
# Higher for large bodies (torso/hips), lower for extremities.
BODY_GAINS = {
    "L_Hip": (300, 30),    "L_Knee": (300, 30),    "L_Ankle": (200, 20),
    "L_Toe": (100, 10),    "R_Hip": (300, 30),      "R_Knee": (300, 30),
    "R_Ankle": (200, 20),  "R_Toe": (100, 10),
    "Torso": (500, 50),    "Spine": (500, 50),      "Chest": (500, 50),
    "Neck": (200, 20),     "Head": (100, 10),
    "L_Thorax": (200, 20), "L_Shoulder": (200, 20), "L_Elbow": (150, 15),
    "L_Wrist": (100, 10),  "L_Hand": (50, 5),
    "R_Thorax": (200, 20), "R_Shoulder": (200, 20), "R_Elbow": (150, 15),
    "R_Wrist": (100, 10),  "R_Hand": (50, 5),
}

# Ordered body names (root + 23 joints)
BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# DOF names: 6 root + 23*3 hinges
DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]
for _body in BODY_NAMES[1:]:
    DOF_NAMES.extend([f"{_body}_x", f"{_body}_y", f"{_body}_z"])


# ═══════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════

def build_model(betas_list, device="cuda:0", with_ground=True,
                enable_self_collisions=False):
    """
    Build a Newton model for one or more persons.

    Args:
        betas_list: list of betas arrays (one per person), or a single
                    betas array for one person.
        device: compute device string
        with_ground: add ground plane (needed for MuJoCo solver)
        enable_self_collisions: enable self-collision detection

    Returns:
        model: finalized Newton Model
        xml_paths: list of XML paths used (for reference)
    """
    if isinstance(betas_list, np.ndarray) and betas_list.ndim == 1:
        betas_list = [betas_list]

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    xml_paths = []
    for betas in betas_list:
        xml_path = get_or_create_xml(betas)
        builder.add_mjcf(xml_path, enable_self_collisions=enable_self_collisions)
        xml_paths.append(xml_path)

    if with_ground:
        builder.add_ground_plane()

    model = builder.finalize(device=device)
    return model, xml_paths


def setup_model_properties(model, n_persons, device="cuda:0"):
    """
    Configure model properties shared across all PD simulations:
    - Disable passive springs (we use explicit PD)
    - Set armature values for stability
    """
    n_dof = model.joint_dof_count

    # Disable passive springs — we apply explicit PD
    model.mujoco.dof_passive_stiffness.fill_(0.0)
    model.mujoco.dof_passive_damping.fill_(0.0)
    model.joint_target_ke.fill_(0.0)
    model.joint_target_kd.fill_(0.0)

    # Armature for stability
    arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
    for i in range(n_persons):
        off = i * DOFS_PER_PERSON
        arm[off:off + 6] = ARMATURE_ROOT
    model.joint_armature = wp.array(arm, dtype=wp.float32, device=device)


def create_mujoco_solver(model, n_persons):
    """
    Create a SolverMuJoCo with tuned parameters for PD tracking.

    Args:
        model: Newton Model
        n_persons: number of persons in the model

    Returns:
        solver: SolverMuJoCo instance
    """
    return newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=450 * n_persons, nconmax=150 * n_persons,
        impratio=10, iterations=100, ls_iterations=50,
    )


# ═══════════════════════════════════════════════════════════════
# PD gain setup
# ═══════════════════════════════════════════════════════════════

def build_pd_gains(model, n_persons, gain_scale=1.0):
    """
    Build per-DOF PD gain arrays for one or more persons.

    Args:
        model: Newton Model (used to read body names via body_label)
        n_persons: number of persons in model
        gain_scale: multiplier for all gains (>1 = stiffer)

    Returns:
        kp: (n_dof,) proportional gains
        kd: (n_dof,) derivative gains
    """
    n_dof = model.joint_dof_count
    kp = np.zeros(n_dof, dtype=np.float32)
    kd = np.zeros(n_dof, dtype=np.float32)

    for i in range(n_persons):
        d = i * DOFS_PER_PERSON

        # Root virtual forces
        kp[d:d + 3] = ROOT_POS_KP * gain_scale
        kd[d:d + 3] = ROOT_POS_KD * gain_scale
        kp[d + 3:d + 6] = ROOT_ROT_KP * gain_scale
        kd[d + 3:d + 6] = ROOT_ROT_KD * gain_scale

        # Per-body hinge gains (23 bodies × 3 DOF each)
        for b_idx in range(N_JOINT_BODIES):
            s = d + 6 + b_idx * 3
            body_name = model.body_label[i * BODIES_PER_PERSON + 1 + b_idx].rsplit('/', 1)[-1]
            k, kd_val = BODY_GAINS.get(body_name, (100, 10))
            kp[s:s + 3] = k * gain_scale
            kd[s:s + 3] = kd_val * gain_scale

    return kp, kd


# ═══════════════════════════════════════════════════════════════
# PD torque computation (CPU / numpy)
# ═══════════════════════════════════════════════════════════════

def compute_pd_torques_np(cq, cqd, ref_q, kp, kd, person_idx=0,
                          pd_scale=1.0, torque_limit=DEFAULT_TORQUE_LIMIT,
                          tau=None):
    """
    Compute PD torques for one person using numpy (CPU).

    Computes: τ = kp * (q_ref - q) - kd * q̇
    with quaternion error for root orientation.

    Args:
        cq: current joint coordinates (full multi-person array)
        cqd: current joint velocities (full multi-person array)
        ref_q: reference joint_q for this person (76,)
        kp: proportional gains (full multi-person array)
        kd: derivative gains (full multi-person array)
        person_idx: which person (0-indexed)
        pd_scale: gain multiplier (1.0 = full PD, 0.1 = correction)
        torque_limit: max torque magnitude (Nm)
        tau: pre-allocated output array to write into (optional);
             if None, a new array is created. If provided, torques
             are ADDED to the existing values (useful for I.D. + PD).

    Returns:
        tau: torque array (same size as kp)
    """
    d = person_idx * DOFS_PER_PERSON
    c = person_idx * COORDS_PER_PERSON

    if tau is None:
        tau = np.zeros_like(kp)

    # Root position PD
    tau[d:d + 3] += (
        kp[d:d + 3] * pd_scale * (ref_q[:3] - cq[c:c + 3])
        - kd[d:d + 3] * pd_scale * cqd[d:d + 3]
    )

    # Root orientation PD (quaternion → axis-angle error)
    q_cur = cq[c + 3:c + 7].copy()
    qn = np.linalg.norm(q_cur)
    if qn > 1e-8:
        q_cur /= qn
    R_err = (
        Rotation.from_quat(ref_q[3:7])
        * Rotation.from_quat(q_cur).inv()
    ).as_rotvec()
    tau[d + 3:d + 6] += (
        kp[d + 3:d + 6] * pd_scale * R_err
        - kd[d + 3:d + 6] * pd_scale * cqd[d + 3:d + 6]
    )

    # Hinge PD (23 bodies × 3 DOF = 69 DOF)
    tau[d + 6:d + DOFS_PER_PERSON] += (
        kp[d + 6:d + DOFS_PER_PERSON] * pd_scale
        * (ref_q[7:] - cq[c + 7:c + COORDS_PER_PERSON])
        - kd[d + 6:d + DOFS_PER_PERSON] * pd_scale
        * cqd[d + 6:d + DOFS_PER_PERSON]
    )

    return tau


def _root_pd_np(cq, cqd, ref_q, kp, kd, person_idx, pd_scale, tau):
    """Add PD torques for root DOFs only (position + orientation)."""
    d = person_idx * DOFS_PER_PERSON
    c = person_idx * COORDS_PER_PERSON

    # Root position PD
    tau[d:d + 3] += (
        kp[d:d + 3] * pd_scale * (ref_q[:3] - cq[c:c + 3])
        - kd[d:d + 3] * pd_scale * cqd[d:d + 3]
    )

    # Root orientation PD (quaternion → axis-angle error)
    q_cur = cq[c + 3:c + 7].copy()
    qn = np.linalg.norm(q_cur)
    if qn > 1e-8:
        q_cur /= qn
    R_err = (
        Rotation.from_quat(ref_q[3:7])
        * Rotation.from_quat(q_cur).inv()
    ).as_rotvec()
    tau[d + 3:d + 6] += (
        kp[d + 3:d + 6] * pd_scale * R_err
        - kd[d + 3:d + 6] * pd_scale * cqd[d + 3:d + 6]
    )


def _hinge_pd_np(cq, cqd, ref_q, kp, kd, person_idx, pd_scale,
                 torque_limit, tau):
    """Add PD torques for hinge (joint) DOFs only."""
    d = person_idx * DOFS_PER_PERSON
    c = person_idx * COORDS_PER_PERSON

    tau[d + 6:d + DOFS_PER_PERSON] += (
        kp[d + 6:d + DOFS_PER_PERSON] * pd_scale
        * (ref_q[7:] - cq[c + 7:c + COORDS_PER_PERSON])
        - kd[d + 6:d + DOFS_PER_PERSON] * pd_scale
        * cqd[d + 6:d + DOFS_PER_PERSON]
    )


def compute_all_pd_torques_np(cq, cqd, all_ref_jq, frame_idx, kp, kd,
                              n_persons, pd_scale=1.0,
                              torque_limit=DEFAULT_TORQUE_LIMIT,
                              precomputed_torques=None,
                              precomputed_pd_scale=0.1):
    """
    Compute PD torques for all persons in one call.

    Optionally starts from precomputed torques (inverse dynamics or
    optimized) with a reduced PD correction on top.

    Args:
        cq: current joint coordinates (full state)
        cqd: current joint velocities (full state)
        all_ref_jq: list of (T, 76) reference trajectories per person
        frame_idx: current frame index
        kp: proportional gains array
        kd: derivative gains array
        n_persons: number of persons
        pd_scale: gain multiplier for PD (1.0 = full)
        torque_limit: max torque (Nm)
        precomputed_torques: optional list of (T, 75) precomputed torques
                            per person (None entries = skip that person)
        precomputed_pd_scale: PD scale when using precomputed torques
                             (default 0.1 = 10% correction)

    Returns:
        tau: clipped torque array (n_dof,)
    """
    tau = np.zeros_like(kp)

    for i in range(n_persons):
        ref_q = all_ref_jq[i][min(frame_idx, all_ref_jq[i].shape[0] - 1)]
        d = i * DOFS_PER_PERSON

        # Start with precomputed torques if available
        has_precomputed = (
            precomputed_torques is not None
            and i < len(precomputed_torques)
            and precomputed_torques[i] is not None
        )
        if has_precomputed:
            pc = precomputed_torques[i]
            t = min(frame_idx, pc.shape[0] - 1)
            # Only apply JOINT torques (DOFs 6:) from precomputed I.D.
            # Root DOFs (0:6) are virtual forces from I.D. — they are
            # NOT real actuators and would launch the character if applied.
            # Root tracking is always handled by full-strength PD below.
            tau[d + 6:d + DOFS_PER_PERSON] = pc[t, 6:]

            # Root PD at full strength (I.D. root forces are excluded,
            # so PD alone must hold the pelvis at the reference position).
            # Joint PD at reduced scale (I.D. provides the main torque,
            # PD only corrects small tracking errors).
            _root_pd_np(cq, cqd, ref_q, kp, kd, i, pd_scale, tau)
            _hinge_pd_np(cq, cqd, ref_q, kp, kd, i,
                         precomputed_pd_scale, torque_limit, tau)
        else:
            scale = pd_scale

            compute_pd_torques_np(
                cq, cqd, ref_q, kp, kd,
                person_idx=i, pd_scale=scale,
                torque_limit=torque_limit, tau=tau,
            )

    return np.clip(tau, -torque_limit, torque_limit)


# ═══════════════════════════════════════════════════════════════
# PD torque computation (GPU / Warp kernels)
# ═══════════════════════════════════════════════════════════════

@wp.kernel
def pd_torque_kernel(
    joint_q: wp.array(dtype=wp.float32),     # current state q (76)
    joint_qd: wp.array(dtype=wp.float32),    # current state qd (75)
    ref_q: wp.array(dtype=wp.float32),       # reference q (76)
    kp: wp.array(dtype=wp.float32),          # P gains (75)
    kd: wp.array(dtype=wp.float32),          # D gains (75)
    torque_limit: float,
    tau_out: wp.array(dtype=wp.float32),     # output torques (75)
):
    """Compute PD torques entirely on GPU. 75 threads, one per DOF."""
    tid = wp.tid()

    if tid < 3:
        # Root position PD: tau = kp*(ref - cur) - kd*vel
        tau = kp[tid] * (ref_q[tid] - joint_q[tid]) - kd[tid] * joint_qd[tid]
    elif tid < 6:
        # Root orientation PD: quaternion error → axis-angle
        cur_qx = joint_q[3]
        cur_qy = joint_q[4]
        cur_qz = joint_q[5]
        cur_qw = joint_q[6]
        # Normalize
        qn = wp.sqrt(cur_qx * cur_qx + cur_qy * cur_qy
                      + cur_qz * cur_qz + cur_qw * cur_qw)
        if qn > 1.0e-8:
            cur_qx = cur_qx / qn
            cur_qy = cur_qy / qn
            cur_qz = cur_qz / qn
            cur_qw = cur_qw / qn

        q_cur = wp.quat(cur_qx, cur_qy, cur_qz, cur_qw)
        q_ref = wp.quat(ref_q[3], ref_q[4], ref_q[5], ref_q[6])

        # Error quaternion: q_err = q_ref * q_cur^-1
        q_err = q_ref * wp.quat_inverse(q_cur)
        axis = wp.vec3(0.0, 0.0, 0.0)
        angle = float(0.0)
        wp.quat_to_axis_angle(q_err, axis, angle)
        rotvec = axis * angle

        r_idx = tid - 3
        r_val = float(0.0)
        if r_idx == 0:
            r_val = rotvec[0]
        elif r_idx == 1:
            r_val = rotvec[1]
        else:
            r_val = rotvec[2]

        tau = kp[tid] * r_val - kd[tid] * joint_qd[tid]
    else:
        # Hinge PD: tau = kp*(ref_hinge - cur_hinge) - kd*vel
        q_idx = tid + 1
        tau = kp[tid] * (ref_q[q_idx] - joint_q[q_idx]) - kd[tid] * joint_qd[tid]

    tau_out[tid] = wp.clamp(tau, -torque_limit, torque_limit)


@wp.kernel
def accumulate_torque_kernel(
    tau_substep: wp.array(dtype=wp.float32),
    tau_accum: wp.array(dtype=wp.float32),
):
    """Add substep torques into accumulator."""
    tid = wp.tid()
    tau_accum[tid] = tau_accum[tid] + tau_substep[tid]


@wp.kernel
def scale_torque_kernel(
    tau_accum: wp.array(dtype=wp.float32),
    scale: float,
    tau_out: wp.array(dtype=wp.float32),
):
    """Scale accumulated torques (divide by sim_steps) and store."""
    tid = wp.tid()
    tau_out[tid] = tau_accum[tid] * scale


@wp.kernel
def zero_kernel(
    arr: wp.array(dtype=wp.float32),
):
    """Zero out an array."""
    tid = wp.tid()
    arr[tid] = float(0.0)


# ═══════════════════════════════════════════════════════════════
# Position extraction
# ═══════════════════════════════════════════════════════════════

def extract_positions_from_state(state, n_persons):
    """
    Extract 22 SMPL joint positions per person from Newton state.

    Args:
        state: Newton state with body_q
        n_persons: number of persons in the model

    Returns:
        positions: (n_persons, 22, 3) float32
    """
    body_q = state.body_q.numpy().reshape(-1, 7)  # (n_bodies, 7)
    positions = np.zeros((n_persons, N_SMPL_JOINTS, 3), dtype=np.float32)
    for p in range(n_persons):
        body_offset = p * BODIES_PER_PERSON
        for j in range(N_SMPL_JOINTS):
            positions[p, j] = body_q[body_offset + SMPL_TO_NEWTON[j], :3]
    return positions


# ═══════════════════════════════════════════════════════════════
# Simulation helpers
# ═══════════════════════════════════════════════════════════════

def init_state(model, all_ref_jq, n_persons, device="cuda:0"):
    """
    Initialize simulation state from first frame of reference trajectories.

    Args:
        model: Newton Model
        all_ref_jq: list of (T, 76) arrays per person
        n_persons: number of persons
        device: compute device

    Returns:
        state_0, state_1, control: Newton states and control
    """
    n_coords = model.joint_coord_count
    n_dof = model.joint_dof_count

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    init_q = np.zeros(n_coords, dtype=np.float32)
    for i, jq in enumerate(all_ref_jq):
        c = i * COORDS_PER_PERSON
        init_q[c:c + COORDS_PER_PERSON] = jq[0]

    state_0.joint_q = wp.array(init_q, dtype=wp.float32, device=device)
    state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    return state_0, state_1, control


def downsample_trajectory(jq, factor):
    """Downsample a joint_q trajectory by the given factor."""
    if factor > 1:
        return jq[::factor]
    return jq


# ═══════════════════════════════════════════════════════════════
# Contact sensors (requires SolverMuJoCo + newton.sensors)
# ═══════════════════════════════════════════════════════════════

# Body names that correspond to feet / hands for sensor targeting.
# These use wildcard patterns matched against Newton shape labels.
# Note: *Ankle* matches both Ankle and Toe shapes because Toe is
# nested under the Ankle path (e.g. ".../L_Ankle/L_Toe/L_Toe_geom_0")
FOOT_SHAPE_PATTERN = "*Ankle*"       # L_Ankle, R_Ankle, L_Toe, R_Toe
HAND_SHAPE_PATTERN = "*Hand*"        # L_Hand, R_Hand (+ geom children)
GROUND_SHAPE_PATTERN = "*ground*"    # ground_plane


def create_contact_sensors(model, solver, n_persons=1, verbose=True):
    """
    Create contact sensors for feet (ground reaction) and hands
    (inter-person touch detection).

    SensorContact requires SolverMuJoCo because it needs
    solver.update_contacts() to populate the Contacts force data.

    Args:
        model:  Newton Model (must include ground plane)
        solver: SolverMuJoCo instance
        n_persons: number of persons in model
        verbose: print sensor shape info

    Returns:
        dict with keys:
            'contacts':    Contacts buffer (pre-allocated)
            'foot_sensor': SensorContact for feet vs ground
            'hand_sensor': SensorContact for hands vs all (or None for 1-person)
        Returns None if newton.sensors is not available.
    """
    if not _HAS_SENSORS:
        if verbose:
            print("WARNING: newton.sensors not available — skipping contact sensors")
        return None

    if not isinstance(solver, newton.solvers.SolverMuJoCo):
        if verbose:
            print("WARNING: contact sensors require SolverMuJoCo — skipping")
        return None

    # Create SensorContact objects FIRST — their __init__ calls
    # model.request_contact_attributes("force"), which must happen
    # before the Contacts buffer is allocated so that force arrays
    # are included in the Contacts object.

    # Foot → ground sensor (ground reaction forces)
    foot_sensor = SensorContact(
        model,
        sensing_obj_shapes=FOOT_SHAPE_PATTERN,
        counterpart_shapes=[GROUND_SHAPE_PATTERN],
        include_total=True,
        verbose=verbose,
    )

    # Hand → any contact sensor (inter-person touch detection)
    # For 2-person models, this detects when hands touch the other body.
    # For 1-person, hands rarely contact anything useful, so skip.
    hand_sensor = None
    if n_persons >= 2:
        hand_sensor = SensorContact(
            model,
            sensing_obj_shapes=HAND_SHAPE_PATTERN,
            include_total=True,
            verbose=verbose,
        )

    # Now allocate Contacts buffer — model.get_requested_contact_attributes()
    # will include "force" because the SensorContact(s) above registered it.
    contacts = Contacts(
        solver.get_max_contact_count(),
        0,
        requested_attributes=model.get_requested_contact_attributes(),
    )

    if verbose:
        print(f"Contact sensors created: feet={len(foot_sensor.sensing_objs)} groups"
              + (f", hands={len(hand_sensor.sensing_objs)} groups" if hand_sensor else ""))

    return {
        'contacts': contacts,
        'foot_sensor': foot_sensor,
        'hand_sensor': hand_sensor,
    }


def update_contact_sensors(solver, state, sensor_dict):
    """
    Update contact sensor readings after a solver.step().

    Call this once per control frame (not per substep — the last substep's
    contacts are the ones that matter for force readout).

    Args:
        solver: SolverMuJoCo instance
        state: current state (after solver.step)
        sensor_dict: dict returned by create_contact_sensors()

    Returns:
        dict with numpy arrays:
            'foot_forces':  (n_foot_groups, 3) net force per foot group
            'hand_forces':  (n_hand_groups, 3) net force per hand group, or None
    """
    if sensor_dict is None:
        return None

    contacts = sensor_dict['contacts']
    foot_sensor = sensor_dict['foot_sensor']
    hand_sensor = sensor_dict['hand_sensor']

    # Populate Contacts with MuJoCo force data
    solver.update_contacts(contacts, state)

    # Update foot sensor
    foot_sensor.update(contacts)
    foot_forces = foot_sensor.net_force.numpy()  # (n_groups, n_counterparts, 3)

    result = {'foot_forces': foot_forces}

    # Update hand sensor
    if hand_sensor is not None:
        hand_sensor.update(contacts)
        hand_forces = hand_sensor.net_force.numpy()
        result['hand_forces'] = hand_forces
    else:
        result['hand_forces'] = None

    return result
