"""
interaction_metrics.py — Physical plausibility metrics for two-person interactions.

Compares paired (both persons) vs solo (one person alone) torques to
quantify physical consistency of interaction motions.

Metrics:
  TD   — Torque Delta: |τ_paired - τ_solo| per DOF per frame
  RFD  — Root Force Delta: interaction force proxy at root
  N3LV — Newton's 3rd Law Violation: consistency of interaction forces
  SII  — Solo Impossibility Index: fraction of impossible-without-partner frames
  BPS  — Biomechanical Plausibility Score: torques exceeding human limits
  CTC  — Contact-Torque Correlation: proximity vs torque delta

Usage:
    from prepare4.interaction_metrics import compute_all_metrics

    result = compute_paired_vs_solo(...)
    metrics = compute_all_metrics(result, positions_A, positions_B)
"""
import numpy as np
from scipy.stats import pearsonr

# ─── Constants ───────────────────────────────────────────────────

GRAVITY = 9.81
DEFAULT_BODY_MASS = 75.0  # kg

# Body group DOF slices within the 75-DOF torque vector
# DOFs 0:6 = root (3 trans + 3 rot), 6:75 = 23 hinge bodies × 3 DOF
BODY_GROUPS = {
    "Root":        slice(0, 6),
    "L_Leg":       slice(6, 18),    # L_Hip(3) + L_Knee(3) + L_Ankle(3) + L_Toe(3)
    "R_Leg":       slice(18, 30),   # R_Hip(3) + R_Knee(3) + R_Ankle(3) + R_Toe(3)
    "Spine/Torso": slice(30, 45),   # Torso(3) + Spine(3) + Chest(3) + Neck(3) + Head(3)
    "L_Arm":       slice(45, 60),   # L_Thorax(3) + L_Shoulder(3) + L_Elbow(3) + L_Wrist(3) + L_Hand(3)
    "R_Arm":       slice(60, 75),   # R_Thorax(3) + R_Shoulder(3) + R_Elbow(3) + R_Wrist(3) + R_Hand(3)
}

HINGE_SLICE = slice(6, 75)  # all non-root DOFs

# Joint names in DOF order (23 bodies × 3 DOF each, starting at DOF 6)
JOINT_NAMES_23 = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# Biomechanical torque limits (Nm) — peak isometric for ~75 kg adult male
# Sources: De Leva 1996, Anderson et al. 2007, Jiang et al. SIGGRAPH 2019
# Each joint has 3 DOFs; we use the max across flexion/extension/abduction.
BIOMECH_LIMITS = {
    "L_Hip": 250.0,    "R_Hip": 250.0,
    "L_Knee": 260.0,   "R_Knee": 260.0,
    "L_Ankle": 140.0,  "R_Ankle": 140.0,
    "L_Toe": 50.0,     "R_Toe": 50.0,
    "Torso": 300.0,    "Spine": 300.0,    "Chest": 300.0,
    "Neck": 50.0,      "Head": 30.0,
    "L_Thorax": 100.0, "L_Shoulder": 77.0,
    "L_Elbow": 80.0,   "L_Wrist": 18.0,  "L_Hand": 10.0,
    "R_Thorax": 100.0, "R_Shoulder": 77.0,
    "R_Elbow": 80.0,   "R_Wrist": 18.0,  "R_Hand": 10.0,
}


# ─── Individual metric functions ─────────────────────────────────

def torque_delta(tau_paired, tau_solo):
    """Per-DOF per-frame absolute torque difference.

    Args:
        tau_paired: (T, 75) torques from paired simulation
        tau_solo:   (T, 75) torques from solo simulation

    Returns:
        td: (T, 75) |τ_paired - τ_solo|
    """
    return np.abs(tau_paired - tau_solo)


def root_force_delta(tau_paired, tau_solo):
    """Root DOF force difference — proxy for interaction forces.

    DOFs 0:3 = translational (Newtons), 3:6 = rotational (Nm).

    Returns:
        rfd_trans: (T,) translational root force delta magnitude
        rfd_rot:   (T,) rotational root torque delta magnitude
        rfd_vec:   (T, 6) signed delta vector
    """
    delta = tau_paired[:, :6] - tau_solo[:, :6]
    rfd_trans = np.linalg.norm(delta[:, :3], axis=-1)
    rfd_rot = np.linalg.norm(delta[:, 3:6], axis=-1)
    return rfd_trans, rfd_rot, delta


def newtons_third_law_violation(tau_paired_A, tau_solo_A,
                                 tau_paired_B, tau_solo_B):
    """Newton's 3rd law violation from root force deltas.

    The interaction force proxy for person A is:
        F_int_A = root_force_paired_A - root_force_solo_A
    For Newton's 3rd law: F_int_A + F_int_B ≈ 0.

    Returns:
        n3lv: (T,) normalized violation [0=perfect, 1=total violation]
        f_int_A: (T, 3) interaction force proxy for A (translational)
        f_int_B: (T, 3) interaction force proxy for B (translational)
    """
    f_int_A = tau_paired_A[:, :3] - tau_solo_A[:, :3]
    f_int_B = tau_paired_B[:, :3] - tau_solo_B[:, :3]

    violation = np.linalg.norm(f_int_A + f_int_B, axis=-1)
    normalizer = np.linalg.norm(f_int_A, axis=-1) + \
                 np.linalg.norm(f_int_B, axis=-1) + 1e-6

    n3lv = violation / normalizer
    return n3lv, f_int_A, f_int_B


def solo_impossibility_index(tau_solo, mass=DEFAULT_BODY_MASS,
                              threshold_factor=2.0):
    """Fraction of frames where solo root forces exceed plausible limits.

    A root translational force >> m*g indicates the person cannot
    maintain the motion without external support (from their partner
    or a "skyhook").

    Args:
        tau_solo: (T, 75) solo torques
        mass: body mass in kg
        threshold_factor: multiplier on body weight (default 2.0 = 2×mg)

    Returns:
        sii: scalar in [0, 1], fraction of impossible frames
        root_force_mag: (T,) root translational force magnitude
        threshold: the force threshold used
    """
    root_force = tau_solo[:, :3]
    root_force_mag = np.linalg.norm(root_force, axis=-1)
    threshold = mass * GRAVITY * threshold_factor

    impossible_frames = root_force_mag > threshold
    sii = impossible_frames.mean()

    return sii, root_force_mag, threshold


def biomechanical_plausibility_score(tau, limits=None):
    """Fraction of frames where joint torques exceed biomechanical limits.

    Args:
        tau: (T, 75) torques
        limits: dict of joint_name → max torque (Nm). Default: BIOMECH_LIMITS.

    Returns:
        bps_per_joint: dict of joint_name → fraction of violating frames
        bps_overall: scalar, fraction of (frame, joint) pairs violating limits
        violation_mask: (T, 23) bool, True where any DOF of that joint exceeds limit
    """
    if limits is None:
        limits = BIOMECH_LIMITS

    T = tau.shape[0]
    violation_mask = np.zeros((T, len(JOINT_NAMES_23)), dtype=bool)
    bps_per_joint = {}

    for j_idx, jname in enumerate(JOINT_NAMES_23):
        dof_start = 6 + j_idx * 3
        dof_end = dof_start + 3
        joint_tau = tau[:, dof_start:dof_end]
        max_tau = np.abs(joint_tau).max(axis=-1)  # (T,)

        limit = limits.get(jname, 1000.0)
        violated = max_tau > limit
        violation_mask[:, j_idx] = violated
        bps_per_joint[jname] = violated.mean()

    bps_overall = violation_mask.mean()
    return bps_per_joint, bps_overall, violation_mask


def contact_torque_correlation(torque_delta_mag, positions_A, positions_B):
    """Correlation between torque delta and inter-person proximity.

    For physically plausible interactions, torques should change more
    when the persons are close (i.e., contacting).

    Args:
        torque_delta_mag: (T,) total torque delta magnitude per frame
        positions_A: (T, J, 3) joint positions for person A
        positions_B: (T, J, 3) joint positions for person B

    Returns:
        ctc: Pearson correlation coefficient
        p_value: statistical significance
        min_dist: (T,) minimum inter-person distance per frame
    """
    T = min(torque_delta_mag.shape[0], positions_A.shape[0],
            positions_B.shape[0])
    td = torque_delta_mag[:T]
    pos_A = positions_A[:T]
    pos_B = positions_B[:T]

    # Compute minimum distance between any pair of joints
    min_dist = np.zeros(T, dtype=np.float32)
    for t in range(T):
        # (J_A, J_B) distance matrix
        diff = pos_A[t, :, None, :] - pos_B[t, None, :, :]  # (J, J, 3)
        dists = np.linalg.norm(diff, axis=-1)  # (J, J)
        min_dist[t] = dists.min()

    # Proximity = inverse distance (clamped to avoid inf)
    proximity = 1.0 / (min_dist + 0.01)

    # Pearson correlation between proximity and torque delta
    if np.std(td) < 1e-8 or np.std(proximity) < 1e-8:
        return 0.0, 1.0, min_dist

    ctc, p_value = pearsonr(proximity, td)
    return ctc, p_value, min_dist


# ─── Aggregate metric computation ────────────────────────────────

def compute_all_metrics(result, positions_A=None, positions_B=None,
                        mass=DEFAULT_BODY_MASS):
    """Compute all interaction metrics from paired-vs-solo result dict.

    Args:
        result: dict from compute_paired_vs_solo()
        positions_A: (T, J, 3) joint positions for person A (optional, for CTC)
        positions_B: (T, J, 3) joint positions for person B (optional, for CTC)
        mass: body mass in kg

    Returns:
        metrics: dict with all computed metrics
    """
    tau_p_A = result['torques_paired_A']
    tau_p_B = result['torques_paired_B']
    tau_s_A = result['torques_solo_A']
    tau_s_B = result['torques_solo_B']

    T = tau_p_A.shape[0]
    metrics = {'n_frames': T}

    # --- Torque Delta ---
    td_A = torque_delta(tau_p_A, tau_s_A)
    td_B = torque_delta(tau_p_B, tau_s_B)
    metrics['torque_delta_A_mean'] = td_A[:, HINGE_SLICE].mean()
    metrics['torque_delta_B_mean'] = td_B[:, HINGE_SLICE].mean()
    metrics['torque_delta_A_max'] = td_A[:, HINGE_SLICE].max()
    metrics['torque_delta_B_max'] = td_B[:, HINGE_SLICE].max()

    # Per body-group torque delta
    for group_name, s in BODY_GROUPS.items():
        if group_name == "Root":
            continue
        metrics[f'td_A_{group_name}'] = td_A[:, s].mean()
        metrics[f'td_B_{group_name}'] = td_B[:, s].mean()

    # --- Root Force Delta ---
    rfd_trans_A, rfd_rot_A, _ = root_force_delta(tau_p_A, tau_s_A)
    rfd_trans_B, rfd_rot_B, _ = root_force_delta(tau_p_B, tau_s_B)
    metrics['root_force_delta_trans_A_mean'] = rfd_trans_A.mean()
    metrics['root_force_delta_trans_B_mean'] = rfd_trans_B.mean()
    metrics['root_force_delta_trans_A_p90'] = np.percentile(rfd_trans_A, 90)
    metrics['root_force_delta_trans_B_p90'] = np.percentile(rfd_trans_B, 90)

    # --- Newton's 3rd Law Violation ---
    n3lv, f_int_A, f_int_B = newtons_third_law_violation(
        tau_p_A, tau_s_A, tau_p_B, tau_s_B)
    metrics['n3lv_mean'] = n3lv.mean()
    metrics['n3lv_median'] = np.median(n3lv)
    metrics['n3lv_p90'] = np.percentile(n3lv, 90)
    metrics['f_int_A_mean_mag'] = np.linalg.norm(f_int_A, axis=-1).mean()
    metrics['f_int_B_mean_mag'] = np.linalg.norm(f_int_B, axis=-1).mean()

    # --- Solo Impossibility Index ---
    sii_A, rf_mag_A, thresh = solo_impossibility_index(tau_s_A, mass=mass)
    sii_B, rf_mag_B, _ = solo_impossibility_index(tau_s_B, mass=mass)
    metrics['sii_A'] = sii_A
    metrics['sii_B'] = sii_B
    metrics['sii_threshold_N'] = thresh
    metrics['solo_root_force_A_mean'] = rf_mag_A.mean()
    metrics['solo_root_force_B_mean'] = rf_mag_B.mean()
    metrics['solo_root_force_A_p90'] = np.percentile(rf_mag_A, 90)
    metrics['solo_root_force_B_p90'] = np.percentile(rf_mag_B, 90)

    # --- Biomechanical Plausibility Score ---
    bps_joints_pA, bps_overall_pA, _ = biomechanical_plausibility_score(tau_p_A)
    bps_joints_pB, bps_overall_pB, _ = biomechanical_plausibility_score(tau_p_B)
    bps_joints_sA, bps_overall_sA, _ = biomechanical_plausibility_score(tau_s_A)
    bps_joints_sB, bps_overall_sB, _ = biomechanical_plausibility_score(tau_s_B)
    metrics['bps_paired_A'] = bps_overall_pA
    metrics['bps_paired_B'] = bps_overall_pB
    metrics['bps_solo_A'] = bps_overall_sA
    metrics['bps_solo_B'] = bps_overall_sB
    metrics['bps_per_joint_paired_A'] = bps_joints_pA
    metrics['bps_per_joint_paired_B'] = bps_joints_pB
    metrics['bps_per_joint_solo_A'] = bps_joints_sA
    metrics['bps_per_joint_solo_B'] = bps_joints_sB

    # --- Contact-Torque Correlation ---
    if positions_A is not None and positions_B is not None:
        td_mag_A = np.linalg.norm(td_A[:, HINGE_SLICE], axis=-1)
        ctc, p_val, min_dist = contact_torque_correlation(
            td_mag_A, positions_A, positions_B)
        metrics['ctc'] = ctc
        metrics['ctc_p_value'] = p_val
        metrics['min_inter_distance_mean'] = min_dist.mean()
        metrics['min_inter_distance_min'] = min_dist.min()

    # --- Summary torque statistics ---
    for label, tau in [('paired_A', tau_p_A), ('paired_B', tau_p_B),
                       ('solo_A', tau_s_A), ('solo_B', tau_s_B)]:
        hinge = tau[:, HINGE_SLICE]
        metrics[f'tau_{label}_hinge_mean'] = np.abs(hinge).mean()
        metrics[f'tau_{label}_hinge_p90'] = np.percentile(np.abs(hinge), 90)
        metrics[f'tau_{label}_hinge_max'] = np.abs(hinge).max()
        metrics[f'tau_{label}_root_trans_mean'] = np.linalg.norm(
            tau[:, :3], axis=-1).mean()

    return metrics


def format_metrics_summary(metrics, clip_id="", text=""):
    """Format metrics dict into a human-readable summary string."""
    lines = []
    if clip_id:
        lines.append(f"Clip: {clip_id}")
    if text:
        lines.append(f"Text: {text}")
    lines.append(f"Frames: {metrics['n_frames']}")
    lines.append("")

    lines.append("=== Torque Delta (paired - solo) ===")
    lines.append(f"  Hinge mean: A={metrics['torque_delta_A_mean']:.2f} Nm  "
                 f"B={metrics['torque_delta_B_mean']:.2f} Nm")
    lines.append(f"  Hinge max:  A={metrics['torque_delta_A_max']:.1f} Nm  "
                 f"B={metrics['torque_delta_B_max']:.1f} Nm")
    for g in ["L_Leg", "R_Leg", "Spine/Torso", "L_Arm", "R_Arm"]:
        lines.append(f"  {g:14s}: A={metrics[f'td_A_{g}']:.2f}  "
                     f"B={metrics[f'td_B_{g}']:.2f} Nm")

    lines.append("")
    lines.append("=== Root Force Delta ===")
    lines.append(f"  Trans mean: A={metrics['root_force_delta_trans_A_mean']:.1f} N  "
                 f"B={metrics['root_force_delta_trans_B_mean']:.1f} N")
    lines.append(f"  Trans P90:  A={metrics['root_force_delta_trans_A_p90']:.1f} N  "
                 f"B={metrics['root_force_delta_trans_B_p90']:.1f} N")

    lines.append("")
    lines.append("=== Newton's 3rd Law Violation ===")
    lines.append(f"  N3LV mean={metrics['n3lv_mean']:.3f}  "
                 f"median={metrics['n3lv_median']:.3f}  "
                 f"P90={metrics['n3lv_p90']:.3f}")
    lines.append(f"  |F_int| mean: A={metrics['f_int_A_mean_mag']:.1f} N  "
                 f"B={metrics['f_int_B_mean_mag']:.1f} N")

    lines.append("")
    lines.append("=== Solo Impossibility Index ===")
    lines.append(f"  SII: A={metrics['sii_A']:.1%}  B={metrics['sii_B']:.1%}  "
                 f"(threshold={metrics['sii_threshold_N']:.0f} N)")
    lines.append(f"  Solo root force mean: A={metrics['solo_root_force_A_mean']:.0f} N  "
                 f"B={metrics['solo_root_force_B_mean']:.0f} N")

    lines.append("")
    lines.append("=== Biomechanical Plausibility ===")
    lines.append(f"  Violation rate (paired): A={metrics['bps_paired_A']:.1%}  "
                 f"B={metrics['bps_paired_B']:.1%}")
    lines.append(f"  Violation rate (solo):   A={metrics['bps_solo_A']:.1%}  "
                 f"B={metrics['bps_solo_B']:.1%}")

    if 'ctc' in metrics:
        lines.append("")
        lines.append("=== Contact-Torque Correlation ===")
        lines.append(f"  CTC={metrics['ctc']:.3f}  (p={metrics['ctc_p_value']:.4f})")
        lines.append(f"  Min inter-person dist: "
                     f"mean={metrics['min_inter_distance_mean']:.3f}m  "
                     f"min={metrics['min_inter_distance_min']:.3f}m")

    lines.append("")
    lines.append("=== Torque Statistics ===")
    for label in ['paired_A', 'paired_B', 'solo_A', 'solo_B']:
        lines.append(f"  {label:10s}: hinge_mean={metrics[f'tau_{label}_hinge_mean']:.1f} Nm  "
                     f"hinge_P90={metrics[f'tau_{label}_hinge_p90']:.1f} Nm  "
                     f"root_trans={metrics[f'tau_{label}_root_trans_mean']:.0f} N")

    return "\n".join(lines)
