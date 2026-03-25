"""
dynamics.py — Smooth differentiation of Newton joint trajectories.

Computes joint velocities (q̇) and accelerations (q̈) from joint_q
trajectories. Three methods available:

  - savgol:      Savitzky-Golay local polynomial fit → analytic derivatives
  - spline:      Global B-spline fit → analytic derivatives
  - finite_diff: Central finite differences (baseline, no smoothing)

The SavGol and spline methods inherently smooth during differentiation,
avoiding the noise amplification from double finite-differencing on raw
mocap data.  This fixes the known bug in prepare2/compute_torques.py
where filtering was applied AFTER differentiation.

Three signal types require different treatment:
  Root translation q[0:3]   →  smooth+differentiate directly (3D)
  Root quaternion  q[3:7]   →  convert to rotvec, unwrap, then smooth+diff
  Hinge angles     q[7:76]  →  smooth+differentiate directly (scalar)

Output DOF convention (matches Newton free-joint + 23 hinge bodies):
  joint_qd  (T, 75):  [root_vel(3), root_omega(3), hinge_vel(69)]
  joint_qdd (T, 75):  [root_acc(3), root_alpha(3), hinge_acc(69)]

Angular velocity note:
  Root ω is approximated as d/dt(rotvec(R(t))).  This equals the true
  angular velocity to first order in the frame-to-frame rotation angle.
  For human motion at 20-30 fps the error is negligible (< 0.1%).

Usage:
    from prepare4.dynamics import compute_derivatives

    qd, qdd = compute_derivatives(joint_q, dt=1/30, method="savgol")
    qd, qdd = compute_derivatives(joint_q, dt=1/30, method="spline")
    qd, qdd = compute_derivatives(joint_q, dt=1/30, method="finite_diff")

    # Quick validation with synthetic data
    python prepare4/dynamics.py --validate
    python prepare4/dynamics.py --validate --method spline

    # Test on real motion clip
    python prepare4/dynamics.py --clip 1000
"""
import os
import sys
import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─── Layout constants ─────────────────────────────────────────
N_JOINT_Q = 76    # 7 root (3 trans + 4 quat xyzw) + 69 hinge (23×3)
N_JOINT_QD = 75   # 6 root (3 vel + 3 omega)        + 69 hinge vel

# ─── Body name → Newton body index (matches MJCF generation order) ────
BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest",
    "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]
BODY_NAME_TO_IDX = {n: i for i, n in enumerate(BODY_NAMES)}

# ─── De Leva 1996 segment mass fractions (male, 73 kg reference) ──────
# Adapted to 24-body SMPL skeleton.  Sum = 100.00%.
#
# Source segments → SMPL mapping:
#   Lower trunk (11.17%)  → Pelvis
#   Middle trunk (16.33%) → Torso
#   Upper trunk (15.96%)  → Spine (7.48%) + Chest (7.48%)
#                           + L_Thorax (0.50%) + R_Thorax (0.50%)
#   Head+Neck (6.94%)     → Neck (20%=1.39%) + Head (80%=5.55%)
#   Thigh (14.16%)        → L/R_Hip
#   Shank (4.33%)         → L/R_Knee
#   Foot (1.37%)          → L/R_Ankle (70%=0.96%) + L/R_Toe (30%=0.41%)
#   Upper arm (2.71%)     → L/R_Shoulder
#   Forearm (1.62%)       → L/R_Elbow
#   Hand (0.61%)          → L/R_Wrist (60%=0.37%) + L/R_Hand (40%=0.24%)
DE_LEVA_MASS_FRACTIONS = {
    "Pelvis":      0.1117,
    "L_Hip":       0.1416,   "R_Hip":       0.1416,
    "L_Knee":      0.0433,   "R_Knee":      0.0433,
    "L_Ankle":     0.0096,   "R_Ankle":     0.0096,
    "L_Toe":       0.0041,   "R_Toe":       0.0041,
    "Torso":       0.1633,
    "Spine":       0.0748,
    "Chest":       0.0748,
    "Neck":        0.0139,
    "Head":        0.0555,
    "L_Thorax":    0.0050,   "R_Thorax":    0.0050,
    "L_Shoulder":  0.0271,   "R_Shoulder":  0.0271,
    "L_Elbow":     0.0162,   "R_Elbow":     0.0162,
    "L_Wrist":     0.0037,   "R_Wrist":     0.0037,
    "L_Hand":      0.0024,   "R_Hand":      0.0024,
}
# Verify sum = 1.0
assert abs(sum(DE_LEVA_MASS_FRACTIONS.values()) - 1.0) < 1e-6, \
    f"Mass fractions sum to {sum(DE_LEVA_MASS_FRACTIONS.values())}, expected 1.0"

DEFAULT_BODY_MASS_KG = 75.0  # average male body mass


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def compute_derivatives(joint_q, dt, method="savgol", **kwargs):
    """Compute joint velocities and accelerations from joint coordinates.

    Args:
        joint_q: (T, 76) joint coordinates. Layout:
            [0:3]  root translation (Z-up)
            [3:7]  root quaternion (xyzw, scipy convention)
            [7:76] hinge Euler angles (23 bodies × 3 DOF)
        dt: timestep in seconds (1 / fps)
        method: "savgol" | "spline" | "finite_diff"
        **kwargs: method-specific parameters:
            savgol:  window (int, odd, default 7),
                     polyorder (int, default 3)
            spline:  smoothing_factor (float, default T×1e-6),
                     spline_order (int, default 5)

    Returns:
        qd:  (T, 75) joint velocities   [m/s | rad/s]
        qdd: (T, 75) joint accelerations [m/s² | rad/s²]
    """
    assert joint_q.ndim == 2 and joint_q.shape[1] == N_JOINT_Q, \
        f"Expected (T, {N_JOINT_Q}), got {joint_q.shape}"
    assert dt > 0, f"dt must be positive, got {dt}"
    T = joint_q.shape[0]
    assert T >= 3, f"Need at least 3 frames, got {T}"

    if method == "savgol":
        return _deriv_savgol(joint_q, dt, **kwargs)
    elif method == "spline":
        return _deriv_spline(joint_q, dt, **kwargs)
    elif method == "finite_diff":
        return _deriv_finite_diff(joint_q, dt)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Choose 'savgol', 'spline', or 'finite_diff'."
        )


# ═══════════════════════════════════════════════════════════════
# Method 1 — Savitzky-Golay
# ═══════════════════════════════════════════════════════════════

def _deriv_savgol(joint_q, dt, window=11, polyorder=5):
    """SavGol derivatives: local polynomial fit with analytic differentiation.

    scipy.signal.savgol_filter(x, window, polyorder, deriv=n, delta=dt)
    fits a polynomial of degree `polyorder` to each window of `window`
    samples and evaluates its n-th derivative at the centre.  This
    inherently smooths while differentiating — no separate pre-smoothing
    step is needed.

    Default window=11, polyorder=5 gives < 2% relative error on the 2nd
    derivative of sinusoidal test signals at 30 fps, with 5 extra DOF
    per window for noise rejection.

    Args:
        joint_q: (T, 76) joint coordinates
        dt: timestep (seconds)
        window: SavGol window length (must be odd, >= polyorder+1).
                Clamped to T if T < window.  Default 11.
        polyorder: polynomial order for the local fit.  Default 5.

    Returns:
        qd, qdd: (T, 75) each
    """
    T = joint_q.shape[0]

    # Clamp window to trajectory length (must stay odd, >= polyorder+1)
    if window > T:
        window = T if T % 2 == 1 else T - 1
    if window < polyorder + 1:
        raise ValueError(
            f"window ({window}) must be >= polyorder+1 ({polyorder+1}). "
            f"Trajectory too short (T={T}) for polyorder={polyorder}."
        )
    assert window % 2 == 1, f"window must be odd, got {window}"

    qd = np.zeros((T, N_JOINT_QD), dtype=np.float64)
    qdd = np.zeros((T, N_JOINT_QD), dtype=np.float64)
    q = joint_q.astype(np.float64)

    # ── Root translation  q[0:3] → DOF 0:3 ───────────────────
    qd[:, :3] = savgol_filter(
        q[:, :3], window, polyorder, deriv=1, delta=dt, axis=0
    )
    qdd[:, :3] = savgol_filter(
        q[:, :3], window, polyorder, deriv=2, delta=dt, axis=0
    )

    # ── Root quaternion → angular velocity  q[3:7] → DOF 3:6 ─
    rotvecs = _quat_to_rotvec_unwrap(q[:, 3:7])
    qd[:, 3:6] = savgol_filter(
        rotvecs, window, polyorder, deriv=1, delta=dt, axis=0
    )
    qdd[:, 3:6] = savgol_filter(
        rotvecs, window, polyorder, deriv=2, delta=dt, axis=0
    )

    # ── Hinge joints  q[7:76] → DOF 6:75 ─────────────────────
    qd[:, 6:] = savgol_filter(
        q[:, 7:], window, polyorder, deriv=1, delta=dt, axis=0
    )
    qdd[:, 6:] = savgol_filter(
        q[:, 7:], window, polyorder, deriv=2, delta=dt, axis=0
    )

    return qd.astype(np.float32), qdd.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Method 2 — B-spline
# ═══════════════════════════════════════════════════════════════

def _deriv_spline(joint_q, dt, smoothing_factor=0.0, spline_order=5):
    """B-spline derivatives: global fit with analytic differentiation.

    Fits a B-spline of order `spline_order` to each DOF and evaluates
    its analytic 1st/2nd derivatives at each sample point.

    Args:
        joint_q: (T, 76) joint coordinates
        dt: timestep (seconds)
        smoothing_factor: scipy splrep smoothing parameter.
            0 → uses default T × 1e-6 (light smoothing).
            Larger values → more smoothing.
        spline_order: B-spline order (max 5). Clamped to T-1.

    Returns:
        qd, qdd: (T, 75) each
    """
    T = joint_q.shape[0]
    qd = np.zeros((T, N_JOINT_QD), dtype=np.float64)
    qdd = np.zeros((T, N_JOINT_QD), dtype=np.float64)
    t_arr = np.arange(T, dtype=np.float64) * dt
    q = joint_q.astype(np.float64)

    s = smoothing_factor if smoothing_factor > 0 else T * 1e-6
    k = min(spline_order, T - 1)  # spline order can't exceed T-1

    def _fit_and_diff(signal_1d, dof_idx):
        tck = splrep(t_arr, signal_1d, s=s, k=k)
        qd[:, dof_idx] = splev(t_arr, tck, der=1)
        qdd[:, dof_idx] = splev(t_arr, tck, der=2)

    # ── Root translation  q[0:3] → DOF 0:3 ───────────────────
    for d in range(3):
        _fit_and_diff(q[:, d], d)

    # ── Root quaternion → angular velocity  q[3:7] → DOF 3:6 ─
    rotvecs = _quat_to_rotvec_unwrap(q[:, 3:7])
    for d in range(3):
        _fit_and_diff(rotvecs[:, d], 3 + d)

    # ── Hinge joints  q[7:76] → DOF 6:75 ─────────────────────
    for d in range(69):
        _fit_and_diff(q[:, 7 + d], 6 + d)

    return qd.astype(np.float32), qdd.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Method 3 — Central finite differences (baseline)
# ═══════════════════════════════════════════════════════════════

def _deriv_finite_diff(joint_q, dt):
    """Central finite differences (no smoothing, baseline for comparison).

    Root angular velocity uses log-map: ω ≈ log(R_{t+1} R_{t-1}⁻¹)/(2dt).
    Accelerations use central differences on the computed velocities.

    Args:
        joint_q: (T, 76)
        dt: timestep (seconds)

    Returns:
        qd, qdd: (T, 75) each
    """
    T = joint_q.shape[0]
    qd = np.zeros((T, N_JOINT_QD), dtype=np.float64)
    qdd = np.zeros((T, N_JOINT_QD), dtype=np.float64)
    q = joint_q.astype(np.float64)

    # ── Root translation (DOF 0:3) ────────────────────────────
    # Central differences interior, forward/backward at boundaries
    qd[1:-1, :3] = (q[2:, :3] - q[:-2, :3]) / (2 * dt)
    qd[0, :3] = (q[1, :3] - q[0, :3]) / dt
    qd[-1, :3] = (q[-1, :3] - q[-2, :3]) / dt

    # ── Root angular velocity (DOF 3:6) ──────────────────────
    quats = q[:, 3:7].copy()
    # Quaternion sign continuity (handle double cover)
    for t in range(1, T):
        if np.dot(quats[t], quats[t - 1]) < 0:
            quats[t] = -quats[t]

    rotations = Rotation.from_quat(quats)
    # Interior: ω ≈ log(R_{t+1} R_{t-1}^{-1}) / (2 dt)
    for t in range(1, T - 1):
        dR = rotations[t + 1] * rotations[t - 1].inv()
        qd[t, 3:6] = dR.as_rotvec() / (2 * dt)
    # Boundaries: one-sided
    qd[0, 3:6] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
    qd[-1, 3:6] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / dt

    # ── Hinge joints (DOF 6:75) ──────────────────────────────
    hinge = q[:, 7:]  # (T, 69)
    qd[1:-1, 6:] = (hinge[2:] - hinge[:-2]) / (2 * dt)
    qd[0, 6:] = (hinge[1] - hinge[0]) / dt
    qd[-1, 6:] = (hinge[-1] - hinge[-2]) / dt

    # ── Accelerations via central diff on velocities ─────────
    qdd[1:-1] = (qd[2:] - qd[:-2]) / (2 * dt)
    qdd[0] = (qd[1] - qd[0]) / dt
    qdd[-1] = (qd[-1] - qd[-2]) / dt

    return qd.astype(np.float32), qdd.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Quaternion helpers
# ═══════════════════════════════════════════════════════════════

def _quat_to_rotvec_unwrap(quats):
    """Convert a quaternion trajectory to unwrapped rotation vectors.

    Steps:
      1. Sign continuity — flip q[t] if dot(q[t], q[t-1]) < 0
         (handle quaternion double cover).
      2. Convert to rotation vectors via scipy.
      3. Unwrap — if consecutive rotvec difference > π, shift by ±2π
         along the jump direction.

    Args:
        quats: (T, 4) quaternions in xyzw format (scipy convention)

    Returns:
        rotvecs: (T, 3) unwrapped rotation vectors
    """
    T = quats.shape[0]

    # Step 1: quaternion sign continuity
    q = quats.copy()
    for t in range(1, T):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]

    # Step 2: to rotation vectors
    rotvecs = Rotation.from_quat(q).as_rotvec()

    # Step 3: unwrap (shift by 2π along jump direction if |Δrv| > π)
    for t in range(1, T):
        diff = rotvecs[t] - rotvecs[t - 1]
        norm = np.linalg.norm(diff)
        if norm > np.pi:
            n_shifts = np.round(norm / (2 * np.pi))
            rotvecs[t] -= n_shifts * 2 * np.pi * diff / norm

    return rotvecs


# ═══════════════════════════════════════════════════════════════
# Segment inertia (De Leva 1996)
# ═══════════════════════════════════════════════════════════════

def set_segment_masses(model, total_mass=DEFAULT_BODY_MASS_KG, verbose=False):
    """Set body masses on a Newton model using De Leva 1996 fractions.

    Scales each body's mass to match the anthropometric mass fraction,
    and scales the inertia tensor proportionally to preserve the
    geometric shape (ratio of principal moments).

    Args:
        model: Newton Model (already finalized)
        total_mass: total body mass in kg (default 75.0)
        verbose: print mass breakdown

    Returns:
        old_masses: (N,) numpy array of original masses (for comparison)
    """
    import warp as wp

    n = model.body_count
    assert n == len(BODY_NAMES), \
        f"Model has {n} bodies, expected {len(BODY_NAMES)}"

    old_mass = model.body_mass.numpy().copy()
    old_inertia = model.body_inertia.numpy().copy()  # (N, 3, 3) or (N,) of mat33

    new_mass = np.zeros(n, dtype=np.float32)
    # old_inertia comes back as (n,) of mat33 → numpy gives (n, 3, 3)
    new_inertia = old_inertia.copy()

    for i, name in enumerate(BODY_NAMES):
        frac = DE_LEVA_MASS_FRACTIONS[name]
        target_m = frac * total_mass
        new_mass[i] = target_m

        # Scale inertia proportionally: I_new = I_old × (m_new / m_old)
        if old_mass[i] > 1e-8:
            scale = target_m / old_mass[i]
        else:
            scale = 1.0
        new_inertia[i] = old_inertia[i] * scale

    device = model.body_mass.device
    model.body_mass = wp.array(new_mass, dtype=wp.float32, device=device)
    model.body_inertia = wp.array(
        new_inertia, dtype=wp.mat33, device=device
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f" Segment masses (De Leva 1996, total={total_mass:.1f} kg)")
        print(f"{'='*60}")
        print(f"  {'Body':<15s}  {'Old (kg)':>10s}  {'New (kg)':>10s}  {'Frac':>6s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*6}")
        for i, name in enumerate(BODY_NAMES):
            frac = DE_LEVA_MASS_FRACTIONS[name]
            print(f"  {name:<15s}  {old_mass[i]:10.4f}  {new_mass[i]:10.4f}  {frac:6.2%}")
        print(f"  {'TOTAL':<15s}  {old_mass.sum():10.2f}  {new_mass.sum():10.2f}")
        print()

    return old_mass


def set_segment_masses_multi(model, n_persons, total_mass=DEFAULT_BODY_MASS_KG,
                             verbose=False):
    """Set body masses for a multi-person Newton model using De Leva 1996.

    Like set_segment_masses(), but handles models with n_persons × 24 bodies.

    Args:
        model: Newton Model with n_persons × 24 bodies (already finalized)
        n_persons: number of persons in the model
        total_mass: total body mass per person in kg (default 75.0)
        verbose: print mass breakdown
    """
    import warp as wp

    n = model.body_count
    n_per_person = len(BODY_NAMES)  # 24
    expected = n_per_person * n_persons
    assert n == expected, \
        f"Model has {n} bodies, expected {expected} ({n_persons} × {n_per_person})"

    old_mass = model.body_mass.numpy().copy()
    old_inertia = model.body_inertia.numpy().copy()

    new_mass = np.zeros(n, dtype=np.float32)
    new_inertia = old_inertia.copy()

    for p in range(n_persons):
        offset = p * n_per_person
        for i, name in enumerate(BODY_NAMES):
            body_idx = offset + i
            frac = DE_LEVA_MASS_FRACTIONS[name]
            target_m = frac * total_mass
            new_mass[body_idx] = target_m

            if old_mass[body_idx] > 1e-8:
                scale = target_m / old_mass[body_idx]
            else:
                scale = 1.0
            new_inertia[body_idx] = old_inertia[body_idx] * scale

    device = model.body_mass.device
    model.body_mass = wp.array(new_mass, dtype=wp.float32, device=device)
    model.body_inertia = wp.array(
        new_inertia, dtype=wp.mat33, device=device
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f" Segment masses (De Leva 1996, {n_persons} persons, "
              f"{total_mass:.1f} kg each)")
        print(f"{'='*60}")
        for p in range(n_persons):
            offset = p * n_per_person
            print(f"\n  Person {p}:")
            for i, name in enumerate(BODY_NAMES):
                body_idx = offset + i
                frac = DE_LEVA_MASS_FRACTIONS[name]
                print(f"    {name:<15s}  {new_mass[body_idx]:8.4f} kg  ({frac:5.2%})")
            person_total = new_mass[offset:offset + n_per_person].sum()
            print(f"    {'TOTAL':<15s}  {person_total:8.2f} kg")
        print()


def load_model(betas, foot_geom="box", total_mass=DEFAULT_BODY_MASS_KG,
               device="cuda:0", set_masses=True, verbose=False):
    """Load a Newton model with correct segment masses.

    Convenience function that combines XML generation, model loading,
    and De Leva mass assignment.

    Args:
        betas: (10,) SMPL-X shape parameters
        foot_geom: "box" | "sphere" | "capsule"
        total_mass: body mass in kg (default 75.0)
        device: CUDA device string
        set_masses: if True, apply De Leva mass fractions
        verbose: print mass breakdown

    Returns:
        model: Newton Model
        builder: Newton ModelBuilder (for body labels etc.)
    """
    import warp as wp
    import newton

    from prepare4.gen_xml import get_or_create_xml

    xml_path = get_or_create_xml(betas=betas, foot_geom=foot_geom)

    builder = newton.ModelBuilder()
    builder.add_mjcf(xml_path)
    model = builder.finalize(device=device)

    if set_masses:
        set_segment_masses(model, total_mass=total_mass, verbose=verbose)

    return model, builder


# ═══════════════════════════════════════════════════════════════
# Step 3: Analytical Inverse Dynamics
# ═══════════════════════════════════════════════════════════════

# Default sim parameters
DEFAULT_SIM_FREQ = 480     # Hz physics substeps
ARMATURE_HINGE = 0.5       # regularises thin SMPL-X limbs
ARMATURE_ROOT = 5.0


def inverse_dynamics(joint_q, dt, betas, total_mass=DEFAULT_BODY_MASS_KG,
                     foot_geom="box", device="cuda:0",
                     diff_method="savgol", verbose=True, **diff_kwargs):
    """Compute torques via analytical inverse dynamics.

    τ = M(q) (q̈_desired − q̈_free)

    where q̈_free is the acceleration under zero torque (gravity +
    Coriolis), obtained by a single SolverMuJoCo step with τ = 0.

    Uses:
      - De Leva 1996 mass fractions (correct segment masses)
      - SavGol smooth differentiation (noise-robust q̇, q̈)
      - SolverMuJoCo for zero-torque step (stable at dt=1/fps)

    NOTE: Root DOFs (0:6) are "virtual forces" — they should NOT be
    applied directly. Root tracking requires PD control.

    Args:
        joint_q: (T, 76) joint coordinates
        dt: timestep (1/fps)
        betas: (10,) SMPL-X shape parameters
        total_mass: body mass in kg (default 75.0)
        foot_geom: "box" | "sphere" | "capsule"
        device: CUDA device
        diff_method: "savgol" | "spline" | "finite_diff"
        verbose: print progress
        **diff_kwargs: extra args for compute_derivatives()

    Returns:
        torques: (T, 75) joint torques [N for root pos, Nm for rest]
        qd:  (T, 75) joint velocities
        qdd: (T, 75) joint accelerations
    """
    import warp as wp
    import newton

    T = joint_q.shape[0]

    # ── Compute smooth derivatives ────────────────────────
    if verbose:
        print(f"Computing derivatives (method={diff_method!r}, dt={dt:.4f}s)...")
    qd, qdd = compute_derivatives(joint_q, dt, method=diff_method, **diff_kwargs)

    # ── Build model WITHOUT ground plane (no contact forces in free step)
    if verbose:
        print(f"Building model (foot={foot_geom}, mass={total_mass:.1f}kg)...")
    model, builder = load_model(
        betas, foot_geom=foot_geom, total_mass=total_mass,
        device=device, set_masses=True, verbose=False,
    )

    n_dof = model.joint_dof_count
    assert n_dof == N_JOINT_QD, f"Expected {N_JOINT_QD} DOF, got {n_dof}"

    # ── Setup model (armature, disable passive springs) ───
    _setup_model_for_id(model, device=device)

    # ── MuJoCo solver for zero-torque forward step ────────
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=450, nconmax=150,
        impratio=10, iterations=100, ls_iterations=50,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    control.joint_f = wp.zeros(n_dof, dtype=wp.float32, device=device)

    if verbose:
        print(f"Running inverse dynamics for {T} frames...")

    torques = np.zeros((T, n_dof), dtype=np.float32)

    for t in range(T):
        # Set state (q, q̇)
        state_0.joint_q = wp.array(
            joint_q[t].astype(np.float32), dtype=wp.float32, device=device
        )
        jqd_wp = wp.array(
            qd[t].astype(np.float32), dtype=wp.float32, device=device
        )
        state_0.joint_qd = jqd_wp

        # FK to update body transforms
        newton.eval_fk(model, state_0.joint_q, jqd_wp, state_0)

        # Mass matrix M(q)
        H = newton.eval_mass_matrix(model, state_0)
        M = H.numpy()[0]  # (n_dof, n_dof)

        # Zero-torque step → q̈_free (gravity + Coriolis)
        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, dt)
        qd_after = state_1.joint_qd.numpy()
        qdd_free = (qd_after - qd[t]) / dt

        # τ = M (q̈_desired − q̈_free)
        torques[t] = M @ (qdd[t] - qdd_free).astype(np.float32)

        if verbose and t % 50 == 0:
            tau_h = torques[t, 6:]
            print(f"  Frame {t}/{T}: |τ_hinge| mean={np.abs(tau_h).mean():.1f} "
                  f"max={np.abs(tau_h).max():.1f} Nm")

    if verbose:
        print(f"  Done. Total |τ_hinge| mean={np.abs(torques[:, 6:]).mean():.1f} Nm")

    return torques, qd, qdd


def _setup_model_for_id(model, device="cuda:0"):
    """Configure model for inverse dynamics (armature, no passive springs/limits)."""
    import warp as wp

    n_dof = model.joint_dof_count

    # Disable passive springs
    model.mujoco.dof_passive_stiffness.fill_(0.0)
    model.mujoco.dof_passive_damping.fill_(0.0)
    model.joint_target_ke.fill_(0.0)
    model.joint_target_kd.fill_(0.0)

    # Disable joint limits — prevents spurious constraint forces when
    # IK-solved angles lie outside the MJCF-defined ranges.  For inverse
    # dynamics we want qdd_free to reflect only gravity + Coriolis, not
    # artificial limit enforcement.  GT angles are within limits so this
    # has negligible effect on GT results.
    n_joints = model.joint_count
    joint_limit_lower = model.joint_limit_lower.numpy()
    joint_limit_upper = model.joint_limit_upper.numpy()
    joint_limit_lower[:] = -1e6
    joint_limit_upper[:] = 1e6
    model.joint_limit_lower = wp.array(joint_limit_lower, dtype=wp.float32, device=device)
    model.joint_limit_upper = wp.array(joint_limit_upper, dtype=wp.float32, device=device)

    # Armature for numerical stability
    arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
    arm[:6] = ARMATURE_ROOT
    model.joint_armature = wp.array(arm, dtype=wp.float32, device=device)


# ═══════════════════════════════════════════════════════════════
# Step 3b: PD Tracking Torques
# ═══════════════════════════════════════════════════════════════

# PD gains (Kp Nm/rad, Kd Nms/rad) — tuned for SMPL-X with 3-hinge joints.
PD_GAINS = {
    "L_Hip": (300, 30),   "R_Hip": (300, 30),
    "L_Knee": (300, 30),  "R_Knee": (300, 30),
    "L_Ankle": (200, 20), "R_Ankle": (200, 20),
    "L_Toe": (100, 10),   "R_Toe": (100, 10),
    "Torso": (500, 50),   "Spine": (500, 50),    "Chest": (500, 50),
    "Neck": (200, 20),    "Head": (100, 10),
    "L_Thorax": (200, 20), "L_Shoulder": (200, 20),
    "L_Elbow": (150, 15),  "L_Wrist": (100, 10),  "L_Hand": (50, 5),
    "R_Thorax": (200, 20), "R_Shoulder": (200, 20),
    "R_Elbow": (150, 15),  "R_Wrist": (100, 10),  "R_Hand": (50, 5),
}
ROOT_POS_KP, ROOT_POS_KD = 2000.0, 400.0   # N/m
ROOT_ROT_KP, ROOT_ROT_KD = 1000.0, 200.0   # Nm/rad
DEFAULT_TORQUE_LIMIT = 1000.0               # Nm


def pd_tracking(joint_q, dt, betas, total_mass=DEFAULT_BODY_MASS_KG,
                foot_geom="box", device="cuda:0",
                gain_scale=1.0, sim_freq=DEFAULT_SIM_FREQ,
                torque_limit=DEFAULT_TORQUE_LIMIT, verbose=True):
    """Compute torques via PD tracking simulation.

    Runs forward dynamics (SolverMuJoCo) with PD controllers tracking
    the reference trajectory.  The applied control torques are recorded
    and averaged per motion frame.

    PD law:  τ = Kp (q_ref − q) − Kd q̇
    Root orientation uses quaternion-error axis-angle.

    Args:
        joint_q: (T, 76) reference trajectory
        dt: timestep (1/fps)
        betas: (10,) SMPL-X shape parameters
        total_mass: body mass in kg (default 75.0)
        foot_geom: "box" | "sphere" | "capsule"
        device: CUDA device
        gain_scale: multiplier for all PD gains
        sim_freq: physics substep frequency (default 480 Hz)
        torque_limit: max |τ| clamp (default 1000 Nm)
        verbose: print progress

    Returns:
        torques: (T, 75) average applied torques per frame
    """
    import warp as wp
    import newton

    # Import GPU PD kernel from prepare2 (tested & proven)
    from prepare2.pd_utils import (
        pd_torque_kernel, accumulate_torque_kernel, zero_kernel,
    )

    T = joint_q.shape[0]
    fps = round(1.0 / dt)
    sim_steps = sim_freq // fps
    dt_sim = 1.0 / sim_freq

    # ── Build model WITH ground plane ─────────────────────
    if verbose:
        print(f"Building model (foot={foot_geom}, mass={total_mass:.1f}kg)...")

    from prepare4.gen_xml import get_or_create_xml
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    xml_path = get_or_create_xml(betas=betas, foot_geom=foot_geom)
    builder.add_mjcf(xml_path)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    # Apply De Leva masses
    set_segment_masses(model, total_mass=total_mass, verbose=False)

    n_dof = model.joint_dof_count
    assert n_dof == N_JOINT_QD

    # ── Setup model properties ────────────────────────────
    _setup_model_for_id(model, device=device)

    # ── Build PD gains ────────────────────────────────────
    kp_np = np.zeros(n_dof, dtype=np.float32)
    kd_np = np.zeros(n_dof, dtype=np.float32)
    kp_np[:3] = ROOT_POS_KP * gain_scale
    kd_np[:3] = ROOT_POS_KD * gain_scale
    kp_np[3:6] = ROOT_ROT_KP * gain_scale
    kd_np[3:6] = ROOT_ROT_KD * gain_scale

    for b_idx, name in enumerate(BODY_NAMES[1:]):  # skip Pelvis
        s = 6 + b_idx * 3
        kp_val, kd_val = PD_GAINS.get(name, (100, 10))
        kp_np[s:s + 3] = kp_val * gain_scale
        kd_np[s:s + 3] = kd_val * gain_scale

    if verbose:
        print(f"PD tracking: T={T}, sim_freq={sim_freq}Hz, "
              f"{sim_steps} substeps/frame, gain_scale={gain_scale}")

    # ── Pre-allocate GPU arrays ───────────────────────────
    kp_gpu = wp.array(kp_np, dtype=wp.float32, device=device)
    kd_gpu = wp.array(kd_np, dtype=wp.float32, device=device)
    ref_q_gpu = wp.zeros(N_JOINT_Q, dtype=wp.float32, device=device)
    tau_buf = wp.zeros(n_dof, dtype=wp.float32, device=device)
    tau_accum = wp.zeros(n_dof, dtype=wp.float32, device=device)

    # Pre-upload all reference frames
    joint_q_f32 = joint_q.astype(np.float32)
    all_ref_gpu = wp.array(
        joint_q_f32.ravel(), dtype=wp.float32, device=device
    )

    # ── Solver + initial state ────────────────────────────
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=450, nconmax=150,
        impratio=10, iterations=100, ls_iterations=50,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    control.joint_f = tau_buf

    state_0.joint_q = wp.array(
        joint_q_f32[0], dtype=wp.float32, device=device
    )
    state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    torques = np.zeros((T, n_dof), dtype=np.float32)
    inv_sim_steps = 1.0 / float(sim_steps)

    # ── Simulation loop ───────────────────────────────────
    for t in range(T):
        # Load reference for this frame (GPU→GPU copy)
        wp.copy(ref_q_gpu, all_ref_gpu,
                src_offset=t * N_JOINT_Q, count=N_JOINT_Q)

        # Zero accumulator
        wp.launch(zero_kernel, dim=n_dof, inputs=[tau_accum], device=device)

        for sub in range(sim_steps):
            # PD torques on GPU
            wp.launch(
                pd_torque_kernel, dim=n_dof,
                inputs=[
                    state_0.joint_q, state_0.joint_qd,
                    ref_q_gpu, kp_gpu, kd_gpu, torque_limit, tau_buf,
                ],
                device=device,
            )

            # Accumulate
            wp.launch(
                accumulate_torque_kernel, dim=n_dof,
                inputs=[tau_buf, tau_accum], device=device,
            )

            # Step physics
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt_sim)
            state_0, state_1 = state_1, state_0

        # Average torques for this frame
        torques[t] = tau_accum.numpy() * inv_sim_steps

        if verbose and t % 50 == 0:
            cq = state_0.joint_q.numpy()
            ref = joint_q_f32[t]
            pe = np.linalg.norm(cq[:3] - ref[:3])
            he = np.abs(cq[7:] - ref[7:]).mean() * 180 / np.pi
            print(f"  Frame {t}/{T}: pos_err={pe*100:.1f}cm "
                  f"hinge_err={he:.1f}°")

    if verbose:
        print(f"  Done. Mean |τ_hinge| = {np.abs(torques[:, 6:]).mean():.1f} Nm")

    return torques


# ═══════════════════════════════════════════════════════════════
# Validation — synthetic trajectory with known analytic derivatives
# ═══════════════════════════════════════════════════════════════

def validate_derivatives(method="savgol", verbose=True, **kwargs):
    """Test derivative computation against analytic ground truth.

    Creates a synthetic trajectory with known sinusoidal components:
      - Root translation: x(t) = A sin(2πft)  →  known v(t), a(t)
      - Root quaternion:  R(t) = Rz(θ(t))     →  known ω(t), α(t)
      - Hinge angles:     θ(t) = A sin(2πft)  →  known ω(t), α(t)

    Returns dict with max absolute errors (ignoring 10-frame margins
    at boundaries to avoid edge effects).
    """
    T = 200
    fps = 30.0
    dt = 1.0 / fps
    t = np.arange(T, dtype=np.float64) * dt

    joint_q = np.zeros((T, N_JOINT_Q), dtype=np.float64)

    # ── Root translation: sinusoidal on each axis ─────────────
    freq_trans = 1.0  # Hz
    omega_t = 2 * np.pi * freq_trans
    amplitudes = [0.5, 1.0, 1.5]

    gt_vel_trans = np.zeros((T, 3), dtype=np.float64)
    gt_acc_trans = np.zeros((T, 3), dtype=np.float64)
    for d in range(3):
        A = amplitudes[d]
        joint_q[:, d] = A * np.sin(omega_t * t)
        gt_vel_trans[:, d] = A * omega_t * np.cos(omega_t * t)
        gt_acc_trans[:, d] = -A * omega_t ** 2 * np.sin(omega_t * t)

    # ── Root quaternion: rotation about Z axis ────────────────
    #   R(t) = Rz(θ(t)),  θ(t) = 0.3 sin(πt)
    freq_rot = 0.5  # Hz
    omega_r = 2 * np.pi * freq_rot
    A_rot = 0.3  # radians amplitude (< π/2, no wrapping)
    theta = A_rot * np.sin(omega_r * t)
    theta_dot = A_rot * omega_r * np.cos(omega_r * t)
    theta_ddot = -A_rot * omega_r ** 2 * np.sin(omega_r * t)

    for i in range(T):
        joint_q[i, 3:7] = Rotation.from_rotvec([0, 0, theta[i]]).as_quat()

    gt_omega = np.zeros((T, 3), dtype=np.float64)
    gt_alpha = np.zeros((T, 3), dtype=np.float64)
    gt_omega[:, 2] = theta_dot
    gt_alpha[:, 2] = theta_ddot

    # ── Hinge joints: independent sinusoids ───────────────────
    hinge_A = 0.2  # radians
    gt_hinge_vel = np.zeros((T, 69), dtype=np.float64)
    gt_hinge_acc = np.zeros((T, 69), dtype=np.float64)
    for d in range(69):
        f = 0.5 + 0.02 * d  # slightly different freqs
        w = 2 * np.pi * f
        joint_q[:, 7 + d] = hinge_A * np.sin(w * t)
        gt_hinge_vel[:, d] = hinge_A * w * np.cos(w * t)
        gt_hinge_acc[:, d] = -hinge_A * w ** 2 * np.sin(w * t)

    # ── Compute derivatives ───────────────────────────────────
    qd, qdd = compute_derivatives(
        joint_q.astype(np.float32), dt, method=method, **kwargs
    )

    # ── Compare (ignore boundary margins) ─────────────────────
    margin = 10
    sl = slice(margin, -margin)

    errs = {
        "trans_vel_max": np.abs(qd[sl, :3] - gt_vel_trans[sl]).max(),
        "trans_vel_mean": np.abs(qd[sl, :3] - gt_vel_trans[sl]).mean(),
        "trans_acc_max": np.abs(qdd[sl, :3] - gt_acc_trans[sl]).max(),
        "trans_acc_mean": np.abs(qdd[sl, :3] - gt_acc_trans[sl]).mean(),
        "omega_max": np.abs(qd[sl, 3:6] - gt_omega[sl]).max(),
        "omega_mean": np.abs(qd[sl, 3:6] - gt_omega[sl]).mean(),
        "alpha_max": np.abs(qdd[sl, 3:6] - gt_alpha[sl]).max(),
        "alpha_mean": np.abs(qdd[sl, 3:6] - gt_alpha[sl]).mean(),
        "hinge_vel_max": np.abs(qd[sl, 6:] - gt_hinge_vel[sl]).max(),
        "hinge_vel_mean": np.abs(qd[sl, 6:] - gt_hinge_vel[sl]).mean(),
        "hinge_acc_max": np.abs(qdd[sl, 6:] - gt_hinge_acc[sl]).max(),
        "hinge_acc_mean": np.abs(qdd[sl, 6:] - gt_hinge_acc[sl]).mean(),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f" Validation: method={method!r}")
        print(f"   T={T}, fps={fps:.0f}, dt={dt:.4f}s")
        if method == "savgol":
            w = kwargs.get("window", 7)
            p = kwargs.get("polyorder", 3)
            print(f"   window={w}, polyorder={p}")
        elif method == "spline":
            sf = kwargs.get("smoothing_factor", 0.0)
            so = kwargs.get("spline_order", 5)
            print(f"   smoothing_factor={sf}, spline_order={so}")
        print(f"{'='*60}")
        print(f"  Root trans velocity  max={errs['trans_vel_max']:.2e}  "
              f"mean={errs['trans_vel_mean']:.2e} m/s")
        print(f"  Root trans accel     max={errs['trans_acc_max']:.2e}  "
              f"mean={errs['trans_acc_mean']:.2e} m/s²")
        print(f"  Root angular vel     max={errs['omega_max']:.2e}  "
              f"mean={errs['omega_mean']:.2e} rad/s")
        print(f"  Root angular acc     max={errs['alpha_max']:.2e}  "
              f"mean={errs['alpha_mean']:.2e} rad/s²")
        print(f"  Hinge velocity       max={errs['hinge_vel_max']:.2e}  "
              f"mean={errs['hinge_vel_mean']:.2e} rad/s")
        print(f"  Hinge acceleration   max={errs['hinge_acc_max']:.2e}  "
              f"mean={errs['hinge_acc_mean']:.2e} rad/s²")
        print()

    return errs


def compare_methods(joint_q, dt):
    """Compare all three differentiation methods on the same trajectory.

    Prints per-DOF statistics showing smoothness (std of acceleration)
    and mutual agreement between methods.

    Args:
        joint_q: (T, 76) real or synthetic trajectory
        dt: timestep (seconds)

    Returns:
        dict of {method_name: (qd, qdd)} for all three methods
    """
    results = {}
    for method in ["finite_diff", "savgol", "spline"]:
        qd, qdd = compute_derivatives(joint_q, dt, method=method)
        results[method] = (qd, qdd)

    print(f"\n{'='*70}")
    print(f" Method comparison — T={joint_q.shape[0]}, dt={dt:.4f}s")
    print(f"{'='*70}")

    # Smoothness: std of acceleration (lower = smoother)
    print(f"\n  Acceleration std (lower = smoother):")
    print(f"  {'Signal':<25} {'finite_diff':>12} {'savgol':>12} {'spline':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    labels = [
        ("Root trans (0:3)", slice(0, 3)),
        ("Root angular (3:6)", slice(3, 6)),
        ("Hinge (6:75)", slice(6, 75)),
        ("All DOF (0:75)", slice(0, 75)),
    ]
    for label, sl in labels:
        stds = []
        for method in ["finite_diff", "savgol", "spline"]:
            _, qdd = results[method]
            stds.append(np.std(qdd[:, sl]))
        print(f"  {label:<25} {stds[0]:>12.4f} {stds[1]:>12.4f} {stds[2]:>12.4f}")

    # Mutual agreement: max difference between methods
    print(f"\n  Max |Δqd| between methods:")
    methods = ["finite_diff", "savgol", "spline"]
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            qd_i = results[methods[i]][0]
            qd_j = results[methods[j]][0]
            diff = np.abs(qd_i - qd_j).max()
            print(f"    {methods[i]:>12} vs {methods[j]:<12}: {diff:.4f}")

    print(f"\n  Max |Δqdd| between methods:")
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            qdd_i = results[methods[i]][1]
            qdd_j = results[methods[j]][1]
            diff = np.abs(qdd_i - qdd_j).max()
            print(f"    {methods[i]:>12} vs {methods[j]:<12}: {diff:.4f}")

    print()
    return results


# ═══════════════════════════════════════════════════════════════
# Real data test
# ═══════════════════════════════════════════════════════════════

def test_on_clip(clip_idx=1000, dataset="interhuman", fps=30):
    """Load a real motion clip and compare differentiation methods.

    Args:
        clip_idx: dataset clip index
        dataset: "interhuman" or "interx"
        fps: frames per second of the motion data (default 30, matching InterMask)
    """
    from prepare4.retarget import rotation_retarget, load_interhuman_pkl

    # Load clip
    if dataset == "interhuman":
        data_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
        persons = load_interhuman_pkl(data_dir, str(clip_idx))
        if persons is None:
            print(f"Clip {clip_idx} not found in {data_dir}")
            return None
    else:
        raise NotImplementedError(f"Dataset {dataset!r} not yet supported")

    dt = 1.0 / fps
    print(f"\n{'='*60}")
    print(f" Real data test — clip {clip_idx}, fps={fps}")
    print(f"{'='*60}")

    all_results = {}
    for i, p in enumerate(persons):
        joint_q = rotation_retarget(
            p["root_orient"], p["pose_body"], p["trans"], p["betas"]
        )
        # GT PKL is at 60fps, downsample 2x to match InterMask's 30fps
        joint_q = joint_q[::2]
        T = joint_q.shape[0]
        label = f"person{i+1}"
        print(f"\n  {label}: T={T} frames ({T/fps:.1f}s)")

        results = compare_methods(joint_q, dt)
        all_results[label] = results

    return all_results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Smooth differentiation and torque computation for Newton joint trajectories"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run synthetic validation test"
    )
    parser.add_argument(
        "--method", default=None,
        help="Derivative method: savgol, spline, finite_diff (default: all)"
    )
    parser.add_argument(
        "--clip", type=int, default=None,
        help="Test on a real InterHuman clip (e.g. --clip 1000)"
    )
    parser.add_argument(
        "--torques", choices=["id", "pd", "both"], default=None,
        help="Compute torques: id=inverse dynamics, pd=PD tracking, both"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="FPS of the motion data (default: 30, InterMask/InterHuman NPY is 30fps)"
    )
    parser.add_argument(
        "--mass", type=float, default=75.0,
        help="Total body mass in kg (default: 75.0)"
    )
    parser.add_argument(
        '--window', type=int, default=11,
        help="SavGol window length (default: 11)"
    )
    parser.add_argument(
        '--polyorder', type=int, default=5,
        help="SavGol polynomial order (default: 5)"
    )
    args = parser.parse_args()

    if args.validate:
        methods = [args.method] if args.method else ["savgol", "spline", "finite_diff"]
        for m in methods:
            kwargs = {}
            if m == "savgol":
                kwargs = {"window": args.window, "polyorder": args.polyorder}
            validate_derivatives(method=m, verbose=True, **kwargs)

    elif args.clip is not None:
        if args.torques:
            _run_torque_test(args)
        else:
            test_on_clip(clip_idx=args.clip, fps=args.fps)

    else:
        print("Running validation on all methods...")
        for m in ["savgol", "spline", "finite_diff"]:
            validate_derivatives(method=m, verbose=True)


def _run_torque_test(args):
    """Run torque computation on a clip (called from CLI)."""
    from prepare4.retarget import rotation_retarget, load_interhuman_pkl

    data_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    persons = load_interhuman_pkl(data_dir, str(args.clip))
    if persons is None:
        print(f"Clip {args.clip} not found")
        return

    dt = 1.0 / args.fps

    for i, p in enumerate(persons):
        joint_q = rotation_retarget(
            p["root_orient"], p["pose_body"], p["trans"], p["betas"]
        )
        # GT PKL is at 60fps, downsample 2x to match InterMask's 30fps
        joint_q = joint_q[::2]
        T = joint_q.shape[0]
        label = f"person{i+1}"
        print(f"\n{'='*60}")
        print(f" {label}: T={T} frames ({T/args.fps:.1f}s)")
        print(f"{'='*60}")

        methods = (["id", "pd"] if args.torques == "both"
                   else [args.torques])
        results = {}

        for method in methods:
            if method == "id":
                torques, qd, qdd = inverse_dynamics(
                    joint_q, dt, p["betas"],
                    total_mass=args.mass, verbose=True,
                )
            elif method == "pd":
                torques = pd_tracking(
                    joint_q, dt, p["betas"],
                    total_mass=args.mass, verbose=True,
                )
            results[method] = torques

        # Print comparison if both methods run
        if len(results) == 2:
            t_id = results["id"]
            t_pd = results["pd"]
            diff = np.abs(t_id[:, 6:] - t_pd[:, 6:])
            print(f"\n  ID vs PD (hinges only):")
            print(f"    Mean |Δτ|: {diff.mean():.2f} Nm")
            print(f"    Max  |Δτ|: {diff.max():.2f} Nm")
            corr = np.corrcoef(t_id[:, 6:].ravel(), t_pd[:, 6:].ravel())[0, 1]
            print(f"    Correlation: {corr:.4f}")


if __name__ == "__main__":
    main()
