"""
Compute joint torques required to execute retargeted motions.

Two methods:
  1. Inverse Dynamics (RNEA-like): τ = M(q)(q̈ - q̈_free)
     - M(q): mass matrix from Newton
     - q̈: desired accelerations from finite differences
     - q̈_free: free accelerations under zero torque (captures
       gravity + Coriolis bias forces)
     - This gives τ = M*q̈ + h(q,q̇) where h = -M*q̈_free

  2. PD Tracking: Run physics simulation tracking reference motion
     with PD controllers, extract the control torques.

Usage:
    # Inverse dynamics on a single clip (saves _torques_solo.npy)
    python prepare2/compute_torques.py --clip 1000 --method inverse --save

    # PD tracking simulation (GPU-accelerated)
    python prepare2/compute_torques.py --clip 1000 --method pd --save

    # Batch process entire dataset
    python prepare2/compute_torques.py --dataset interhuman --method pd --save

    # Parallel batch processing (multiple workers)
    python prepare2/compute_torques.py --dataset interhuman --method pd --save --workers 3

    # Custom output directory (default: data/compute_torques/{dataset})
    python prepare2/compute_torques.py --clip 1000 --method pd --save \
        --output-dir data/my_torques
"""
import os
import sys
import time
import argparse
import warnings
import multiprocessing as mp
import numpy as np

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import get_or_create_xml, SMPL_TO_NEWTON, N_SMPL_JOINTS
from prepare2.pd_utils import (
    BODY_NAMES, DOF_NAMES, BODY_GAINS as _PD_BODY_GAINS,
    DOFS_PER_PERSON, COORDS_PER_PERSON,
    ARMATURE_HINGE, ARMATURE_ROOT,
    ROOT_POS_KP, ROOT_POS_KD, ROOT_ROT_KP, ROOT_ROT_KD,
    DEFAULT_SIM_FREQ, DEFAULT_TORQUE_LIMIT,
    pd_torque_kernel, accumulate_torque_kernel, zero_kernel,
    build_model, setup_model_properties, create_mujoco_solver,
)


# GPU kernels for PD torque computation are in pd_utils.py


def compute_qd_qdd(joint_q, dt):
    """
    Compute joint velocities and accelerations via finite differences.

    Note: joint_q has 76 coords (7 root + 23*3) but DOF = 75
    (6 root DOF + 23*3). For the root free joint, qd uses angular
    velocity (3 DOF) instead of quaternion derivative (4 params).

    For simplicity, we compute velocities in generalized coordinates
    using central differences. The root quaternion is handled by
    converting to angular velocity.

    Args:
        joint_q: (T, 76) joint coordinates
        dt: time step in seconds

    Returns:
        qd: (T, 75) joint velocities
        qdd: (T, 75) joint accelerations
    """
    from scipy.spatial.transform import Rotation

    T = joint_q.shape[0]
    qd = np.zeros((T, 75), dtype=np.float64)
    qdd = np.zeros((T, 75), dtype=np.float64)

    # ── Root translation velocity (3 DOF) ────────────────
    # Central differences for position
    for t in range(1, T - 1):
        qd[t, :3] = (joint_q[t + 1, :3] - joint_q[t - 1, :3]) / (2 * dt)
    # Forward/backward at boundaries
    qd[0, :3] = (joint_q[1, :3] - joint_q[0, :3]) / dt
    qd[-1, :3] = (joint_q[-1, :3] - joint_q[-2, :3]) / dt

    # ── Root angular velocity (3 DOF) ────────────────────
    # Convert quaternions (xyzw) to angular velocity
    for t in range(1, T - 1):
        q_prev = joint_q[t - 1, 3:7]
        q_next = joint_q[t + 1, 3:7]
        # scipy expects xyzw
        R_prev = Rotation.from_quat(q_prev)
        R_next = Rotation.from_quat(q_next)
        dR = R_next * R_prev.inv()
        omega = dR.as_rotvec() / (2 * dt)
        qd[t, 3:6] = omega
    # Boundaries
    R0 = Rotation.from_quat(joint_q[0, 3:7])
    R1 = Rotation.from_quat(joint_q[1, 3:7])
    qd[0, 3:6] = (R1 * R0.inv()).as_rotvec() / dt
    Rn1 = Rotation.from_quat(joint_q[-2, 3:7])
    Rn = Rotation.from_quat(joint_q[-1, 3:7])
    qd[-1, 3:6] = (Rn * Rn1.inv()).as_rotvec() / dt

    # ── Hinge joint velocities (69 DOF) ──────────────────
    # Joint angles are in joint_q[7:76] (69 values = 23 bodies * 3 hinges)
    # DOF indices 6:75 correspond to joint_q[7:76]
    hinge_q = joint_q[:, 7:]  # (T, 69)
    for t in range(1, T - 1):
        qd[t, 6:] = (hinge_q[t + 1] - hinge_q[t - 1]) / (2 * dt)
    qd[0, 6:] = (hinge_q[1] - hinge_q[0]) / dt
    qd[-1, 6:] = (hinge_q[-1] - hinge_q[-2]) / dt

    # ── Accelerations (central differences on velocities) ──
    for t in range(1, T - 1):
        qdd[t] = (qd[t + 1] - qd[t - 1]) / (2 * dt)
    qdd[0] = (qd[1] - qd[0]) / dt
    qdd[-1] = (qd[-1] - qd[-2]) / dt

    return qd.astype(np.float32), qdd.astype(np.float32)


def compute_qd_qdd_spline(joint_q, dt, smoothing_factor=0.0):
    """
    Compute joint velocities and accelerations via cubic B-spline fitting.

    Unlike finite differences, spline fitting performs implicit smoothing
    during the fit rather than amplifying noise through differentiation.
    The spline's analytic derivatives are evaluated at each sample point.

    This avoids the O(1/dt²) noise amplification inherent in central
    finite-difference double-differentiation.

    Args:
        joint_q: (T, 76) joint coordinates
        dt: time step in seconds
        smoothing_factor: scipy splrep smoothing (0 = interpolating spline,
                          positive values add smoothing). A small value like
                          T * 1e-6 works well.

    Returns:
        qd: (T, 75) joint velocities
        qdd: (T, 75) joint accelerations
    """
    from scipy.interpolate import splrep, splev
    from scipy.spatial.transform import Rotation, Slerp

    T = joint_q.shape[0]
    qd = np.zeros((T, 75), dtype=np.float64)
    qdd = np.zeros((T, 75), dtype=np.float64)
    t_arr = np.arange(T, dtype=np.float64) * dt

    s = smoothing_factor if smoothing_factor > 0 else T * 1e-6

    # ── Root translation (DOF 0:3) via spline ────────────────
    for d in range(3):
        tck = splrep(t_arr, joint_q[:, d].astype(np.float64), s=s, k=5)
        qd[:, d] = splev(t_arr, tck, der=1)
        qdd[:, d] = splev(t_arr, tck, der=2)

    # ── Root angular velocity (DOF 3:6) ──────────────────────
    # Use Slerp on quaternions, then differentiate the resulting
    # rotation vector representation with a spline.
    quats = joint_q[:, 3:7].astype(np.float64)
    rotations = Rotation.from_quat(quats)
    # Convert to rotation vectors for smooth spline fitting
    rotvecs = rotations.as_rotvec()
    # Unwrap rotation vectors to avoid discontinuities
    for t in range(1, T):
        diff = rotvecs[t] - rotvecs[t - 1]
        # If the rotation vector jumps by more than π, unwrap
        norm = np.linalg.norm(diff)
        if norm > np.pi:
            rotvecs[t] -= 2 * np.pi * diff / norm

    for d in range(3):
        tck = splrep(t_arr, rotvecs[:, d], s=s, k=5)
        qd[:, 3 + d] = splev(t_arr, tck, der=1)
        qdd[:, 3 + d] = splev(t_arr, tck, der=2)

    # ── Hinge joints (DOF 6:75) via spline ───────────────────
    hinge_q = joint_q[:, 7:].astype(np.float64)  # (T, 69)
    for d in range(69):
        tck = splrep(t_arr, hinge_q[:, d], s=s, k=5)
        qd[:, 6 + d] = splev(t_arr, tck, der=1)
        qdd[:, 6 + d] = splev(t_arr, tck, der=2)

    return qd.astype(np.float32), qdd.astype(np.float32)


def smooth_trajectory(data, fps, cutoff=6.0, order=4):
    """
    Zero-phase Butterworth low-pass filter on trajectory data.

    Applied per-DOF independently. Uses filtfilt for zero phase
    distortion. Handles edge effects with padding.

    Args:
        data: (T, D) array — time series per DOF
        fps: sampling frequency in Hz
        cutoff: cutoff frequency in Hz (default 6.0)
        order: Butterworth filter order (default 4)

    Returns:
        filtered: (T, D) smoothed data
    """
    from scipy.signal import butter, filtfilt

    T = data.shape[0]
    if T < 2 * order + 1:
        return data.copy()

    nyq = fps / 2.0
    if cutoff >= nyq:
        cutoff = nyq * 0.9  # clamp to avoid instability

    b, a = butter(order, cutoff / nyq, btype="low")
    filtered = np.zeros_like(data)
    for d in range(data.shape[1]):
        filtered[:, d] = filtfilt(b, a, data[:, d])
    return filtered


def inverse_dynamics(model, joint_q, fps, device="cuda:0",
                     smooth=True, cutoff_hz=6.0, diff_method="spline"):
    """
    Compute torques via inverse dynamics: τ = M(q)(q̈_desired - q̈_free)

    The bias force h(q,q̇) = C(q,q̇)q̇ + g(q) is captured implicitly:
      - q̈_free is the acceleration that would occur under zero external
        torque (gravity + Coriolis + contacts), obtained by a single
        forward dynamics step with SolverMuJoCo and τ=0.
      - Then τ = M * (q̈_desired - q̈_free) gives the full actuator
        torques including gravity compensation and Coriolis terms.

    Uses SolverMuJoCo (implicit contacts) for the zero-torque step.
    SolverFeatherstone uses penalty-based contacts that are numerically
    unstable at the motion-capture timestep (1/30s).

    NOTE: The root DOFs (0:6) in the output are "virtual forces" needed
    to achieve the desired root trajectory. They should NOT be applied
    directly in simulation — root tracking must use PD control.

    Args:
        model: Newton Model (must have armature + ground plane configured)
        joint_q: (T, 76) joint coordinates
        fps: frames per second
        device: compute device
        smooth: apply Butterworth smoothing to qd/qdd (default True).
                Only used when diff_method="fd".
        cutoff_hz: smoothing cutoff frequency (default 6.0 Hz)
        diff_method: "spline" (cubic B-spline, recommended) or "fd"
                     (central finite differences + Butterworth filtering).
                     Spline method avoids O(1/dt²) noise amplification.

    Returns:
        torques: (T, 75) joint torques
        qd: (T, 75) joint velocities
        qdd: (T, 75) joint accelerations
    """
    dt = 1.0 / fps
    T = joint_q.shape[0]
    n_dof = model.joint_dof_count

    if diff_method == "spline":
        print(f"Computing velocities/accelerations via B-spline (dt={dt:.4f}s)...")
        qd, qdd = compute_qd_qdd_spline(joint_q, dt)
    else:
        print(f"Computing velocities/accelerations via finite differences (dt={dt:.4f}s)...")
        qd, qdd = compute_qd_qdd(joint_q, dt)
        # Only apply Butterworth smoothing for finite-difference method
        if smooth and T > 10:
            print(f"Applying Butterworth smoothing (cutoff={cutoff_hz} Hz)...")
            qd = smooth_trajectory(qd, fps, cutoff=cutoff_hz)
            qdd = smooth_trajectory(qdd, fps, cutoff=cutoff_hz)

    print(f"Computing inverse dynamics for {T} frames...")
    torques = np.zeros((T, n_dof), dtype=np.float32)

    # Set up model properties (armature for stability, disable passive springs)
    setup_model_properties(model, n_persons=1, device=device)

    # Use MuJoCo solver for zero-torque step — implicit contacts are
    # numerically stable at the motion-capture timestep, unlike
    # Featherstone's penalty contacts which diverge at dt > ~0.001s.
    solver = create_mujoco_solver(model, n_persons=1)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    # Zero torques for the free-fall step
    control.joint_f = wp.zeros(n_dof, dtype=wp.float32, device=device)

    for t in range(T):
        # ── Set current state (position + velocity) ──────────
        state_0.joint_q = wp.array(
            joint_q[t].astype(np.float32), dtype=wp.float32, device=device
        )
        jqd_wp = wp.array(qd[t].astype(np.float32), dtype=wp.float32, device=device)
        state_0.joint_qd = jqd_wp

        # Evaluate FK to update body transforms
        newton.eval_fk(model, state_0.joint_q, jqd_wp, state_0)

        # ── Get mass matrix M(q) ─────────────────────────────
        H = newton.eval_mass_matrix(model, state_0)
        M = H.numpy()[0]  # (n_dof, n_dof)

        # ── Zero-torque forward step → q̈_free ───────────────
        # MuJoCo implicit contacts give stable q̈_free at dt=1/fps
        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, dt)

        qd_after = state_1.joint_qd.numpy()
        qdd_free = (qd_after - qd[t]) / dt  # (n_dof,)

        # ── τ = M * (q̈_desired - q̈_free) ────────────────────
        torques[t] = M @ (qdd[t] - qdd_free).astype(np.float32)

        if t % 50 == 0:
            tau_hinge = torques[t, 6:]
            print(f"  Frame {t}/{T}: |τ_hinge|={np.abs(tau_hinge).mean():.1f} Nm "
                  f"(max={np.abs(tau_hinge).max():.1f} Nm)")

    return torques, qd, qdd


def pd_tracking(model, joint_q, fps, gain_scale=1.0,
                device="cuda:0"):
    """
    Compute torques via PD tracking simulation (GPU-accelerated).

    Runs Newton physics (SolverMuJoCo) with PD controllers computed
    entirely on GPU via warp kernels, avoiding CPU↔GPU transfers in
    the inner loop.

    Args:
        model: Newton Model
        joint_q: (T, 76) joint coordinates
        fps: frames per second
        gain_scale: multiplier for all PD gains (default 1.0)
        device: compute device

    Returns:
        torques: (T, 75) joint torques (per-frame average)
    """
    sim_freq = DEFAULT_SIM_FREQ
    sim_steps = sim_freq // fps
    dt_sim = 1.0 / sim_freq
    T = joint_q.shape[0]
    n_dof = model.joint_dof_count
    n_coords = COORDS_PER_PERSON

    # ── Disable passive springs, use explicit PD ─────────
    setup_model_properties(model, n_persons=1, device=device)

    # ── Per-body PD gains ────────────────────────────────
    kp_np = np.zeros(n_dof, dtype=np.float32)
    kd_np = np.zeros(n_dof, dtype=np.float32)
    kp_np[:3] = ROOT_POS_KP * gain_scale
    kd_np[:3] = ROOT_POS_KD * gain_scale
    kp_np[3:6] = ROOT_ROT_KP * gain_scale
    kd_np[3:6] = ROOT_ROT_KD * gain_scale
    for i, name in enumerate(lbl.rsplit('/', 1)[-1] for lbl in model.body_label[1:]):
        s = 6 + i * 3
        k, d = _PD_BODY_GAINS.get(name, (100, 10))
        kp_np[s:s + 3] = k * gain_scale
        kd_np[s:s + 3] = d * gain_scale
    torque_limit = DEFAULT_TORQUE_LIMIT

    print(f"PD tracking (GPU): {T} frames, sim_freq={sim_freq}Hz, "
          f"dt={dt_sim:.5f}s, {sim_steps} substeps/frame, "
          f"gain_scale={gain_scale}")

    # ── Pre-allocate GPU arrays (allocated once, reused) ──
    kp_gpu = wp.array(kp_np, dtype=wp.float32, device=device)
    kd_gpu = wp.array(kd_np, dtype=wp.float32, device=device)
    ref_q_gpu = wp.zeros(n_coords, dtype=wp.float32, device=device)
    tau_buf = wp.zeros(n_dof, dtype=wp.float32, device=device)
    tau_accum = wp.zeros(n_dof, dtype=wp.float32, device=device)

    # Pre-upload all reference frames to GPU as contiguous block
    joint_q_f32 = joint_q.astype(np.float32)
    all_ref_gpu = wp.array(
        joint_q_f32.ravel(), dtype=wp.float32, device=device
    )

    # ── Initialize solver ────────────────────────────────
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton", njmax=450, nconmax=150,
        impratio=10, iterations=100, ls_iterations=50,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    control.joint_f = tau_buf  # reuse same GPU buffer

    state_0.joint_q = wp.array(
        joint_q_f32[0], dtype=wp.float32, device=device
    )
    state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    torques = np.zeros((T, n_dof), dtype=np.float32)
    inv_sim_steps = 1.0 / float(sim_steps)

    for t in range(T):
        # Copy this frame's ref_q from pre-uploaded block (GPU→GPU)
        wp.copy(ref_q_gpu, all_ref_gpu,
                src_offset=t * n_coords, count=n_coords)

        # Zero the frame accumulator
        wp.launch(zero_kernel, dim=n_dof, inputs=[tau_accum], device=device)

        for sub in range(sim_steps):
            # Compute PD torques entirely on GPU
            wp.launch(
                pd_torque_kernel, dim=n_dof,
                inputs=[
                    state_0.joint_q, state_0.joint_qd,
                    ref_q_gpu, kp_gpu, kd_gpu, torque_limit, tau_buf,
                ],
                device=device,
            )

            # Accumulate for frame averaging
            wp.launch(
                accumulate_torque_kernel, dim=n_dof,
                inputs=[tau_buf, tau_accum],
                device=device,
            )

            # Step physics (tau_buf is already control.joint_f)
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt_sim)
            state_0, state_1 = state_1, state_0

        # Pull averaged torques for this frame (1 transfer per frame)
        tau_accum_np = tau_accum.numpy()
        torques[t] = tau_accum_np * inv_sim_steps

        if t % 50 == 0:
            cq = state_0.joint_q.numpy()
            ref = joint_q_f32[t]
            pe = np.linalg.norm(cq[:3] - ref[:3])
            he = np.abs(cq[7:] - ref[7:]).mean() * 180 / np.pi
            print(f"  Frame {t}/{T}: pos_err={pe*100:.1f}cm "
                  f"hinge_err={he:.1f}°")

    return torques


def analyze_torques(torques, fps, save_path=None):
    """Print torque statistics and optionally save."""
    T, n_dof = torques.shape

    print(f"\n{'='*60}")
    print(f"Torque Analysis: {T} frames, {n_dof} DOF, FPS={fps}")
    print(f"{'='*60}")

    # Per-DOF statistics
    print(f"\n{'DOF':<20s} {'Mean':>8s} {'Std':>8s} {'Max':>8s} {'Unit'}")
    print("-" * 52)

    # Root (6 DOF: 3 forces + 3 torques)
    for i in range(3):
        name = DOF_NAMES[i]
        vals = torques[:, i]
        print(f"{name:<20s} {vals.mean():8.2f} {vals.std():8.2f} "
              f"{np.abs(vals).max():8.2f}  N")
    for i in range(3, 6):
        name = DOF_NAMES[i]
        vals = torques[:, i]
        print(f"{name:<20s} {vals.mean():8.2f} {vals.std():8.2f} "
              f"{np.abs(vals).max():8.2f}  Nm")

    print()

    # Body joints grouped by body (3 DOF each)
    for body_idx in range(23):  # 23 non-root bodies
        dof_start = 6 + body_idx * 3
        body_name = BODY_NAMES[body_idx + 1]  # skip Pelvis
        for j in range(3):
            dof = dof_start + j
            name = DOF_NAMES[dof]
            vals = torques[:, dof]
            print(f"{name:<20s} {vals.mean():8.2f} {vals.std():8.2f} "
                  f"{np.abs(vals).max():8.2f}  Nm")

    # Summary
    print(f"\n{'='*60}")
    print("Summary (excluding root forces):")
    joint_torques = torques[:, 6:]  # only joint torques, not root
    print(f"  Overall mean |τ|:  {np.abs(joint_torques).mean():.2f} Nm")
    print(f"  Overall max  |τ|:  {np.abs(joint_torques).max():.2f} Nm")
    print(f"  RMS torque:        {np.sqrt((joint_torques**2).mean()):.2f} Nm")

    # Per-body total torque magnitude
    print(f"\n{'Body':<15s} {'Mean |τ|':>10s} {'Max |τ|':>10s}  Nm")
    print("-" * 40)
    for body_idx in range(23):
        dof_start = 6 + body_idx * 3
        body_name = BODY_NAMES[body_idx + 1]
        body_tau = torques[:, dof_start:dof_start + 3]
        tau_mag = np.linalg.norm(body_tau, axis=1)
        print(f"{body_name:<15s} {tau_mag.mean():10.2f} {tau_mag.max():10.2f}")

    if save_path:
        np.save(save_path, torques)
        print(f"\nTorques saved to: {save_path}")


def list_retargeted_clips(data_dir):
    """List all clip IDs that have retargeted joint_q files."""
    clips = set()
    for f in os.listdir(data_dir):
        if f.endswith("_joint_q.npy"):
            # e.g. "1000_person0_joint_q.npy" → "1000"
            parts = f.replace("_joint_q.npy", "").rsplit("_person", 1)
            if len(parts) == 2:
                clips.add(parts[0])
    return sorted(clips)


def process_clip(clip_id, person_idx, data_dir, output_dir, method, fps,
                 gain_scale, device, save, downsample=2, diff_method="spline"):
    """
    Process a single clip+person: compute torques, analyze, optionally save.

    Args:
        data_dir: directory with input joint_q + betas files
        output_dir: directory for output torque files

    Returns:
        torques: (T, 75) array or None on error
    """
    jq_path = os.path.join(data_dir, f"{clip_id}_person{person_idx}_joint_q.npy")
    betas_path = os.path.join(data_dir, f"{clip_id}_person{person_idx}_betas.npy")

    if not os.path.exists(jq_path):
        print(f"Error: {jq_path} not found")
        return None
    if not os.path.exists(betas_path):
        print(f"Error: {betas_path} not found")
        return None

    joint_q = np.load(jq_path)
    betas = np.load(betas_path)

    # Downsample from data FPS to target FPS (e.g. 60→30)
    if downsample > 1:
        joint_q = joint_q[::downsample]

    print(f"Loaded clip {clip_id} person {person_idx}: "
          f"{joint_q.shape[0]} frames, FPS={fps}"
          f" (downsample={downsample}x)")

    # Build model (ground plane always needed — SolverFeatherstone uses
    # penalty-based soft contacts via eval_body_contact, so the ground
    # plane must be present for inverse dynamics too, otherwise the
    # computed torques include redundant gravity compensation and the
    # characters fly when replayed in MuJoCo)
    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    print(f"Model: {model.joint_dof_count} DOF, "
          f"{model.state().body_count} bodies")

    # Compute torques
    if method == "inverse":
        torques, qd, qdd = inverse_dynamics(
            model, joint_q, fps, device=device,
            diff_method=diff_method,
        )
    else:
        torques = pd_tracking(
            model, joint_q, fps,
            gain_scale=gain_scale, device=device
        )
        qd, qdd = None, None

    # Analysis
    save_path = None
    if save:
        os.makedirs(output_dir, exist_ok=True)
        suffix = "_torques_solo" if method == "inverse" else "_torques_pd"
        save_path = os.path.join(
            output_dir,
            f"{clip_id}_person{person_idx}{suffix}.npy"
        )

    analyze_torques(torques, fps, save_path=save_path)

    # Also save qd/qdd for inverse dynamics
    if save and method == "inverse" and qd is not None:
        qd_path = os.path.join(
            output_dir, f"{clip_id}_person{person_idx}_qvel.npy"
        )
        qdd_path = os.path.join(
            output_dir, f"{clip_id}_person{person_idx}_qacc.npy"
        )
        np.save(qd_path, qd)
        np.save(qdd_path, qdd)
        print(f"Saved qvel → {qd_path}")
        print(f"Saved qacc → {qdd_path}")

    return torques


def _worker_init():
    """Initialize warp in each worker process (fresh CUDA context)."""
    import warp as wp
    wp.config.verbose = False
    wp.init()


def _worker_process_clip(args):
    """Multiprocessing worker: process a single clip+person."""
    (clip_id, person_idx, data_dir, output_dir, method,
     fps, gain_scale, device, save, downsample, diff_method) = args
    try:
        result = process_clip(
            clip_id, person_idx, data_dir, output_dir, method,
            fps, gain_scale, device, save, downsample, diff_method,
        )
        return clip_id, person_idx, result is not None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return clip_id, person_idx, False


def main():
    parser = argparse.ArgumentParser(
        description="Compute joint torques for retargeted motions"
    )
    parser.add_argument("--clip", type=str, default=None,
                        help="Clip ID (e.g. '1000')")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["interhuman", "interx"],
                        help="Process entire dataset")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Person index (default: both)")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman",
                        help="Directory with joint_q + betas files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/compute_torques/{dataset})")
    parser.add_argument("--method", choices=["inverse", "pd"],
                        default="inverse",
                        help="Method: inverse dynamics or PD tracking")
    parser.add_argument("--diff-method", choices=["spline", "fd"],
                        default="spline",
                        help="Differentiation method for I.D.: 'spline' (B-spline, "
                             "recommended) or 'fd' (finite differences + Butterworth)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Motion FPS (default 30 = InterMask eval rate)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample loaded data by this factor "
                             "(2 = 60→30fps to match InterMask)")
    parser.add_argument("--gain-scale", type=float, default=1.0,
                        help="Scale PD gains (pd method only)")
    parser.add_argument("--save", action="store_true",
                        help="Save torques to .npy file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--gpu", default="cuda:0")
    args = parser.parse_args()

    if args.dataset:
        args.data_dir = f"data/retargeted_v2/{args.dataset}"

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.isdir(data_dir):
        if os.path.isdir(args.data_dir):
            data_dir = args.data_dir
        else:
            print(f"ERROR: data directory not found: {data_dir}")
            sys.exit(1)

    # ── Output directory ─────────────────────────────────
    if args.output_dir:
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
        if not os.path.isabs(args.output_dir) and not os.path.isdir(output_dir):
            output_dir = args.output_dir
    elif args.dataset:
        output_dir = os.path.join(
            PROJECT_ROOT, "data", "compute_torques", args.dataset
        )
    else:
        output_dir = os.path.join(
            PROJECT_ROOT, "data", "compute_torques", "default"
        )

    if args.save:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # ── Determine clips to process ───────────────────────
    if args.clip:
        clips = [args.clip]
    elif args.dataset:
        clips = list_retargeted_clips(data_dir)
        print(f"Found {len(clips)} retargeted clips in {data_dir}")
    else:
        print("ERROR: specify --clip or --dataset")
        sys.exit(1)

    # ── Determine persons ────────────────────────────────
    persons = [args.person] if args.person is not None else [0, 1]

    # ── Build task list ──────────────────────────────────
    tasks = []
    total_skipped = 0
    for clip_id in clips:
        for p_idx in persons:
            if args.save and not args.force:
                suffix = "_torques_solo" if args.method == "inverse" else "_torques_pd"
                out_path = os.path.join(
                    output_dir, f"{clip_id}_person{p_idx}{suffix}.npy"
                )
                if os.path.exists(out_path):
                    total_skipped += 1
                    continue
            tasks.append((
                clip_id, p_idx, data_dir, output_dir, args.method,
                args.fps, args.gain_scale, args.gpu, args.save,
                args.downsample, args.diff_method,
            ))

    print(f"Tasks: {len(tasks)} to process, {total_skipped} skipped (existing)")

    # ── Process ──────────────────────────────────────────
    total_processed = 0
    total_errors = 0
    t_start = time.time()

    if args.workers > 1:
        # Multiprocessing mode
        ctx = mp.get_context("spawn")
        print(f"Using {args.workers} worker processes")
        with ctx.Pool(args.workers, initializer=_worker_init) as pool:
            for i, (clip_id, p_idx, success) in enumerate(
                pool.imap_unordered(_worker_process_clip, tasks)
            ):
                elapsed = time.time() - t_start
                done = total_processed + total_errors + 1
                rate = done / elapsed * 3600 if elapsed > 0 else 0
                eta_h = (len(tasks) - done) / rate if rate > 0 else 0

                if success:
                    total_processed += 1
                    print(f"  [{done}/{len(tasks)}] {clip_id} p{p_idx} OK "
                          f"({rate:.0f}/h, ETA {eta_h:.1f}h)")
                else:
                    total_errors += 1
                    print(f"  [{done}/{len(tasks)}] {clip_id} p{p_idx} FAILED")
    else:
        # Sequential mode
        for i, task_args in enumerate(tasks):
            clip_id, p_idx = task_args[0], task_args[1]
            elapsed = time.time() - t_start
            rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
            eta_h = ((len(tasks) - i) / rate) if rate > 0 else 0

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(tasks)}] Clip: {clip_id}, Person: {p_idx}, "
                  f"Method: {args.method} "
                  f"({rate:.0f}/h, ETA {eta_h:.1f}h)")
            print(f"{'='*60}")

            try:
                result = process_clip(
                    clip_id, p_idx, data_dir, output_dir, args.method,
                    args.fps, args.gain_scale, args.gpu, args.save,
                    args.downsample, args.diff_method,
                )
                if result is not None:
                    total_processed += 1
                else:
                    total_errors += 1
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                total_errors += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE: {total_processed} processed, {total_skipped} skipped, "
          f"{total_errors} errors")
    print(f"Time: {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
