"""
paired_simulation.py — Paired vs Solo PD simulation for interaction analysis.

Simulates two-person motions in three scenarios:
  1. Paired:  Both persons in the same Newton scene (inter-body contacts)
  2. Solo A:  Person A alone (no person B)
  3. Solo B:  Person B alone (no person A)

Comparing torques across these scenarios reveals whether the interaction
is physically consistent.  For physically plausible interactions, the
torque difference (paired - solo) should reflect real contact forces.
For generated motions, inconsistencies indicate physical implausibility.

Uses existing building blocks:
  - prepare2/pd_utils.py:   build_model, setup_model_properties,
                            create_mujoco_solver, build_pd_gains,
                            compute_all_pd_torques_np, init_state
  - prepare4/dynamics.py:   set_segment_masses_multi, constants
  - prepare4/run_full_analysis.py:  pd_forward_torques (solo sim)

Usage:
    from prepare4.paired_simulation import compute_paired_vs_solo

    result = compute_paired_vs_solo(joint_q_A, joint_q_B, betas_A, betas_B)
    # result['torques_paired_A']  (T, 75)
    # result['torques_solo_A']    (T, 75)
    # result['torque_delta_A']    (T, 75)
    # ...
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare4.dynamics import (
    set_segment_masses_multi,
    DEFAULT_BODY_MASS_KG, N_JOINT_Q, N_JOINT_QD,
)

# Re-export constants used across pipeline
FPS = 30
DT = 1.0 / FPS
SIM_FREQ = 480
SIM_SUBSTEPS = SIM_FREQ // FPS  # 16

# Per-person DOF/coordinate counts (from prepare2/pd_utils.py)
DOFS_PER_PERSON = 75
COORDS_PER_PERSON = 76


def pd_forward_torques_paired(joint_q_A, joint_q_B, betas_A, betas_B,
                               dt=DT, device="cuda:0", verbose=False,
                               settle_frames=15, total_mass=DEFAULT_BODY_MASS_KG):
    """Simulate both persons in the same Newton scene with inter-body contacts.

    Runs a PD-tracked forward simulation at 480 Hz with MuJoCo solver.
    Both persons share the same collision world — inter-person contacts
    (e.g., hand-to-body pushes) generate reaction forces automatically.

    PD torques are computed on CPU via compute_all_pd_torques_np() to
    correctly handle multi-person DOF/coordinate offsets.

    Args:
        joint_q_A: (T, 76) reference joint coordinates for person A
        joint_q_B: (T, 76) reference joint coordinates for person B
        betas_A: (10,) SMPL-X shape parameters for person A
        betas_B: (10,) SMPL-X shape parameters for person B
        dt: timestep (1/fps), default 1/30
        device: CUDA device string
        verbose: print progress
        settle_frames: frames to hold initial pose for contact establishment
        total_mass: body mass per person in kg

    Returns:
        torques_A: (T, 75) average PD torques per frame for person A
        torques_B: (T, 75) average PD torques per frame for person B
        sim_jq_A:  (T, 76) simulated joint coordinates for person A
        sim_jq_B:  (T, 76) simulated joint coordinates for person B
    """
    import warp as wp
    import newton
    from prepare2.pd_utils import (
        build_model, setup_model_properties, create_mujoco_solver,
        build_pd_gains, compute_all_pd_torques_np, init_state,
        DEFAULT_TORQUE_LIMIT,
    )

    T = min(joint_q_A.shape[0], joint_q_B.shape[0])
    jq_A = joint_q_A[:T].astype(np.float32)
    jq_B = joint_q_B[:T].astype(np.float32)

    fps = round(1.0 / dt)
    sim_steps = SIM_FREQ // fps
    dt_sim = 1.0 / SIM_FREQ

    # Build 2-person model
    model, _ = build_model([betas_A, betas_B], device=device, with_ground=True)
    setup_model_properties(model, n_persons=2, device=device)
    set_segment_masses_multi(model, n_persons=2, total_mass=total_mass,
                             verbose=verbose)

    kp, kd = build_pd_gains(model, n_persons=2)
    n_dof = model.joint_dof_count

    solver = create_mujoco_solver(model, n_persons=2)
    state_0, state_1, control = init_state(model, [jq_A, jq_B],
                                            n_persons=2, device=device)

    # Output buffers
    torques_out = np.zeros((T, n_dof), dtype=np.float32)
    sim_jq = np.zeros((T, 2 * COORDS_PER_PERSON), dtype=np.float32)
    sim_jq[0, :COORDS_PER_PERSON] = jq_A[0]
    sim_jq[0, COORDS_PER_PERSON:] = jq_B[0]

    inv_steps = 1.0 / float(sim_steps)

    if verbose:
        print(f"  Paired PD sim: {T} frames, {sim_steps} substeps/frame, "
              f"{SIM_FREQ} Hz, settle={settle_frames}")

    # Settle phase
    if settle_frames > 0:
        for sf in range(settle_frames):
            for sub in range(sim_steps):
                cq = state_0.joint_q.numpy()
                cqd = state_0.joint_qd.numpy()
                tau = compute_all_pd_torques_np(
                    cq, cqd, [jq_A, jq_B], 0, kp, kd,
                    n_persons=2, torque_limit=DEFAULT_TORQUE_LIMIT,
                )
                control.joint_f = wp.array(tau, dtype=wp.float32, device=device)
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

        if verbose:
            cq = state_0.joint_q.numpy()
            err_A = np.linalg.norm(cq[:3] - jq_A[0, :3])
            err_B = np.linalg.norm(cq[COORDS_PER_PERSON:COORDS_PER_PERSON + 3]
                                   - jq_B[0, :3])
            print(f"    Settle done: pos_err A={err_A*100:.1f}cm B={err_B*100:.1f}cm")

    # Main simulation loop
    for t in range(1, T):
        tau_accum = np.zeros(n_dof, dtype=np.float32)

        for sub in range(sim_steps):
            cq = state_0.joint_q.numpy()
            cqd = state_0.joint_qd.numpy()
            tau = compute_all_pd_torques_np(
                cq, cqd, [jq_A, jq_B], t, kp, kd,
                n_persons=2, torque_limit=DEFAULT_TORQUE_LIMIT,
            )
            tau_accum += tau
            control.joint_f = wp.array(tau, dtype=wp.float32, device=device)
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt_sim)
            state_0, state_1 = state_1, state_0

        torques_out[t] = tau_accum * inv_steps
        cq_np = state_0.joint_q.numpy()
        sim_jq[t, :COORDS_PER_PERSON] = cq_np[:COORDS_PER_PERSON]
        sim_jq[t, COORDS_PER_PERSON:] = cq_np[COORDS_PER_PERSON:2 * COORDS_PER_PERSON]

        if verbose and t % 50 == 0:
            tau_A = torques_out[t, :DOFS_PER_PERSON]
            tau_B = torques_out[t, DOFS_PER_PERSON:2 * DOFS_PER_PERSON]
            print(f"    Frame {t}/{T}: "
                  f"|τ_hinge_A|={np.abs(tau_A[6:]).mean():.1f} Nm  "
                  f"|τ_hinge_B|={np.abs(tau_B[6:]).mean():.1f} Nm")

    # Split into per-person arrays
    torques_A = torques_out[:, :DOFS_PER_PERSON]
    torques_B = torques_out[:, DOFS_PER_PERSON:2 * DOFS_PER_PERSON]
    sim_jq_A = sim_jq[:, :COORDS_PER_PERSON]
    sim_jq_B = sim_jq[:, COORDS_PER_PERSON:]

    if verbose:
        print(f"  Paired done. "
              f"|τ_hinge_A| mean={np.abs(torques_A[:, 6:]).mean():.1f} Nm  "
              f"|τ_hinge_B| mean={np.abs(torques_B[:, 6:]).mean():.1f} Nm  "
              f"|root_A| mean={np.linalg.norm(torques_A[:, :3], axis=-1).mean():.0f} N  "
              f"|root_B| mean={np.linalg.norm(torques_B[:, :3], axis=-1).mean():.0f} N")

    return torques_A, torques_B, sim_jq_A, sim_jq_B


def compute_paired_vs_solo(joint_q_A, joint_q_B, betas_A, betas_B,
                            dt=DT, device="cuda:0", verbose=True,
                            settle_frames=15, total_mass=DEFAULT_BODY_MASS_KG):
    """Run all 3 scenarios (paired + 2 solo) and compute deltas.

    Args:
        joint_q_A: (T, 76) reference trajectory for person A
        joint_q_B: (T, 76) reference trajectory for person B
        betas_A: (10,) SMPL-X shape parameters for person A
        betas_B: (10,) SMPL-X shape parameters for person B
        dt: timestep (1/fps)
        device: CUDA device
        verbose: print progress
        settle_frames: settle frames per simulation
        total_mass: body mass per person in kg

    Returns:
        dict with:
          torques_paired_A/B:    (T, 75) torques with both persons
          torques_solo_A/B:      (T, 75) torques with only that person
          sim_jq_paired_A/B:     (T, 76) simulated coords (paired)
          sim_jq_solo_A/B:       (T, 76) simulated coords (solo)
          torque_delta_A/B:      (T, 75) paired - solo difference
          root_force_paired_A/B: (T, 6) root DOFs from paired sim
          root_force_solo_A/B:   (T, 6) root DOFs from solo sim
          root_force_delta_A/B:  (T, 6) paired - solo root force diff
    """
    from prepare4.run_full_analysis import pd_forward_torques

    T = min(joint_q_A.shape[0], joint_q_B.shape[0])
    jq_A = joint_q_A[:T]
    jq_B = joint_q_B[:T]

    # Scenario 1: Paired simulation
    if verbose:
        print("\n[Scenario 1] Paired simulation (both persons)...")
    tau_paired_A, tau_paired_B, sjq_paired_A, sjq_paired_B = \
        pd_forward_torques_paired(
            jq_A, jq_B, betas_A, betas_B,
            dt=dt, device=device, verbose=verbose,
            settle_frames=settle_frames, total_mass=total_mass,
        )

    # Scenario 2: Solo A (no person B)
    if verbose:
        print("\n[Scenario 2] Solo simulation (person A only)...")
    tau_solo_A, sjq_solo_A = pd_forward_torques(
        jq_A, betas_A, dt=dt, device=device, verbose=verbose,
        settle_frames=settle_frames,
    )

    # Scenario 3: Solo B (no person A)
    if verbose:
        print("\n[Scenario 3] Solo simulation (person B only)...")
    tau_solo_B, sjq_solo_B = pd_forward_torques(
        jq_B, betas_B, dt=dt, device=device, verbose=verbose,
        settle_frames=settle_frames,
    )

    # Compute deltas
    delta_A = tau_paired_A - tau_solo_A
    delta_B = tau_paired_B - tau_solo_B

    return {
        # Paired
        'torques_paired_A': tau_paired_A,
        'torques_paired_B': tau_paired_B,
        'sim_jq_paired_A': sjq_paired_A,
        'sim_jq_paired_B': sjq_paired_B,
        # Solo
        'torques_solo_A': tau_solo_A,
        'torques_solo_B': tau_solo_B,
        'sim_jq_solo_A': sjq_solo_A,
        'sim_jq_solo_B': sjq_solo_B,
        # Deltas
        'torque_delta_A': delta_A,
        'torque_delta_B': delta_B,
        # Root forces (DOFs 0:6 = translational + rotational)
        'root_force_paired_A': tau_paired_A[:, :6],
        'root_force_paired_B': tau_paired_B[:, :6],
        'root_force_solo_A': tau_solo_A[:, :6],
        'root_force_solo_B': tau_solo_B[:, :6],
        'root_force_delta_A': delta_A[:, :6],
        'root_force_delta_B': delta_B[:, :6],
    }
