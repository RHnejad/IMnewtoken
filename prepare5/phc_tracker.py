"""
phc_tracker.py — PHC-style physics-based motion tracker using Newton.

Implements the core tracking pipeline inspired by PHC (Perpetual Humanoid
Controller) but using Newton's built-in PD actuators instead of IsaacGym's.

Architecture:
  1. Newton's built-in PD (joint_target_ke/kd + control.joint_target_pos)
     handles hinge DOF tracking — equivalent to IsaacGym's
     set_dof_position_target_tensor.
  2. Custom root PD via control.joint_f handles root position/orientation
     (FREE joints skip Newton's built-in PD).
  3. MuJoCo solver provides ground contacts, gravity, friction.

Key difference from old prepare4/ approach:
  - Old: computed explicit PD torques externally → control.joint_f (all DOFs)
  - New: Newton's built-in PD computes hinge torques internally, only root
         DOFs use external forces. This is closer to how IsaacGym/PHC works.

Usage:
    from prepare5.phc_tracker import PHCTracker

    tracker = PHCTracker(device="cuda:0")
    result = tracker.track(joint_q, betas)
    # result['sim_positions']   (T, 22, 3)
    # result['ref_positions']   (T, 22, 3)
    # result['rewards']         (T,)
    # result['mpjpe_mm']        float
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare5.phc_config import (
    FPS, DT, SIM_FREQ, SIM_SUBSTEPS,
    COORDS_PER_PERSON, DOFS_PER_PERSON, BODIES_PER_PERSON,
    PHC_BODY_GAINS, OLD_BODY_GAINS,
    ROOT_POS_KP, ROOT_POS_KD, ROOT_ROT_KP, ROOT_ROT_KD,
    OLD_ROOT_POS_KP, OLD_ROOT_POS_KD, OLD_ROOT_ROT_KP, OLD_ROOT_ROT_KD,
    TORQUE_LIMIT, ARMATURE_HINGE, ARMATURE_ROOT,
    DEFAULT_BODY_MASS_KG, SETTLE_FRAMES,
    SMPL_TO_NEWTON, N_SMPL_JOINTS, BODY_NAMES,
    TERMINATION_DISTANCE, MIN_HEIGHT,
)
from prepare5.phc_reward import (
    extract_body_state, compute_imitation_reward, compute_tracking_errors,
)


class PHCTracker:
    """Physics-based motion tracker using Newton + PHC-style PD targets.

    Simulates a humanoid tracking a reference motion via PD control,
    producing the closest physically feasible motion. The difference
    between reference and simulated motion quantifies physical plausibility.
    """

    def __init__(self, device="cuda:0", total_mass=DEFAULT_BODY_MASS_KG,
                 settle_frames=SETTLE_FRAMES, verbose=True,
                 gain_scale=1.0, use_builtin_pd=False,
                 gain_preset="phc", foot_geom="sphere",
                 root_mode="free"):
        """
        Args:
            device: CUDA device
            total_mass: body mass in kg
            settle_frames: frames to hold initial pose for contact establishment
            verbose: print progress
            gain_scale: multiplier for PD gains (1.0 = default for preset)
            use_builtin_pd: if True, use Newton's built-in PD for hinges;
                           if False, use explicit PD via control.joint_f (default)
            gain_preset: "phc" for PHC-matched gains, "old" for prepare2 gains
            foot_geom: "box" | "sphere" | "capsule" — foot collision geometry
            root_mode: how to handle root (pelvis) forces:
                "free"    — no root forces; character must self-balance via
                            joint torques + ground contacts (like PHC)
                "orient"  — orientation PD only; no positional skyhook
                "skyhook" — full position + orientation PD (old behavior)
        """
        self.device = device
        self.total_mass = total_mass
        self.settle_frames = settle_frames
        self.verbose = verbose
        self.gain_scale = gain_scale
        self.use_builtin_pd = use_builtin_pd
        self.gain_preset = gain_preset
        self.foot_geom = foot_geom
        self.root_mode = root_mode

        if gain_preset == "old":
            self._body_gains = OLD_BODY_GAINS
            self._root_pos_kp = OLD_ROOT_POS_KP
            self._root_pos_kd = OLD_ROOT_POS_KD
            self._root_rot_kp = OLD_ROOT_ROT_KP
            self._root_rot_kd = OLD_ROOT_ROT_KD
        else:  # "phc"
            self._body_gains = PHC_BODY_GAINS
            self._root_pos_kp = ROOT_POS_KP
            self._root_pos_kd = ROOT_POS_KD
            self._root_rot_kp = ROOT_ROT_KP
            self._root_rot_kd = ROOT_ROT_KD

    def track(self, joint_q, betas, dt=DT):
        """Track a single-person reference motion through physics simulation.

        This is the main entry point. Given a reference trajectory (joint_q),
        simulates the humanoid using PHC-style PD targets and returns the
        physically-corrected trajectory along with tracking metrics.

        Args:
            joint_q: (T, 76) reference joint coordinates
            betas: (10,) SMPL-X shape parameters
            dt: timestep (1/fps)

        Returns:
            dict with:
              sim_positions:   (T, 22, 3) simulated body positions
              ref_positions:   (T, 22, 3) reference body positions
              sim_joint_q:     (T, 76) simulated joint coordinates
              rewards:         (T,) per-frame PHC reward
              reward_components: list of per-frame component dicts
              mpjpe_mm:        mean per-joint position error in mm
              per_frame_mpjpe: (T,) per-frame MPJPE in mm
              torques:         (T, 75) average torques per frame
              terminated_at:   frame where termination would occur (or T)
              root_drift_m:    (T,) root position error in meters
        """
        import warp as wp
        import newton

        T = joint_q.shape[0]
        jq = joint_q.astype(np.float32)
        fps = round(1.0 / dt)
        sim_steps = SIM_FREQ // fps
        dt_sim = 1.0 / SIM_FREQ

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PHC Tracker: {T} frames @ {fps} fps, {SIM_FREQ} Hz physics")
            print(f"  PD gains: {self.gain_preset} × {self.gain_scale}")
            print(f"  Root mode: {self.root_mode}")
            print(f"  Foot geom: {self.foot_geom}")
            print(f"  Device: {self.device}")
            print(f"{'='*60}")

        # ─── Build model ───
        model = self._build_model(betas)

        # ─── Setup PD gains ───
        if self.use_builtin_pd:
            self._setup_pd_gains_builtin(model)
        else:
            self._setup_pd_gains_explicit(model)

        # ─── Setup solver ───
        solver = newton.solvers.SolverMuJoCo(
            model, solver="newton",
            njmax=450, nconmax=150,
            impratio=10, iterations=100, ls_iterations=50,
        )

        # ─── Initialize state ───
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        init_q = np.zeros(model.joint_coord_count, dtype=np.float32)
        init_q[:COORDS_PER_PERSON] = jq[0]
        state_0.joint_q = wp.array(init_q, dtype=wp.float32, device=self.device)
        state_0.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32,
                                     device=self.device)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        # ─── Output buffers ───
        sim_jq = np.zeros((T, COORDS_PER_PERSON), dtype=np.float32)
        sim_jq[0] = jq[0]
        sim_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        ref_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        torques_out = np.zeros((T, DOFS_PER_PERSON), dtype=np.float32)
        rewards = np.zeros(T, dtype=np.float32)
        reward_components = []
        root_drift = np.zeros(T, dtype=np.float32)

        # Compute reference positions via FK for all frames
        ref_positions = self._compute_ref_positions(model, jq)

        # Extract initial body state for frame 0
        body_pos, body_rot, body_vel, body_ang_vel = extract_body_state(state_0, 0)
        sim_positions[0] = body_pos

        # Frame 0 is initialized from reference — perfect tracking
        rewards[0] = 1.0
        reward_components.append({'r_pos': 1.0, 'r_rot': 1.0, 'r_vel': 1.0,
                                  'r_ang_vel': 1.0, 'reward': 1.0,
                                  'pos_err_m': 0.0, 'rot_err_rad': 0.0})

        # ─── Settle phase ───
        if self.settle_frames > 0 and self.verbose:
            print(f"\n  Settling: {self.settle_frames} frames...")
        self._settle(model, solver, state_0, state_1, control, jq[0],
                     dt_sim, sim_steps)
        if self.verbose:
            cq = state_0.joint_q.numpy()
            drift = np.linalg.norm(cq[:3] - jq[0, :3])
            print(f"    Settle done: root drift = {drift*100:.1f} cm")

        terminated_at = T

        # ─── Main simulation loop ───
        if self.verbose:
            print(f"\n  Tracking {T} frames...")

        for t in range(1, T):
            if self.use_builtin_pd:
                # Built-in PD: set target position for hinge DOFs
                target_pos = np.zeros(model.joint_dof_count, dtype=np.float32)
                target_pos[6:DOFS_PER_PERSON] = jq[t, 7:COORDS_PER_PERSON]
                control.joint_target_pos = wp.array(
                    target_pos, dtype=wp.float32, device=self.device
                )

            tau_accum = np.zeros(DOFS_PER_PERSON, dtype=np.float32)

            for sub in range(sim_steps):
                cq = state_0.joint_q.numpy()
                cqd = state_0.joint_qd.numpy()

                # Compute PD torques for all DOFs
                tau = self._compute_all_pd(cq, cqd, jq[t])
                tau_accum += tau[:DOFS_PER_PERSON]

                if self.use_builtin_pd:
                    # Built-in PD handles hinges; only root via joint_f
                    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
                    joint_f[:6] = tau[:6]
                else:
                    # Explicit PD for ALL DOFs via joint_f
                    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
                    joint_f[:DOFS_PER_PERSON] = tau[:DOFS_PER_PERSON]

                control.joint_f = wp.array(
                    joint_f, dtype=wp.float32, device=self.device
                )

                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

            torques_out[t] = tau_accum / sim_steps

            # Extract simulated body state
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            body_pos, body_rot, body_vel, body_ang_vel = extract_body_state(
                state_0, 0
            )
            sim_positions[t] = body_pos

            # Store simulated joint coordinates
            cq_np = state_0.joint_q.numpy()
            sim_jq[t] = cq_np[:COORDS_PER_PERSON]

            # Root drift
            root_drift[t] = np.linalg.norm(cq_np[:3] - jq[t, :3])

            # Compute reward (approximate — using position data,
            # ref rotations/velocities not computed for efficiency)
            pos_err2 = np.sum((ref_positions[t] - body_pos) ** 2,
                              axis=-1).mean()
            r_pos = np.exp(-100.0 * pos_err2)
            rewards[t] = r_pos  # Simplified — position-only reward
            reward_components.append({
                'r_pos': float(r_pos),
                'pos_err_m': float(np.sqrt(pos_err2)),
            })

            # Check termination conditions
            if root_drift[t] > TERMINATION_DISTANCE:
                if terminated_at == T:
                    terminated_at = t
                    if self.verbose:
                        print(f"    ! Termination at frame {t}: "
                              f"root drift {root_drift[t]*100:.1f} cm > "
                              f"{TERMINATION_DISTANCE*100:.0f} cm")

            pelvis_z = cq_np[2]
            if pelvis_z < MIN_HEIGHT:
                if terminated_at == T:
                    terminated_at = t
                    if self.verbose:
                        print(f"    ! Termination at frame {t}: "
                              f"height {pelvis_z:.2f} m < {MIN_HEIGHT} m")

            # Progress
            if self.verbose and t % 30 == 0:
                drift_cm = root_drift[t] * 100
                mpjpe = np.linalg.norm(
                    sim_positions[t] - ref_positions[t], axis=-1
                ).mean() * 1000
                print(f"    Frame {t:4d}/{T}: drift={drift_cm:5.1f}cm  "
                      f"MPJPE={mpjpe:6.1f}mm  r_pos={rewards[t]:.3f}")

        # ─── Compute final metrics ───
        tracking_errors = compute_tracking_errors(sim_positions, ref_positions)

        if self.verbose:
            print(f"\n  {'='*40}")
            print(f"  Tracking complete:")
            print(f"    MPJPE:        {tracking_errors['mpjpe_mm']:.1f} mm")
            print(f"    Max error:    {tracking_errors['max_error_mm']:.1f} mm")
            print(f"    Mean reward:  {rewards[1:].mean():.3f}")
            print(f"    Root drift:   {root_drift.mean()*100:.1f} cm (mean), "
                  f"{root_drift.max()*100:.1f} cm (max)")
            print(f"    Terminated:   "
                  f"{'frame ' + str(terminated_at) if terminated_at < T else 'No'}")
            print(f"  {'='*40}\n")

        return {
            'sim_positions': sim_positions,
            'ref_positions': ref_positions,
            'sim_joint_q': sim_jq,
            'rewards': rewards,
            'reward_components': reward_components,
            'mpjpe_mm': tracking_errors['mpjpe_mm'],
            'per_frame_mpjpe_mm': tracking_errors['per_frame_mpjpe_mm'],
            'per_joint_mpjpe_mm': tracking_errors['per_joint_mpjpe_mm'],
            'max_error_mm': tracking_errors['max_error_mm'],
            'torques': torques_out,
            'terminated_at': terminated_at,
            'root_drift_m': root_drift,
        }

    def track_paired(self, joint_q_A, joint_q_B, betas_A, betas_B, dt=DT):
        """Track a two-person interaction through physics simulation.

        Same as track() but with two persons in the same scene, enabling
        inter-body contacts. This produces physically-corrected motions
        where contact forces are properly resolved.

        Args:
            joint_q_A: (T, 76) reference for person A
            joint_q_B: (T, 76) reference for person B
            betas_A, betas_B: (10,) shape parameters
            dt: timestep

        Returns:
            dict with per-person results (keys suffixed with _A, _B)
        """
        import warp as wp
        import newton

        T = min(joint_q_A.shape[0], joint_q_B.shape[0])
        jq_A = joint_q_A[:T].astype(np.float32)
        jq_B = joint_q_B[:T].astype(np.float32)
        fps = round(1.0 / dt)
        sim_steps = SIM_FREQ // fps
        dt_sim = 1.0 / SIM_FREQ

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PHC Tracker (Paired): {T} frames @ {fps} fps")
            print(f"{'='*60}")

        # Build 2-person model
        model = self._build_model_multi([betas_A, betas_B])
        self._setup_pd_gains_multi(model, n_persons=2)

        solver = newton.solvers.SolverMuJoCo(
            model, solver="newton",
            njmax=900, nconmax=300,
            impratio=10, iterations=100, ls_iterations=50,
        )

        # Initialize state
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        n_coords = model.joint_coord_count
        n_dof = model.joint_dof_count
        init_q = np.zeros(n_coords, dtype=np.float32)
        init_q[:COORDS_PER_PERSON] = jq_A[0]
        init_q[COORDS_PER_PERSON:2 * COORDS_PER_PERSON] = jq_B[0]
        state_0.joint_q = wp.array(init_q, dtype=wp.float32, device=self.device)
        state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=self.device)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        # Output buffers
        sim_positions_A = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        sim_positions_B = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        sim_jq = np.zeros((T, 2 * COORDS_PER_PERSON), dtype=np.float32)
        sim_jq[0, :COORDS_PER_PERSON] = jq_A[0]
        sim_jq[0, COORDS_PER_PERSON:] = jq_B[0]
        torques_out = np.zeros((T, n_dof), dtype=np.float32)
        root_drift_A = np.zeros(T, dtype=np.float32)
        root_drift_B = np.zeros(T, dtype=np.float32)

        # Compute reference positions
        ref_pos_A = self._compute_ref_positions_single(model, jq_A, person_idx=0)
        ref_pos_B = self._compute_ref_positions_single(model, jq_B, person_idx=1)

        # Extract initial positions
        body_pos_A, _, _, _ = extract_body_state(state_0, 0)
        body_pos_B, _, _, _ = extract_body_state(state_0, 1)
        sim_positions_A[0] = body_pos_A
        sim_positions_B[0] = body_pos_B

        # Settle
        if self.settle_frames > 0:
            if self.verbose:
                print(f"  Settling {self.settle_frames} frames...")
            self._settle_multi(model, solver, state_0, state_1, control,
                               jq_A[0], jq_B[0], dt_sim, sim_steps)

        # Main loop
        if self.verbose:
            print(f"  Tracking {T} frames (paired)...")

        for t in range(1, T):
            for sub in range(sim_steps):
                cq = state_0.joint_q.numpy()
                cqd = state_0.joint_qd.numpy()
                tau_A = self._compute_all_pd(cq, cqd, jq_A[t], person_idx=0)
                tau_B = self._compute_all_pd(cq, cqd, jq_B[t], person_idx=1)
                joint_f = np.zeros(n_dof, dtype=np.float32)
                joint_f[:DOFS_PER_PERSON] = tau_A
                joint_f[DOFS_PER_PERSON:2 * DOFS_PER_PERSON] = tau_B
                control.joint_f = wp.array(
                    joint_f, dtype=wp.float32, device=self.device
                )
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

            # Extract results
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            body_pos_A, _, _, _ = extract_body_state(state_0, 0)
            body_pos_B, _, _, _ = extract_body_state(state_0, 1)
            sim_positions_A[t] = body_pos_A
            sim_positions_B[t] = body_pos_B

            cq_np = state_0.joint_q.numpy()
            sim_jq[t, :COORDS_PER_PERSON] = cq_np[:COORDS_PER_PERSON]
            sim_jq[t, COORDS_PER_PERSON:] = cq_np[COORDS_PER_PERSON:2 * COORDS_PER_PERSON]

            root_drift_A[t] = np.linalg.norm(cq_np[:3] - jq_A[t, :3])
            root_drift_B[t] = np.linalg.norm(
                cq_np[COORDS_PER_PERSON:COORDS_PER_PERSON + 3] - jq_B[t, :3]
            )

            if self.verbose and t % 30 == 0:
                mpjpe_A = np.linalg.norm(
                    sim_positions_A[t] - ref_pos_A[t], axis=-1).mean() * 1000
                mpjpe_B = np.linalg.norm(
                    sim_positions_B[t] - ref_pos_B[t], axis=-1).mean() * 1000
                print(f"    Frame {t:4d}/{T}: "
                      f"A drift={root_drift_A[t]*100:.1f}cm MPJPE={mpjpe_A:.0f}mm  "
                      f"B drift={root_drift_B[t]*100:.1f}cm MPJPE={mpjpe_B:.0f}mm")

        # Compute metrics
        errors_A = compute_tracking_errors(sim_positions_A, ref_pos_A)
        errors_B = compute_tracking_errors(sim_positions_B, ref_pos_B)

        if self.verbose:
            print(f"\n  Paired tracking complete:")
            print(f"    Person A: MPJPE={errors_A['mpjpe_mm']:.1f}mm, "
                  f"root drift={root_drift_A.mean()*100:.1f}cm")
            print(f"    Person B: MPJPE={errors_B['mpjpe_mm']:.1f}mm, "
                  f"root drift={root_drift_B.mean()*100:.1f}cm")

        return {
            'sim_positions_A': sim_positions_A,
            'sim_positions_B': sim_positions_B,
            'ref_positions_A': ref_pos_A,
            'ref_positions_B': ref_pos_B,
            'sim_joint_q': sim_jq,
            'mpjpe_A_mm': errors_A['mpjpe_mm'],
            'mpjpe_B_mm': errors_B['mpjpe_mm'],
            'root_drift_A_m': root_drift_A,
            'root_drift_B_m': root_drift_B,
            'per_frame_mpjpe_A_mm': errors_A['per_frame_mpjpe_mm'],
            'per_frame_mpjpe_B_mm': errors_B['per_frame_mpjpe_mm'],
        }

    # ═══════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _build_model(self, betas):
        """Build single-person Newton model."""
        import warp as wp
        import newton
        from prepare4.gen_xml import get_or_create_xml
        from prepare4.dynamics import set_segment_masses

        xml_path = get_or_create_xml(betas, foot_geom=self.foot_geom)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)

        set_segment_masses(model, total_mass=self.total_mass, verbose=False)
        self._setup_model_properties(model, n_persons=1)

        return model

    def _build_model_multi(self, betas_list):
        """Build multi-person Newton model."""
        import warp as wp
        import newton
        from prepare4.gen_xml import get_or_create_xml
        from prepare4.dynamics import set_segment_masses_multi

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for betas in betas_list:
            xml_path = get_or_create_xml(betas, foot_geom=self.foot_geom)
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)

        n_persons = len(betas_list)
        set_segment_masses_multi(model, n_persons=n_persons,
                                 total_mass=self.total_mass, verbose=False)
        self._setup_model_properties(model, n_persons=n_persons)

        return model

    def _setup_model_properties(self, model, n_persons):
        """Set armature, disable passive springs and joint limits."""
        import warp as wp

        n_dof = model.joint_dof_count

        # Disable passive springs
        model.mujoco.dof_passive_stiffness.fill_(0.0)
        model.mujoco.dof_passive_damping.fill_(0.0)

        # Disable joint limits — prevents constraint forces when IK-solved
        # angles lie outside MJCF-defined ranges (critical for generated data)
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        joint_limit_lower[:] = -1e6
        joint_limit_upper[:] = 1e6
        model.joint_limit_lower = wp.array(
            joint_limit_lower, dtype=wp.float32, device=self.device
        )
        model.joint_limit_upper = wp.array(
            joint_limit_upper, dtype=wp.float32, device=self.device
        )

        # Armature for numerical stability
        arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
        for i in range(n_persons):
            off = i * DOFS_PER_PERSON
            arm[off:off + 6] = ARMATURE_ROOT
        model.joint_armature = wp.array(arm, dtype=wp.float32, device=self.device)

    def _setup_pd_gains_explicit(self, model):
        """Set up PD gains for explicit PD (all DOFs via joint_f).

        Disables Newton's built-in PD. All torques computed externally.
        Uses self._body_gains / self._root_*_k* from the selected preset.
        """
        import warp as wp

        n_dof = model.joint_dof_count

        # Disable built-in PD
        model.joint_target_ke.fill_(0.0)
        model.joint_target_kd.fill_(0.0)

        # Build per-DOF gain arrays for explicit PD
        kp = np.zeros(n_dof, dtype=np.float32)
        kd = np.zeros(n_dof, dtype=np.float32)

        # Root DOFs
        kp[:3] = self._root_pos_kp * self.gain_scale
        kd[:3] = self._root_pos_kd * self.gain_scale
        kp[3:6] = self._root_rot_kp * self.gain_scale
        kd[3:6] = self._root_rot_kd * self.gain_scale

        # Hinge DOFs (6:75)
        for b_idx in range(23):
            s = 6 + b_idx * 3
            body_name = BODY_NAMES[1 + b_idx]
            kp_val, kd_val = self._body_gains.get(body_name, (300, 30))
            kp[s:s + 3] = kp_val * self.gain_scale
            kd[s:s + 3] = kd_val * self.gain_scale

        self.kp_np = kp
        self.kd_np = kd

        if self.verbose:
            preset = self.gain_preset.upper()
            print(f"  PD gains set (explicit, {preset} preset × {self.gain_scale}):")
            print(f"    Hip={self._body_gains['L_Hip']}, "
                  f"Knee={self._body_gains['L_Knee']}, "
                  f"Torso={self._body_gains['Torso']}, "
                  f"Shoulder={self._body_gains['L_Shoulder']}")
            print(f"    Root: pos_kp={self._root_pos_kp}, "
                  f"rot_kp={self._root_rot_kp}")

    def _setup_pd_gains_builtin(self, model):
        """Set up PD gains using Newton's built-in PD for hinges."""
        import warp as wp

        n_dof = model.joint_dof_count
        ke = np.zeros(n_dof, dtype=np.float32)
        kd_arr = np.zeros(n_dof, dtype=np.float32)

        for b_idx in range(23):
            s = 6 + b_idx * 3
            body_name = BODY_NAMES[1 + b_idx]
            kp_val, kd_val = self._body_gains.get(body_name, (300, 30))
            ke[s:s + 3] = kp_val * self.gain_scale
            kd_arr[s:s + 3] = kd_val * self.gain_scale

        model.joint_target_ke = wp.array(ke, dtype=wp.float32, device=self.device)
        model.joint_target_kd = wp.array(kd_arr, dtype=wp.float32, device=self.device)

        kp = np.zeros(n_dof, dtype=np.float32)
        kd = np.zeros(n_dof, dtype=np.float32)
        kp[:3] = self._root_pos_kp * self.gain_scale
        kd[:3] = self._root_pos_kd * self.gain_scale
        kp[3:6] = self._root_rot_kp * self.gain_scale
        kd[3:6] = self._root_rot_kd * self.gain_scale
        kp[6:DOFS_PER_PERSON] = ke[6:DOFS_PER_PERSON]
        kd[6:DOFS_PER_PERSON] = kd_arr[6:DOFS_PER_PERSON]

        self.kp_np = kp
        self.kd_np = kd

        if self.verbose:
            preset = self.gain_preset.upper()
            print(f"  PD gains set (built-in, {preset} preset):")
            print(f"    Hip={self._body_gains['L_Hip']}, "
                  f"Torso={self._body_gains['Torso']}")
            print(f"    Root: pos_kp={self._root_pos_kp}")

    def _setup_pd_gains_multi(self, model, n_persons):
        """Set up PD gains for multi-person model (explicit PD)."""
        import warp as wp

        n_dof = model.joint_dof_count

        model.joint_target_ke.fill_(0.0)
        model.joint_target_kd.fill_(0.0)

        kp = np.zeros(n_dof, dtype=np.float32)
        kd = np.zeros(n_dof, dtype=np.float32)

        for p in range(n_persons):
            off = p * DOFS_PER_PERSON
            kp[off:off + 3] = self._root_pos_kp * self.gain_scale
            kd[off:off + 3] = self._root_pos_kd * self.gain_scale
            kp[off + 3:off + 6] = self._root_rot_kp * self.gain_scale
            kd[off + 3:off + 6] = self._root_rot_kd * self.gain_scale
            for b_idx in range(23):
                s = off + 6 + b_idx * 3
                body_name = BODY_NAMES[1 + b_idx]
                kp_val, kd_val = self._body_gains.get(body_name, (300, 30))
                kp[s:s + 3] = kp_val * self.gain_scale
                kd[s:s + 3] = kd_val * self.gain_scale

        self.kp_np = kp
        self.kd_np = kd

    def _compute_all_pd(self, cq_full, cqd_full, ref_q, person_idx=0):
        """Compute PD torques for all DOFs of one person.

        Uses the same PD equation as PHC / the old approach:
          Root pos:    τ = kp*(ref - cur) - kd*vel
          Root rot:    τ = kp*axis_angle(q_ref * q_cur^-1) - kd*ω
          Hinges:      τ = kp*(θ_ref - θ) - kd*θ̇

        Args:
            cq_full: full joint_q array from state
            cqd_full: full joint_qd array from state
            ref_q: (76,) reference joint coordinates for this person
            person_idx: which person (0-indexed)

        Returns:
            tau: (DOFS_PER_PERSON,) torques for this person
        """
        c = person_idx * COORDS_PER_PERSON
        d = person_idx * DOFS_PER_PERSON
        kp = self.kp_np[d:d + DOFS_PER_PERSON]
        kd = self.kd_np[d:d + DOFS_PER_PERSON]

        tau = np.zeros(DOFS_PER_PERSON, dtype=np.float32)

        # Root forces depend on root_mode
        if self.root_mode == "skyhook":
            # Full position + orientation PD (artificial skyhook)
            tau[:3] = (
                kp[:3] * (ref_q[:3] - cq_full[c:c + 3])
                - kd[:3] * cqd_full[d:d + 3]
            )
            q_cur = cq_full[c + 3:c + 7].copy()
            qn = np.linalg.norm(q_cur)
            if qn > 1e-8:
                q_cur /= qn
            R_err = (
                Rotation.from_quat(ref_q[3:7])
                * Rotation.from_quat(q_cur).inv()
            ).as_rotvec()
            tau[3:6] = kp[3:6] * R_err - kd[3:6] * cqd_full[d + 3:d + 6]
        elif self.root_mode == "orient":
            # Orientation PD only — no positional skyhook
            # tau[:3] = 0 (no position forces)
            q_cur = cq_full[c + 3:c + 7].copy()
            qn = np.linalg.norm(q_cur)
            if qn > 1e-8:
                q_cur /= qn
            R_err = (
                Rotation.from_quat(ref_q[3:7])
                * Rotation.from_quat(q_cur).inv()
            ).as_rotvec()
            tau[3:6] = kp[3:6] * R_err - kd[3:6] * cqd_full[d + 3:d + 6]
        # else: root_mode == "free" — no root forces at all (tau[:6] = 0)

        # Hinge PD (23 bodies × 3 DOF = 69)
        tau[6:] = (
            kp[6:] * (ref_q[7:COORDS_PER_PERSON] - cq_full[c + 7:c + COORDS_PER_PERSON])
            - kd[6:] * cqd_full[d + 6:d + DOFS_PER_PERSON]
        )

        return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

    def _compute_root_pd_raw(self, cq, cqd, ref_q):
        """Compute root PD for one person's coordinates.

        Args:
            cq: (76,) current joint coordinates for this person
            cqd: (75,) current joint velocities for this person
            ref_q: (76,) reference joint coordinates

        Returns:
            root_tau: (6,) root forces/torques
        """
        kp_root = self.kp_np[:6]
        kd_root = self.kd_np[:6]
        tau = np.zeros(6, dtype=np.float32)

        # Position PD
        tau[:3] = kp_root[:3] * (ref_q[:3] - cq[:3]) - kd_root[:3] * cqd[:3]

        # Orientation PD: quaternion error → axis-angle
        q_cur = cq[3:7].copy()
        qn = np.linalg.norm(q_cur)
        if qn > 1e-8:
            q_cur /= qn
        R_err = (
            Rotation.from_quat(ref_q[3:7])
            * Rotation.from_quat(q_cur).inv()
        ).as_rotvec()
        tau[3:6] = kp_root[3:6] * R_err - kd_root[3:6] * cqd[3:6]

        return np.clip(tau, -TORQUE_LIMIT * 10, TORQUE_LIMIT * 10)

    def _settle(self, model, solver, state_0, state_1, control, ref_q0,
                dt_sim, sim_steps):
        """Hold initial pose to establish contacts.

        Always uses skyhook during settle regardless of root_mode,
        so the character starts in the correct pose with ground contacts.
        """
        import warp as wp

        # Temporarily use skyhook for settling
        saved_root_mode = self.root_mode
        self.root_mode = "skyhook"

        if self.use_builtin_pd:
            target_pos = np.zeros(model.joint_dof_count, dtype=np.float32)
            target_pos[6:DOFS_PER_PERSON] = ref_q0[7:COORDS_PER_PERSON]
            control.joint_target_pos = wp.array(
                target_pos, dtype=wp.float32, device=self.device
            )

        for _ in range(self.settle_frames):
            for _ in range(sim_steps):
                tau = self._compute_all_pd(
                    state_0.joint_q.numpy(),
                    state_0.joint_qd.numpy(),
                    ref_q0,
                )
                joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
                if self.use_builtin_pd:
                    joint_f[:6] = tau[:6]  # Only root via joint_f
                else:
                    joint_f[:DOFS_PER_PERSON] = tau  # All DOFs
                control.joint_f = wp.array(
                    joint_f, dtype=wp.float32, device=self.device
                )
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

        # Restore actual root_mode for main simulation
        self.root_mode = saved_root_mode

    def _settle_multi(self, model, solver, state_0, state_1, control,
                      ref_q0_A, ref_q0_B, dt_sim, sim_steps):
        """Hold initial pose for 2-person model.

        Always uses skyhook during settle regardless of root_mode.
        """
        import warp as wp

        saved_root_mode = self.root_mode
        self.root_mode = "skyhook"
        n_dof = model.joint_dof_count

        for _ in range(self.settle_frames):
            for _ in range(sim_steps):
                cq = state_0.joint_q.numpy()
                cqd = state_0.joint_qd.numpy()
                tau_A = self._compute_all_pd(cq, cqd, ref_q0_A, person_idx=0)
                tau_B = self._compute_all_pd(cq, cqd, ref_q0_B, person_idx=1)
                joint_f = np.zeros(n_dof, dtype=np.float32)
                joint_f[:DOFS_PER_PERSON] = tau_A
                joint_f[DOFS_PER_PERSON:2 * DOFS_PER_PERSON] = tau_B
                control.joint_f = wp.array(
                    joint_f, dtype=wp.float32, device=self.device
                )
                state_0.clear_forces()
                contacts = model.collide(state_0)
                solver.step(state_0, state_1, control, contacts, dt_sim)
                state_0, state_1 = state_1, state_0

        self.root_mode = saved_root_mode

    def _compute_ref_positions(self, model, joint_q):
        """Compute reference body positions for all frames via FK.

        Args:
            model: Newton model
            joint_q: (T, 76) reference joint coordinates

        Returns:
            ref_positions: (T, N_SMPL_JOINTS, 3)
        """
        import warp as wp
        import newton

        T = joint_q.shape[0]
        ref_pos = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        state = model.state()
        jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=self.device)

        for t in range(T):
            jq_full = np.zeros(model.joint_coord_count, dtype=np.float32)
            jq_full[:COORDS_PER_PERSON] = joint_q[t]
            state.joint_q = wp.array(jq_full, dtype=wp.float32, device=self.device)
            newton.eval_fk(model, state.joint_q, jqd, state)
            body_q = state.body_q.numpy().reshape(-1, 7)
            for j, b in SMPL_TO_NEWTON.items():
                ref_pos[t, j] = body_q[b, :3]

        return ref_pos

    def _compute_ref_positions_single(self, model, joint_q, person_idx=0):
        """Compute reference positions for one person in a multi-person model."""
        import warp as wp
        import newton

        T = joint_q.shape[0]
        ref_pos = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
        state = model.state()
        jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=self.device)
        n_coords = model.joint_coord_count

        off = person_idx * BODIES_PER_PERSON
        c = person_idx * COORDS_PER_PERSON

        for t in range(T):
            jq_full = np.zeros(n_coords, dtype=np.float32)
            jq_full[c:c + COORDS_PER_PERSON] = joint_q[t]
            state.joint_q = wp.array(jq_full, dtype=wp.float32, device=self.device)
            newton.eval_fk(model, state.joint_q, jqd, state)
            body_q = state.body_q.numpy().reshape(-1, 7)
            for j, b in SMPL_TO_NEWTON.items():
                ref_pos[t, j] = body_q[off + b, :3]

        return ref_pos
