"""
run_optimize.py — CLI for differentiable trajectory optimization.

Optimizes per-frame PD target residuals (Δq) through Newton's differentiable
Featherstone solver. Each iteration improves tracking.

Usage:
    # Headless optimization (fast, no viewer)
    python prepare5/run_optimize.py --clip-id 1129 --source gt

    # More epochs
    python prepare5/run_optimize.py --clip-id 1129 --source gt --epochs 100

    # Generated clip
    python prepare5/run_optimize.py --clip-id 1129 --source generated

    # Free root (character must self-balance)
    python prepare5/run_optimize.py --clip-id 1129 --source gt --root-mode free

    # After optimization, visualize in Newton viewer:
    python prepare5/visualize_newton_tracking.py \
        --result output/phc_tracker/clip_1129_gt_optimized/optimized_result.npz \
        --clip-id 1129 --source gt
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np

# Force unbuffered output so progress prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_headless(args):
    """Run optimization without viewer — pure compute."""
    from prepare5.optimize_tracking import TrackingOptimizer

    # We don't use the viewer-based TrackingOptimizer for headless.
    # Instead, run the optimization loop directly.
    from prepare5.run_phc_tracker import load_clip, retarget_person
    from prepare5.phc_config import (
        FPS, DT, COORDS_PER_PERSON, DOFS_PER_PERSON, SIM_FREQ, SIM_SUBSTEPS,
        SMPL_TO_NEWTON, N_SMPL_JOINTS, TORQUE_LIMIT,
        ARMATURE_HINGE, ARMATURE_ROOT, DEFAULT_BODY_MASS_KG,
    )

    # Optimization substeps: fewer than simulation for speed.
    # Backprop cost scales ~quadratically with substeps.
    # Default 4 substeps (120Hz) for optimization, 16 (480Hz) for final eval.
    opt_substeps = getattr(args, 'opt_substeps', 4)
    opt_dt = DT / opt_substeps  # dt per substep

    print(f"\n{'='*60}")
    print(f"Trajectory Optimization (headless)")
    print(f"  Clip: {args.clip_id} ({args.source})")
    print(f"  Epochs: {args.epochs}, Window: {args.window}, LR: {args.lr}")
    print(f"  Root mode: {args.root_mode}, Foot geom: {args.foot_geom}")
    print(f"  Opt substeps: {opt_substeps} ({opt_substeps * FPS}Hz), "
          f"Final eval: {SIM_SUBSTEPS} ({SIM_FREQ}Hz)")
    print(f"{'='*60}\n")
    from prepare5.optimize_tracking import (
        pd_compose_kernel, position_loss_kernel, delta_reg_kernel,
    )
    from prepare4.gen_xml import get_or_create_xml
    from prepare4.dynamics import set_segment_masses
    from newton import CollisionPipeline

    device = args.device
    root_mode_int = {"free": 0, "orient": 1, "skyhook": 2}[args.root_mode]

    # ── Load data ────────────────────────────────────────────
    persons, text = load_clip(args.clip_id, args.source)
    joint_q, betas = retarget_person(persons[0], args.source, device=device)
    ref_jq = joint_q.astype(np.float32)
    T = ref_jq.shape[0]
    print(f"Loaded: {T} frames ({T/30:.1f}s), text: {text}")

    # ── Build model ──────────────────────────────────────────
    xml_path = get_or_create_xml(betas, foot_geom=args.foot_geom)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device, requires_grad=True)
    set_segment_masses(model, total_mass=DEFAULT_BODY_MASS_KG, verbose=False)

    model.mujoco.dof_passive_stiffness.fill_(0.0)
    model.mujoco.dof_passive_damping.fill_(0.0)
    model.joint_target_ke.fill_(0.0)
    model.joint_target_kd.fill_(0.0)

    n_dof = model.joint_dof_count
    n_coords = model.joint_coord_count

    model.joint_limit_lower = wp.array(
        np.full(n_dof, -1e6, dtype=np.float32),
        dtype=wp.float32, device=device
    )
    model.joint_limit_upper = wp.array(
        np.full(n_dof, 1e6, dtype=np.float32),
        dtype=wp.float32, device=device
    )

    arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
    arm[:6] = ARMATURE_ROOT
    model.joint_armature = wp.array(arm, dtype=wp.float32, device=device)

    solver = newton.solvers.SolverFeatherstone(model)
    coll_pipeline = CollisionPipeline(
        model, broad_phase="explicit", requires_grad=False
    )
    contacts = coll_pipeline.contacts()

    # ── Pre-allocate ─────────────────────────────────────────
    W = args.window
    max_steps = W * opt_substeps + 1
    states = [model.state() for _ in range(max_steps)]
    control = model.control()

    ref_padded = np.zeros((T, n_coords), dtype=np.float32)
    ref_padded[:, :COORDS_PER_PERSON] = ref_jq
    ref_coords_flat = wp.array(
        ref_padded.flatten(), dtype=wp.float32, device=device
    )

    delta_q = wp.array(
        np.zeros(T * n_dof, dtype=np.float32),
        dtype=wp.float32, device=device, requires_grad=True,
    )

    combined_frames = [
        wp.zeros(n_dof, dtype=wp.float32, device=device, requires_grad=True)
        for _ in range(W)
    ]

    smpl_body = np.array(
        [SMPL_TO_NEWTON[j] for j in range(N_SMPL_JOINTS)], dtype=np.int32
    )
    smpl_to_body_wp = wp.array(smpl_body, dtype=wp.int32, device=device)

    # Ref positions via FK
    print("Computing reference positions...")
    ref_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    state_tmp = model.state()
    jqd_tmp = wp.zeros(n_dof, dtype=wp.float32, device=device)
    ref_pos_wp = []
    for t in range(T):
        q = np.zeros(n_coords, dtype=np.float32)
        q[:COORDS_PER_PERSON] = ref_jq[t]
        state_tmp.joint_q = wp.array(q, dtype=wp.float32, device=device)
        newton.eval_fk(model, state_tmp.joint_q, jqd_tmp, state_tmp)
        body_q = state_tmp.body_q.numpy().reshape(-1, 7)
        pos = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)
        for j, b in SMPL_TO_NEWTON.items():
            pos[j] = body_q[b, :3]
        ref_positions[t] = pos
        ref_pos_wp.append(wp.array(pos, dtype=wp.vec3, device=device))

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    # ── Adam state ───────────────────────────────────────────
    adam_m = None
    adam_v = None
    adam_t = 0
    lr = args.lr

    # ── Optimization loop ────────────────────────────────────
    n_windows = max(1, (T + W - 1) // W)
    best_loss = float('inf')
    best_delta_np = None

    print(f"\nStarting optimization: {args.epochs} epochs × "
          f"{n_windows} windows = {args.epochs * n_windows} iterations\n")

    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []

        for win_idx in range(n_windows):
            w_start = win_idx * W
            w_end = min(w_start + W, T)
            w_len = w_end - w_start

            # Init state from reference (OUTSIDE tape)
            q0 = np.zeros(n_coords, dtype=np.float32)
            q0[:COORDS_PER_PERSON] = ref_jq[w_start]
            states[0].joint_q = wp.array(
                q0, dtype=wp.float32, device=device
            )
            states[0].joint_qd = wp.zeros(
                n_dof, dtype=wp.float32, device=device
            )

            coll_pipeline.collide(states[0], contacts)
            loss.zero_()

            # Forward + backward
            tape = wp.Tape()
            with tape:
                step_idx = 0
                for f, t_frame in enumerate(range(w_start, w_end)):
                    dof_off = t_frame * n_dof
                    coord_off = t_frame * n_coords
                    wp.launch(
                        pd_compose_kernel,
                        dim=n_dof,
                        inputs=[
                            states[step_idx].joint_q,
                            states[step_idx].joint_qd,
                            ref_coords_flat, delta_q,
                            combined_frames[f],
                            dof_off, coord_off,
                            DOFS_PER_PERSON, COORDS_PER_PERSON,
                            args.ke_root, args.kd_root,
                            args.ke_joint, args.kd_joint,
                            TORQUE_LIMIT, root_mode_int,
                        ],
                        device=device,
                    )
                    control.joint_f = combined_frames[f]
                    for _ in range(opt_substeps):
                        solver.step(
                            states[step_idx], states[step_idx + 1],
                            control, contacts, opt_dt,
                        )
                        step_idx += 1

                    if t_frame < len(ref_pos_wp):
                        wp.launch(
                            position_loss_kernel,
                            dim=N_SMPL_JOINTS,
                            inputs=[
                                states[step_idx].body_q,
                                ref_pos_wp[t_frame],
                                smpl_to_body_wp,
                                N_SMPL_JOINTS, loss,
                            ],
                            device=device,
                        )

                # Regularization
                for t_frame in range(w_start, w_end):
                    offset = t_frame * n_dof
                    wp.launch(
                        delta_reg_kernel,
                        dim=n_dof,
                        inputs=[delta_q, args.reg_lambda,
                                offset, n_dof, loss],
                        device=device,
                    )

            tape.backward(loss)
            loss_val = loss.numpy()[0]
            epoch_losses.append(loss_val)

            # Adam update
            if delta_q.grad is not None:
                g = delta_q.grad.numpy().reshape(T, n_dof)
                g[:, :6] = 0.0  # no root grad

                # Window mask
                mask = np.zeros((T, 1), dtype=np.float32)
                mask[w_start:w_end] = 1.0
                g *= mask

                g_flat = g.flatten()
                grad_norm = float(np.linalg.norm(g_flat))
                if grad_norm > 100.0:
                    g_flat *= 100.0 / grad_norm

                if adam_m is None:
                    adam_m = np.zeros_like(g_flat)
                    adam_v = np.zeros_like(g_flat)
                adam_t += 1
                adam_m = 0.9 * adam_m + 0.1 * g_flat
                adam_v = 0.999 * adam_v + 0.001 * g_flat ** 2
                m_hat = adam_m / (1.0 - 0.9 ** adam_t)
                v_hat = adam_v / (1.0 - 0.999 ** adam_t)

                delta_np = delta_q.numpy()
                update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                delta_np -= update
                delta_q = wp.array(
                    delta_np.astype(np.float32),
                    dtype=wp.float32, device=device, requires_grad=True,
                )

            tape.zero()

        # Epoch summary
        epoch_loss = np.mean(epoch_losses)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_delta_np = delta_q.numpy().copy()

        elapsed = time.time() - t_start
        dq = delta_q.numpy().reshape(T, n_dof)
        print(f"  Epoch {epoch + 1:3d}/{args.epochs} | "
              f"loss={epoch_loss:.6f} | best={best_loss:.6f} | "
              f"|Δq|={np.abs(dq[:, 6:]).mean():.5f} | "
              f"{elapsed:.0f}s elapsed")

    # ── Final forward pass with MuJoCo solver ────────────────
    print(f"\n{'='*60}")
    print(f"Running final forward pass with best Δq...")
    print(f"  Using SAME solver (Featherstone) and substeps ({opt_substeps}) as optimization")
    print(f"{'='*60}")

    delta_2d = best_delta_np.reshape(T, n_dof)

    # Use the SAME solver as optimization to avoid sim-to-sim transfer issues
    # (MuJoCo solver at 480Hz diverges with Δq tuned for Featherstone at 120Hz)

    state_0 = model.state()
    state_1 = model.state()
    ctrl = model.control()

    q0 = np.zeros(n_coords, dtype=np.float32)
    q0[:COORDS_PER_PERSON] = ref_jq[0]
    state_0.joint_q = wp.array(q0, dtype=wp.float32, device=device)
    state_0.joint_qd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    sim_jq = np.zeros((T, COORDS_PER_PERSON), dtype=np.float32)
    sim_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    sim_jq[0] = ref_jq[0]

    body_q = state_0.body_q.numpy().reshape(-1, 7)
    for j, b in SMPL_TO_NEWTON.items():
        sim_positions[0, j] = body_q[b, :3]

    from scipy.spatial.transform import Rotation

    eval_substeps = opt_substeps  # match optimization frequency
    eval_dt = DT / eval_substeps

    for t in range(1, T):
        ref_q = ref_jq[t]
        dq = delta_2d[t]

        for sub in range(eval_substeps):
            cq = state_0.joint_q.numpy()
            cqd = state_0.joint_qd.numpy()

            tau = np.zeros(n_dof, dtype=np.float32)

            if root_mode_int == 2:
                tau[:3] = (args.ke_root * (ref_q[:3] - cq[:3])
                           - args.kd_root * cqd[:3])
            if root_mode_int >= 1:
                q_cur = cq[3:7].copy()
                qn = np.linalg.norm(q_cur)
                if qn > 1e-8:
                    q_cur /= qn
                R_err = (Rotation.from_quat(ref_q[3:7])
                         * Rotation.from_quat(q_cur).inv()).as_rotvec()
                tau[3:6] = (args.ke_root * R_err
                            - args.kd_root * cqd[3:6])

            target = ref_q[7:COORDS_PER_PERSON] + dq[6:DOFS_PER_PERSON]
            tau[6:DOFS_PER_PERSON] = (
                args.ke_joint * (target - cq[7:COORDS_PER_PERSON])
                - args.kd_joint * cqd[6:DOFS_PER_PERSON]
            )
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

            joint_f = np.zeros(n_dof, dtype=np.float32)
            joint_f[:DOFS_PER_PERSON] = tau
            ctrl.joint_f = wp.array(
                joint_f, dtype=wp.float32, device=device
            )

            coll_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, ctrl, contacts, eval_dt)
            state_0, state_1 = state_1, state_0

        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        cq_np = state_0.joint_q.numpy()
        sim_jq[t] = cq_np[:COORDS_PER_PERSON]
        body_q = state_0.body_q.numpy().reshape(-1, 7)
        for j, b in SMPL_TO_NEWTON.items():
            sim_positions[t, j] = body_q[b, :3]

        if t % 30 == 0:
            mpjpe = np.linalg.norm(
                sim_positions[t] - ref_positions[t], axis=-1
            ).mean() * 1000
            print(f"    Frame {t:4d}/{T}: MPJPE={mpjpe:.1f}mm")

    # ── Metrics ──────────────────────────────────────────────
    errors = np.linalg.norm(sim_positions - ref_positions, axis=-1)
    mpjpe_mm = errors.mean() * 1000
    max_err_mm = errors.max() * 1000
    root_drift = np.linalg.norm(sim_jq[:, :3] - ref_jq[:, :3], axis=-1)

    print(f"\n  {'='*40}")
    print(f"  Final results:")
    print(f"    MPJPE:      {mpjpe_mm:.1f} mm")
    print(f"    Max error:  {max_err_mm:.1f} mm")
    print(f"    Root drift: {root_drift.mean()*100:.1f} cm (mean), "
          f"{root_drift.max()*100:.1f} cm (max)")
    print(f"    Epochs:     {args.epochs}")
    print(f"    Time:       {time.time() - t_start:.0f}s")
    print(f"  {'='*40}")

    # ── Save ─────────────────────────────────────────────────
    output_dir = os.path.join(
        args.output_dir or "output/phc_tracker",
        f"clip_{args.clip_id}_{args.source}_optimized"
    )
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        os.path.join(output_dir, "optimized_result.npz"),
        sim_positions=sim_positions,
        ref_positions=ref_positions,
        sim_joint_q=sim_jq,
        delta_q=delta_2d,
        ref_joint_q=ref_jq,
    )

    import json
    metrics = {
        'clip_id': args.clip_id,
        'source': args.source,
        'text': text,
        'mpjpe_mm': float(mpjpe_mm),
        'max_error_mm': float(max_err_mm),
        'root_drift_mean_cm': float(root_drift.mean() * 100),
        'root_drift_max_cm': float(root_drift.max() * 100),
        'epochs': args.epochs,
        'lr': args.lr,
        'window': args.window,
        'root_mode': args.root_mode,
        'foot_geom': args.foot_geom,
        'best_loss': float(best_loss),
        'elapsed_s': float(time.time() - t_start),
    }
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved to {output_dir}/")
    print(f"  Files: optimized_result.npz, metrics.json")
    print(f"\n  To visualize:")
    print(f"    python prepare5/visualize_newton_tracking.py \\")
    print(f"        --result {output_dir}/optimized_result.npz \\")
    print(f"        --clip-id {args.clip_id} --source {args.source}")


def main():
    parser = argparse.ArgumentParser(
        description="Differentiable trajectory optimization for motion tracking"
    )
    parser.add_argument("--clip-id", type=int, default=1129,
                        help="InterHuman clip ID")
    parser.add_argument("--source", choices=["gt", "generated"], default="gt",
                        help="Data source")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of optimization epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--window", type=int, default=5,
                        help="Window size (frames per optimization step)")
    parser.add_argument("--opt-substeps", type=int, default=4,
                        help="Physics substeps for optimization (fewer = faster, "
                             "default 4 = 120Hz). Final eval uses full 16 = 480Hz.")
    parser.add_argument("--reg-lambda", type=float, default=1e-4,
                        help="L2 regularization weight on Δq")
    parser.add_argument("--ke-joint", type=float, default=200.0,
                        help="Joint PD stiffness (amplifies Δq)")
    parser.add_argument("--kd-joint", type=float, default=20.0,
                        help="Joint PD damping")
    parser.add_argument("--ke-root", type=float, default=5000.0,
                        help="Root PD stiffness")
    parser.add_argument("--kd-root", type=float, default=500.0,
                        help="Root PD damping")
    parser.add_argument("--root-mode", choices=["free", "orient", "skyhook"],
                        default="free",
                        help="Root force mode")
    parser.add_argument("--foot-geom", choices=["box", "sphere", "capsule"],
                        default="sphere",
                        help="Foot collision geometry")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")

    args = parser.parse_args()
    run_headless(args)


if __name__ == "__main__":
    main()
