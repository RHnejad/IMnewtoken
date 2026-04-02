#!/usr/bin/env python3
"""Forward dynamics validation: apply ImDy-predicted torques in Newton simulator.

Tests whether ImDy's predicted torques can reproduce the reference motion when
applied as direct joint actuation in Newton's forward simulation.

Chain of reasoning:
  PHC (IsaacGym) tracks ref motion → records torques
  ImDy learns to predict those torques from markers
  This script: ImDy torques → Newton forward sim → compare vs reference

Usage:
    python eval_pipeline/imdy_forward_dynamics.py \
        --data-dir data/retargeted_v2/interhuman \
        --clip-ids 1004,1006 \
        --output-dir output/imdy_forward \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_utils.constants import (
    BODY_NAMES, COORDS_PER_PERSON, DOFS_PER_PERSON,
    DOF_NAMES, N_SMPL_JOINTS, R_ROT, SMPL_TO_NEWTON,
)
from prepare_utils.gen_xml import get_or_create_xml

# ── SMPL joint names (24 joints, 0=pelvis) ──
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# ── Build mapping: ImDy prediction index (0-22) → Newton DOF offset ──
# ImDy predicts 23 joints = SMPL joints 1-23 (excluding pelvis)
# Newton DOFs: 0-5 = root, then 3 DOFs per body in XML joint order

def _build_newton_body_dof_start():
    """Build map from Newton body name → DOF start index."""
    body_dof = {}
    for i, name in enumerate(DOF_NAMES):
        if i < 6:  # skip root DOFs
            continue
        body_name = "_".join(name.split("_")[:-1])
        if body_name not in body_dof:
            body_dof[body_name] = i
    return body_dof


def build_imdy_to_newton_dof_map():
    """Build mapping from ImDy's 23 joint torques to Newton's 75 DOFs.

    Returns:
        imdy_to_dof: list of 23 tuples (newton_dof_start, newton_dof_end)
        Each ImDy joint i (= SMPL joint i+1) maps to 3 consecutive Newton DOFs.
    """
    body_dof_start = _build_newton_body_dof_start()

    # Newton body names in the order they appear
    # SMPL_TO_NEWTON maps SMPL joint idx → Newton body idx
    # BODY_NAMES[newton_body_idx] gives the body name

    imdy_to_dof = []
    for imdy_idx in range(23):
        smpl_idx = imdy_idx + 1  # ImDy joint 0 = SMPL joint 1
        smpl_name = SMPL_JOINT_NAMES[smpl_idx]

        # Find corresponding Newton body
        if smpl_idx in SMPL_TO_NEWTON:
            newton_body_idx = SMPL_TO_NEWTON[smpl_idx]
            newton_body_name = BODY_NAMES[newton_body_idx]
        elif smpl_name == "L_Hand":
            newton_body_name = "L_Hand"
        elif smpl_name == "R_Hand":
            newton_body_name = "R_Hand"
        else:
            raise ValueError(f"No Newton mapping for SMPL joint {smpl_idx} ({smpl_name})")

        dof_start = body_dof_start[newton_body_name]
        imdy_to_dof.append((dof_start, dof_start + 3))

    return imdy_to_dof


def imdy_torques_to_newton_dofs(
    imdy_torques: np.ndarray,
    imdy_to_dof: list,
    n_dof: int = DOFS_PER_PERSON,
) -> np.ndarray:
    """Convert ImDy torques (23, 3) to Newton DOF torques (75,).

    ImDy torques are in SMPL body-local frame.
    Newton expects torques in its own body-local frame (rotated by R_ROT).

    Args:
        imdy_torques: (23, 3) predicted torques in SMPL frame
        imdy_to_dof: mapping from build_imdy_to_newton_dof_map()
        n_dof: total DOFs

    Returns:
        newton_torques: (n_dof,) torques for control.joint_f
    """
    newton_tau = np.zeros(n_dof, dtype=np.float32)

    for imdy_idx in range(23):
        dof_start, dof_end = imdy_to_dof[imdy_idx]
        # Rotate from SMPL body-local to Newton body-local via R_ROT
        tau_smpl = imdy_torques[imdy_idx]  # (3,)
        tau_newton = R_ROT @ tau_smpl
        newton_tau[dof_start:dof_end] = tau_newton

    return newton_tau


def run_forward_dynamics(
    ref_jq: np.ndarray,
    imdy_torques_seq: np.ndarray,
    betas: np.ndarray,
    fps: float = 30.0,
    sim_freq: float = 480.0,
    joint_style: str = "phc",
    device: str = "cuda:0",
    root_mode: str = "skyhook",
):
    """Run Newton forward simulation driven by ImDy torques.

    Args:
        ref_jq: (T, COORDS_PER_PERSON) reference joint coordinates
        imdy_torques_seq: (T, 23, 3) per-frame ImDy torques
        betas: (10,) or (16,) SMPL betas
        fps: motion frame rate
        sim_freq: simulation frequency
        joint_style: XML joint style
        device: compute device
        root_mode: "skyhook" (lock root to reference) or "free" (no root control)

    Returns:
        dict with sim_positions, ref_positions, tracking_error, etc.
    """
    import warp as wp
    import newton

    wp.init()

    T = len(ref_jq)
    dt_sim = 1.0 / sim_freq
    sim_steps = int(round(sim_freq / fps))

    # Build model
    xml_path = get_or_create_xml(betas, joint_style=joint_style)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    from prepare_utils.constants import ARMATURE_HINGE, ARMATURE_ROOT

    # Disable passive springs and built-in PD
    model.mujoco.dof_passive_stiffness.fill_(0.0)
    model.mujoco.dof_passive_damping.fill_(0.0)
    model.joint_target_ke.fill_(0.0)
    model.joint_target_kd.fill_(0.0)

    # Widen joint limits (prevents constraint forces at extreme angles)
    joint_limit_lower = model.joint_limit_lower.numpy()
    joint_limit_upper = model.joint_limit_upper.numpy()
    joint_limit_lower[:] = -1e6
    joint_limit_upper[:] = 1e6
    model.joint_limit_lower = wp.array(joint_limit_lower, dtype=wp.float32, device=device)
    model.joint_limit_upper = wp.array(joint_limit_upper, dtype=wp.float32, device=device)

    # Armature for numerical stability
    n_dof = model.joint_dof_count
    arm = np.full(n_dof, ARMATURE_HINGE, dtype=np.float32)
    arm[:6] = ARMATURE_ROOT
    model.joint_armature = wp.array(arm, dtype=wp.float32, device=device)

    # Setup solver
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=450, nconmax=150,
        impratio=10, iterations=100, ls_iterations=50,
    )

    # Initialize state from reference frame 0
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    init_q = np.zeros(model.joint_coord_count, dtype=np.float32)
    init_q[:COORDS_PER_PERSON] = ref_jq[0]
    state_0.joint_q = wp.array(init_q, dtype=wp.float32, device=device)
    state_0.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Build torque mapping
    imdy_to_dof = build_imdy_to_newton_dof_map()

    # Output buffers
    sim_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    ref_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    tracking_errors = np.zeros(T, dtype=np.float32)
    root_drift = np.zeros(T, dtype=np.float32)

    # Compute reference positions for all frames via FK
    for t in range(T):
        temp_q = np.zeros(model.joint_coord_count, dtype=np.float32)
        temp_q[:COORDS_PER_PERSON] = ref_jq[t]
        temp_state = model.state()
        temp_state.joint_q = wp.array(temp_q, dtype=wp.float32, device=device)
        temp_state.joint_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        newton.eval_fk(model, temp_state.joint_q, temp_state.joint_qd, temp_state)
        bp = temp_state.body_q.numpy()
        ref_positions[t] = np.array([bp[SMPL_TO_NEWTON[j], :3] for j in range(N_SMPL_JOINTS)])

    # Extract initial sim positions
    bp0 = state_0.body_q.numpy()
    sim_positions[0] = np.array([bp0[SMPL_TO_NEWTON[j], :3] for j in range(N_SMPL_JOINTS)])
    tracking_errors[0] = 0.0

    print(f"  Running forward dynamics: {T} frames, {sim_steps} substeps/frame")

    for t in range(1, T):
        # Map ImDy torques to Newton DOFs
        if t < len(imdy_torques_seq):
            tau_all = imdy_torques_to_newton_dofs(
                imdy_torques_seq[t], imdy_to_dof, model.joint_dof_count
            )
        else:
            tau_all = np.zeros(model.joint_dof_count, dtype=np.float32)

        # Root handling
        if root_mode == "skyhook":
            # Lock root to reference position/orientation via high-gain PD
            cq = state_0.joint_q.numpy()
            cqd = state_0.joint_qd.numpy()
            root_kp = 5000.0
            root_kd = 500.0
            # Translation
            for i in range(3):
                tau_all[i] = root_kp * (ref_jq[t, i] - cq[i]) - root_kd * cqd[i]
            # Rotation (quaternion error → axis-angle torque)
            ref_quat = ref_jq[t, 3:7]  # xyzw
            cur_quat = cq[3:7]
            # Quaternion error: q_err = q_ref * q_cur_inv
            # For small errors, torque ≈ kp * 2 * vec(q_err)
            # Using simplified axis-angle error
            qe = _quat_multiply(_quat_conjugate(cur_quat), ref_quat)
            if qe[3] < 0:
                qe = -qe
            axis_angle_err = 2.0 * qe[:3]  # small angle approximation
            for i in range(3):
                tau_all[3 + i] = root_kp * axis_angle_err[i] - root_kd * cqd[3 + i]

        for sub in range(sim_steps):
            control.joint_f = wp.array(tau_all, dtype=wp.float32, device=device)
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt_sim)
            state_0, state_1 = state_1, state_0

        # Extract simulated body positions
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        bp = state_0.body_q.numpy()
        sim_positions[t] = np.array([bp[SMPL_TO_NEWTON[j], :3] for j in range(N_SMPL_JOINTS)])

        # Tracking error (mean per-joint position error)
        err = np.sqrt(np.sum((sim_positions[t] - ref_positions[t]) ** 2, axis=-1))
        tracking_errors[t] = err.mean()

        # Root drift
        cq = state_0.joint_q.numpy()
        root_drift[t] = np.linalg.norm(cq[:3] - ref_jq[t, :3])

        if t % 30 == 0 or t == T - 1:
            print(f"    Frame {t}/{T}: tracking_err={tracking_errors[t]*100:.1f}cm, "
                  f"root_drift={root_drift[t]*100:.1f}cm")

    return {
        "sim_positions": sim_positions,
        "ref_positions": ref_positions,
        "tracking_errors": tracking_errors,
        "root_drift": root_drift,
        "mean_tracking_error_cm": float(tracking_errors[1:].mean() * 100),
        "max_tracking_error_cm": float(tracking_errors[1:].max() * 100),
        "mean_root_drift_cm": float(root_drift[1:].mean() * 100),
    }


def _quat_conjugate(q):
    """Conjugate of quaternion (xyzw format)."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_multiply(q1, q2):
    """Multiply two quaternions (xyzw format)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def render_comparison(
    sim_positions: np.ndarray,
    ref_positions: np.ndarray,
    tracking_errors: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    title: str = "",
):
    """Render side-by-side skeleton comparison video."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from prepare_utils.constants import SMPL_22_PARENTS

    T = len(sim_positions)
    parents = SMPL_22_PARENTS

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "3d"})
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    def draw_skeleton(ax, positions, color, label):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 2.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(label)

        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=color, s=20, alpha=0.8)
        for j in range(len(positions)):
            p = parents[j] if j < len(parents) else -1
            if p >= 0:
                ax.plot([positions[j, 0], positions[p, 0]],
                        [positions[j, 1], positions[p, 1]],
                        [positions[j, 2], positions[p, 2]],
                        c=color, linewidth=1.5, alpha=0.7)

    def update(t):
        draw_skeleton(axes[0], ref_positions[t], "blue", f"Reference (frame {t})")
        draw_skeleton(axes[1], sim_positions[t], "red",
                      f"ImDy Torques (err={tracking_errors[t]*100:.1f}cm)")
        return axes

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ImDy forward dynamics validation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="output/imdy_forward")
    parser.add_argument("--clip-ids", default="", help="Comma-separated clip IDs")
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--imdy-config", default="prepare5/ImDy/config/IDFD_mkr.yml")
    parser.add_argument("--imdy-checkpoint", default="prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joint-style", default="phc")
    parser.add_argument("--root-mode", default="skyhook", choices=["skyhook", "free"])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    args = parser.parse_args()

    import glob
    import re
    import json

    # Discover clips
    if args.clip_ids:
        clip_ids = [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    else:
        pat = re.compile(r"^(.+)_person0\.npy$")
        clip_ids = []
        for f in sorted(glob.glob(os.path.join(args.data_dir, "*_person0.npy"))):
            name = os.path.basename(f)
            if "_joint_q" in name or "_betas" in name or "_torques" in name:
                continue
            m = pat.match(name)
            if m:
                clip_ids.append(m.group(1))
        clip_ids = clip_ids[:args.max_clips]

    if not clip_ids:
        print("No clips found!")
        return

    # Load ImDy model
    from eval_pipeline.imdy_model_wrapper import ImDyWrapper
    from eval_pipeline.imdy_preprocessor import preprocess_for_imdy

    print("Loading ImDy model...")
    imdy = ImDyWrapper(
        config_path=args.imdy_config,
        checkpoint_path=args.imdy_checkpoint,
        device=args.device,
    )
    print(f"Model loaded on {imdy.device}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for clip_id in clip_ids:
        print(f"\n{'='*60}")
        print(f"Clip: {clip_id}")
        print(f"{'='*60}")

        # Load reference data
        positions = np.load(os.path.join(args.data_dir, f"{clip_id}_person0.npy")).astype(np.float32)
        jq = np.load(os.path.join(args.data_dir, f"{clip_id}_person0_joint_q.npy")).astype(np.float32)
        betas_path = os.path.join(args.data_dir, f"{clip_id}_person0_betas.npy")
        betas = np.load(betas_path).astype(np.float32) if os.path.exists(betas_path) else np.zeros(10, dtype=np.float32)

        print(f"  Positions: {positions.shape}, Joint Q: {jq.shape}")

        # Run ImDy to get per-frame torques
        print("  Running ImDy inference...")
        mkr, mvel, idx = preprocess_for_imdy(positions, dt=1.0 / args.fps)
        pred = imdy.predict_clip(mkr, mvel, batch_size=256)
        torques = pred["torque"]  # (N_windows, 23, 3)

        # Map windowed predictions back to full-clip frames
        # ImDy gives torques for frames idx[0]..idx[-1]. Pad start/end with nearest.
        T = len(jq)
        full_torques = np.zeros((T, 23, 3), dtype=np.float32)
        for i, frame_idx in enumerate(idx):
            full_torques[frame_idx] = torques[i]
        # Pad start frames
        for t in range(idx[0]):
            full_torques[t] = torques[0]
        # Pad end frames
        for t in range(idx[-1] + 1, T):
            full_torques[t] = torques[-1]

        print(f"  Torques: {full_torques.shape}, range=[{full_torques.min():.1f}, {full_torques.max():.1f}] Nm")

        # Run forward dynamics
        result = run_forward_dynamics(
            ref_jq=jq,
            imdy_torques_seq=full_torques,
            betas=betas,
            fps=args.fps,
            joint_style=args.joint_style,
            device=args.device,
            root_mode=args.root_mode,
        )

        print(f"\n  Results:")
        print(f"    Mean tracking error: {result['mean_tracking_error_cm']:.2f} cm")
        print(f"    Max tracking error:  {result['max_tracking_error_cm']:.2f} cm")
        print(f"    Mean root drift:     {result['mean_root_drift_cm']:.2f} cm")

        # Save metrics
        metrics = {
            "clip_id": clip_id,
            "num_frames": T,
            "root_mode": args.root_mode,
            "joint_style": args.joint_style,
            "mean_tracking_error_cm": result["mean_tracking_error_cm"],
            "max_tracking_error_cm": result["max_tracking_error_cm"],
            "mean_root_drift_cm": result["mean_root_drift_cm"],
            "per_frame_error_cm": result["tracking_errors"].tolist(),
        }
        metrics_path = os.path.join(args.output_dir, f"{clip_id}_forward_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        all_results.append(metrics)

        # Render comparison video
        if args.render:
            video_path = os.path.join(args.output_dir, f"{clip_id}_forward_comparison.mp4")
            render_comparison(
                result["sim_positions"],
                result["ref_positions"],
                result["tracking_errors"],
                video_path,
                fps=args.fps,
                title=f"Clip {clip_id} — ImDy Torque Forward Dynamics ({args.root_mode})",
            )

    # Summary
    if all_results:
        mean_err = np.mean([r["mean_tracking_error_cm"] for r in all_results])
        print(f"\n{'='*60}")
        print(f"Overall: {len(all_results)} clips, mean tracking error = {mean_err:.2f} cm")
        print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
