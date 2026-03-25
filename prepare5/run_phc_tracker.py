"""
run_phc_tracker.py — CLI for PHC-style physics-based motion tracking.

Tracks reference motions through physics simulation using PHC-style PD
targets, producing the closest physically feasible motion. The tracking
error quantifies physical plausibility.

Usage:
    # Track a single GT clip (solo)
    python prepare5/run_phc_tracker.py --clip-id 1129 --source gt

    # Track a single GT clip (paired — both persons)
    python prepare5/run_phc_tracker.py --clip-id 1129 --source gt --paired

    # Track a generated clip
    python prepare5/run_phc_tracker.py --clip-id 1129 --source generated

    # Compare old PD approach vs new PHC approach
    python prepare5/run_phc_tracker.py --clip-id 1129 --source gt --compare-old

    # Adjust PD gain scale
    python prepare5/run_phc_tracker.py --clip-id 1129 --source gt --gain-scale 1.5

    # Specify GPU
    python prepare5/run_phc_tracker.py --clip-id 1129 --source gt --device cuda:1
"""
import os
import sys
import argparse
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_clip(clip_id, source):
    """Load a clip's person data and text label.

    Returns:
        persons: list of 2 dicts (person1, person2), each with:
            root_orient, pose_body, trans, betas, [positions_zup]
        text: annotation text
    """
    from prepare4.run_full_analysis import load_gt_persons, load_gen_persons

    if source == "gt":
        persons = load_gt_persons(clip_id)
        text_path = os.path.join(PROJECT_ROOT, "data", "InterHuman",
                                 "annots", f"{clip_id}.txt")
        text = f"GT clip {clip_id}"
        if os.path.isfile(text_path):
            with open(text_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                text = lines[0]
        return persons, text
    else:
        persons, text = load_gen_persons(clip_id)
        return persons, text or f"Gen clip {clip_id}"


def retarget_person(person_data, source, device="cuda:0"):
    """Retarget SMPL-X parameters to Newton joint_q.

    For GT: rotation retarget (direct mapping), downsample 60→30fps.
    For Generated: IK retarget from positions_zup.

    Returns:
        joint_q: (T, 76) Newton joint coordinates
        betas: (10,) shape parameters
    """
    from prepare4.retarget import rotation_retarget, ik_retarget

    betas = person_data["betas"]
    if isinstance(betas, np.ndarray) and betas.ndim == 2:
        betas = betas[0]

    if source == "generated":
        if "positions_zup" not in person_data:
            raise ValueError("Generated data missing positions_zup")
        joint_q, _ = ik_retarget(
            person_data["positions_zup"], betas,
            ik_iters=50, device=device, sequential=True,
        )
    else:
        joint_q = rotation_retarget(
            person_data["root_orient"], person_data["pose_body"],
            person_data["trans"], betas,
        )
        joint_q = joint_q[::2]  # 60fps → 30fps

    return joint_q, betas


def run_old_pd_tracker(joint_q, betas, device="cuda:0"):
    """Run the old prepare4/ explicit PD tracker for comparison.

    Returns dict with sim_positions, mpjpe_mm, root_drift_m.
    """
    from prepare4.run_full_analysis import pd_forward_torques
    from prepare5.phc_reward import compute_tracking_errors
    from prepare5.phc_config import SMPL_TO_NEWTON, N_SMPL_JOINTS

    print("\n  [Old PD Tracker] Running for comparison...")
    torques, sim_jq = pd_forward_torques(
        joint_q, betas, device=device, verbose=True,
    )

    # Compute positions from sim_jq via FK
    import warp as wp
    import newton
    from prepare4.gen_xml import get_or_create_xml
    from prepare4.dynamics import set_segment_masses

    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    T = sim_jq.shape[0]
    sim_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    ref_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    for t in range(T):
        # Simulated positions
        state.joint_q = wp.array(
            np.pad(sim_jq[t], (0, model.joint_coord_count - 76)),
            dtype=wp.float32, device=device,
        )
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)
        for j, b in SMPL_TO_NEWTON.items():
            sim_positions[t, j] = body_q[b, :3]

        # Reference positions
        state.joint_q = wp.array(
            np.pad(joint_q[t].astype(np.float32), (0, model.joint_coord_count - 76)),
            dtype=wp.float32, device=device,
        )
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)
        for j, b in SMPL_TO_NEWTON.items():
            ref_positions[t, j] = body_q[b, :3]

    errors = compute_tracking_errors(sim_positions, ref_positions)
    root_drift = np.linalg.norm(sim_jq[:, :3] - joint_q[:T, :3], axis=-1)

    print(f"  [Old PD] MPJPE: {errors['mpjpe_mm']:.1f} mm, "
          f"root drift: {root_drift.mean()*100:.1f} cm (mean), "
          f"{root_drift.max()*100:.1f} cm (max)")

    return {
        'sim_positions': sim_positions,
        'ref_positions': ref_positions,
        'mpjpe_mm': errors['mpjpe_mm'],
        'per_frame_mpjpe_mm': errors['per_frame_mpjpe_mm'],
        'root_drift_m': root_drift,
    }


def plot_tracking_comparison(phc_result, old_result, output_dir, clip_id, text):
    """Plot PHC vs old PD tracking quality."""
    T = min(len(phc_result['per_frame_mpjpe_mm']),
            len(old_result['per_frame_mpjpe_mm']))
    frames = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Tracking Quality Comparison — Clip {clip_id}\n{text}",
                 fontsize=13)

    # MPJPE over time
    ax = axes[0]
    ax.plot(frames, phc_result['per_frame_mpjpe_mm'][:T],
            'b-', alpha=0.8, label=f"PHC-style (mean={phc_result['mpjpe_mm']:.0f}mm)")
    ax.plot(frames, old_result['per_frame_mpjpe_mm'][:T],
            'r-', alpha=0.8, label=f"Old PD (mean={old_result['mpjpe_mm']:.0f}mm)")
    ax.set_ylabel("MPJPE (mm)")
    ax.legend(fontsize=11)
    ax.set_title("Per-Frame Mean Joint Position Error")
    ax.grid(True, alpha=0.3)

    # Root drift over time
    ax = axes[1]
    ax.plot(frames, phc_result['root_drift_m'][:T] * 100,
            'b-', alpha=0.8,
            label=f"PHC-style (mean={phc_result['root_drift_m'][:T].mean()*100:.1f}cm)")
    ax.plot(frames, old_result['root_drift_m'][:T] * 100,
            'r-', alpha=0.8,
            label=f"Old PD (mean={old_result['root_drift_m'][:T].mean()*100:.1f}cm)")
    ax.set_ylabel("Root Drift (cm)")
    ax.set_xlabel("Frame")
    ax.legend(fontsize=11)
    ax.set_title("Root Position Drift")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=25, color='k', linestyle='--', alpha=0.3, label='Termination (25cm)')

    plt.tight_layout()
    path = os.path.join(output_dir, "phc_vs_old_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_tracking_result(result, output_dir, clip_id, text, prefix="phc"):
    """Plot per-frame tracking metrics for a single run."""
    T = len(result['per_frame_mpjpe_mm'])
    frames = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{prefix.upper()} Tracking — Clip {clip_id}\n{text}",
                 fontsize=13)

    # MPJPE
    ax = axes[0]
    ax.plot(frames, result['per_frame_mpjpe_mm'], 'b-', alpha=0.8)
    ax.axhline(y=result['mpjpe_mm'], color='b', linestyle='--', alpha=0.5,
               label=f"Mean: {result['mpjpe_mm']:.0f} mm")
    ax.set_ylabel("MPJPE (mm)")
    ax.legend()
    ax.set_title("Per-Frame Mean Joint Position Error")
    ax.grid(True, alpha=0.3)

    # Root drift
    ax = axes[1]
    ax.plot(frames, result['root_drift_m'] * 100, 'r-', alpha=0.8)
    ax.axhline(y=25, color='k', linestyle='--', alpha=0.3, label='Termination')
    ax.set_ylabel("Root Drift (cm)")
    ax.legend()
    ax.set_title("Root Position Drift")
    ax.grid(True, alpha=0.3)

    # Rewards (if available)
    ax = axes[2]
    if 'rewards' in result:
        ax.plot(frames, result['rewards'], 'g-', alpha=0.8)
        ax.set_ylabel("Reward")
        ax.set_title("PHC Position Reward")
    else:
        ax.set_visible(False)
    ax.set_xlabel("Frame")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_tracking.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_joint_errors(result, output_dir, clip_id, prefix="phc"):
    """Bar chart of per-joint MPJPE."""
    from prepare5.phc_config import SMPL_TO_NEWTON, N_SMPL_JOINTS

    # SMPL joint names (22 joints)
    joint_names = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
        "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    ]

    mpjpe = result['per_joint_mpjpe_mm']
    n_joints = min(len(mpjpe), len(joint_names))

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(n_joints), mpjpe[:n_joints], color='steelblue', alpha=0.8)
    ax.set_xticks(range(n_joints))
    ax.set_xticklabels(joint_names[:n_joints], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title(f"{prefix.upper()} Per-Joint Tracking Error — Clip {clip_id}")
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight worst joints
    sorted_idx = np.argsort(mpjpe[:n_joints])[-3:]
    for idx in sorted_idx:
        bars[idx].set_color('salmon')

    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_per_joint_mpjpe.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def analyze_single_clip(args):
    """Analyze a single clip with PHC tracker."""
    from prepare5.phc_tracker import PHCTracker

    clip_id = args.clip_id
    source = args.source
    device = args.device

    output_dir = os.path.join(
        args.output_dir or "output/phc_tracker",
        f"clip_{clip_id}_{source}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHC Tracker Analysis: clip {clip_id} ({source})")
    print(f"{'='*60}")

    # Load data
    persons, text = load_clip(clip_id, source)
    if persons is None:
        print(f"ERROR: Could not load clip {clip_id} ({source})")
        return

    print(f"  Text: {text}")

    # Retarget
    print(f"\n  Retargeting person 1...")
    t0 = time.time()
    joint_q_A, betas_A = retarget_person(persons[0], source, device=device)
    print(f"    Done in {time.time()-t0:.1f}s, {joint_q_A.shape[0]} frames")

    if args.paired:
        print(f"  Retargeting person 2...")
        t0 = time.time()
        joint_q_B, betas_B = retarget_person(persons[1], source, device=device)
        print(f"    Done in {time.time()-t0:.1f}s, {joint_q_B.shape[0]} frames")

    # Run PHC tracker
    tracker = PHCTracker(
        device=device,
        gain_scale=args.gain_scale,
        verbose=True,
        use_builtin_pd=getattr(args, 'use_builtin_pd', False),
        gain_preset=getattr(args, 'gain_preset', 'phc'),
        foot_geom=getattr(args, 'foot_geom', 'sphere'),
        root_mode=getattr(args, 'root_mode', 'free'),
    )

    if args.paired:
        t0 = time.time()
        result = tracker.track_paired(
            joint_q_A, joint_q_B, betas_A, betas_B,
        )
        elapsed = time.time() - t0
        print(f"\n  Paired tracking took {elapsed:.1f}s")

        # Save results
        np.savez(
            os.path.join(output_dir, "phc_paired_result.npz"),
            sim_positions_A=result['sim_positions_A'],
            sim_positions_B=result['sim_positions_B'],
            ref_positions_A=result['ref_positions_A'],
            ref_positions_B=result['ref_positions_B'],
        )

        # Save metrics
        metrics = {
            'clip_id': clip_id, 'source': source, 'text': text,
            'mpjpe_A_mm': result['mpjpe_A_mm'],
            'mpjpe_B_mm': result['mpjpe_B_mm'],
            'root_drift_A_mean_cm': float(result['root_drift_A_m'].mean() * 100),
            'root_drift_B_mean_cm': float(result['root_drift_B_m'].mean() * 100),
            'elapsed_s': elapsed,
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved metrics to {output_dir}/metrics.json")

    else:
        # Solo tracking (person A only)
        t0 = time.time()
        result = tracker.track(joint_q_A, betas_A)
        elapsed = time.time() - t0
        print(f"\n  Solo tracking took {elapsed:.1f}s")

        # Save results
        np.savez(
            os.path.join(output_dir, "phc_result.npz"),
            sim_positions=result['sim_positions'],
            ref_positions=result['ref_positions'],
            sim_joint_q=result['sim_joint_q'],
            torques=result['torques'],
        )

        # Save metrics
        metrics = {
            'clip_id': clip_id, 'source': source, 'text': text,
            'mpjpe_mm': result['mpjpe_mm'],
            'max_error_mm': result['max_error_mm'],
            'root_drift_mean_cm': float(result['root_drift_m'].mean() * 100),
            'root_drift_max_cm': float(result['root_drift_m'].max() * 100),
            'terminated_at': result['terminated_at'],
            'mean_reward': float(result['rewards'][1:].mean()),
            'elapsed_s': elapsed,
            'gain_scale': args.gain_scale,
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Plot
        plot_tracking_result(result, output_dir, clip_id, text)
        plot_per_joint_errors(result, output_dir, clip_id)

        # Compare with old PD approach if requested
        if args.compare_old:
            old_result = run_old_pd_tracker(joint_q_A, betas_A, device=device)
            plot_tracking_comparison(result, old_result, output_dir, clip_id, text)

            # Add old metrics to JSON
            metrics['old_mpjpe_mm'] = old_result['mpjpe_mm']
            metrics['old_root_drift_mean_cm'] = float(
                old_result['root_drift_m'].mean() * 100
            )
            metrics['old_root_drift_max_cm'] = float(
                old_result['root_drift_m'].max() * 100
            )
            metrics['improvement_ratio'] = (
                old_result['mpjpe_mm'] / max(result['mpjpe_mm'], 0.1)
            )
            with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)

    print(f"\n  All outputs saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="PHC-style physics-based motion tracking in Newton"
    )
    parser.add_argument("--clip-id", type=int, default=1129,
                        help="InterHuman clip ID")
    parser.add_argument("--source", choices=["gt", "generated"], default="gt",
                        help="Data source")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--paired", action="store_true",
                        help="Track both persons in paired simulation")
    parser.add_argument("--compare-old", action="store_true",
                        help="Also run old PD tracker for comparison")
    parser.add_argument("--gain-scale", type=float, default=1.0,
                        help="PD gain multiplier (default 1.0 = PHC-matched)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: output/phc_tracker/)")
    parser.add_argument("--use-builtin-pd", action="store_true",
                        help="Use Newton's built-in PD instead of explicit PD")
    parser.add_argument("--gain-preset", choices=["phc", "old"], default="phc",
                        help="PD gain preset: 'phc' (PHC-matched) or 'old' (prepare2)")
    parser.add_argument("--foot-geom", choices=["box", "sphere", "capsule"],
                        default="sphere",
                        help="Foot collision geometry (default: sphere)")
    parser.add_argument("--root-mode", choices=["free", "orient", "skyhook"],
                        default="free",
                        help="Root force mode: free (no root forces, like PHC), "
                             "orient (orientation PD only), "
                             "skyhook (full position+orientation PD)")

    args = parser.parse_args()
    analyze_single_clip(args)


if __name__ == "__main__":
    main()
