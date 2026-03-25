#!/usr/bin/env python
"""
Position-based Newton visualization: derive joint_q from VQ-VAE output positions
via Newton IK (same as how GT positions are processed in prepare/retarget_newton.py).

This avoids using the VQ-VAE's degraded rotations entirely. Instead it:
  1. Extracts the 22-joint positions from the 262-dim decoded output
  2. Converts from processed coordinate frame → Z-up world frame
  3. Runs Newton GPU IK to solve for joint_q from positions
  4. Saves retargeted joint_q/betas npy files for visualization
  5. Runs the same physics analysis

Usage:
    conda activate mimickit
    python physics_analysis/retarget_from_positions.py --clips 4659 3678

This creates data in data/retargeted_v2/gen_from_positions/ and
physics_analysis/gen_from_positions_results/.
"""
import os
import sys
import argparse
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Custom attribute")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warp as wp
wp.config.verbose = False

import newton
import newton.ik as ik

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# trans_matrix used in InterHuman process_motion_np (Z-up → Y-up processed)
TRANS_MATRIX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
INV_TRANS_MATRIX = np.linalg.inv(TRANS_MATRIX)  # Y-up processed → Z-up world

# SMPL joint → Newton body index (from prepare/retarget_newton.py)
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}
N_SMPL_JOINTS = 22


# ═══════════════════════════════════════════════════════════════
# Position extraction from 262-dim VQ-VAE output
# ═══════════════════════════════════════════════════════════════

def extract_positions_from_262(motion_262):
    """
    Extract 22-joint positions from 262-dim decoded VQ-VAE output
    and convert to Z-up world frame (same frame as GT pkl data).

    Args:
        motion_262: (T, 262) decoded VQ-VAE output in processed frame

    Returns:
        positions_zup: (T, 22, 3) joint positions in Z-up world frame (meters)
    """
    T = motion_262.shape[0]
    # [0:66] = 22 joints × 3 positions in processed (Y-up) frame
    positions_proc = motion_262[:, :66].reshape(T, 22, 3).astype(np.float64)
    # Undo trans_matrix: processed → Z-up world
    positions_zup = np.einsum("mn,...n->...m", INV_TRANS_MATRIX, positions_proc)
    return positions_zup.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Newton IK (from prepare/retarget_newton.py, adapted)
# ═══════════════════════════════════════════════════════════════

def build_model(betas=None, device="cuda:0"):
    """Build Newton model. If betas provided, use per-subject XML; else neutral."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    if betas is not None:
        from prepare2.retarget import get_or_create_xml
        xml_path = get_or_create_xml(betas)
    else:
        xml_path = os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    return builder.finalize(device=device)


def build_ik_solver(model, T, ref_pos, device="cuda:0"):
    """Build batched IK solver for T frames."""
    objectives = []
    for smpl_j in range(N_SMPL_JOINTS):
        targets = []
        for t in range(T):
            p = ref_pos[t, smpl_j]
            targets.append(wp.vec3(float(p[0]), float(p[1]), float(p[2])))

        obj = ik.IKObjectivePosition(
            link_index=SMPL_TO_NEWTON[smpl_j],
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array(targets, dtype=wp.vec3, device=device),
            weight=1.0,
        )
        objectives.append(obj)

    solver = ik.IKSolver(
        model=model,
        n_problems=T,
        objectives=objectives,
        lambda_initial=0.01,
        jacobian_mode=ik.IKJacobianType.AUTODIFF,
    )
    return solver, objectives


def extract_positions_from_fk(model, joint_q_np, device="cuda:0"):
    """Extract (22, 3) positions from joint_q via FK."""
    pos = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
    state.joint_q = wp.array(joint_q_np, dtype=wp.float32, device=device)
    newton.eval_fk(model, state.joint_q, jqd, state)
    body_q = state.body_q.numpy().reshape(-1, 7)
    for j in range(N_SMPL_JOINTS):
        pos[j] = body_q[SMPL_TO_NEWTON[j], :3]
    return pos


def ik_from_positions(model, ref_pos, ik_iters=50, device="cuda:0"):
    """
    Run Newton IK: (T, 22, 3) positions → (T, 76) joint_q + (T, 22, 3) FK positions.

    Same as prepare/retarget_newton.py retarget_clip() but also returns joint_q.
    """
    T = ref_pos.shape[0]
    n_coords = model.joint_coord_count  # 76

    # Build solver
    solver, objectives = build_ik_solver(model, T, ref_pos, device)

    # Initialize: pelvis from reference, identity quaternion, rest zero
    jq_init = np.zeros((T, n_coords), dtype=np.float32)
    for t in range(T):
        jq_init[t, 0:3] = ref_pos[t, 0]       # pelvis xyz
        jq_init[t, 3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quat
    jq = wp.array(jq_init, dtype=wp.float32, device=device)

    # Solve all frames at once on GPU
    solver.step(jq, jq, iterations=ik_iters)
    wp.synchronize()

    # FK: joint_q → body positions per frame
    jq_np = jq.numpy()  # (T, 76)
    all_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    for t in range(T):
        all_positions[t] = extract_positions_from_fk(model, jq_np[t], device)

    # Report error
    per_frame_err = np.sqrt(((all_positions - ref_pos) ** 2).sum(-1).mean(-1))
    mean_err = per_frame_err.mean()
    max_err = per_frame_err.max()
    print(f"    IK MPJPE: mean={mean_err*100:.2f}cm, max={max_err*100:.2f}cm")

    return jq_np, all_positions


# ═══════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════

def process_generated_clip(clip_id, gen_dir, output_dir, ik_iters=50, device="cuda:0"):
    """
    Process one generated clip:
      1. Load decoded 262-dim motions
      2. Extract positions and convert to Z-up
      3. Run Newton IK to get joint_q
      4. Save joint_q + betas for visualization
    """
    # Load the generated 262-dim data
    # First check if we have it in the reconstructed dataset format
    recon_dir = os.path.join(PROJECT_ROOT, "data", "reconstructed_dataset", "interhuman")
    
    # Try loading from the generation pkl (which contains decoded 262-dim in SMPL-X format)
    # We need the raw 262-dim — let's load from the process that created it
    # Actually, generate_and_save.py saves pkl (SMPL-X params), not the raw 262-dim.
    # The 262-dim data exists as processed_262/ in the reconstructed dataset.
    
    processed_262_dir = os.path.join(recon_dir, "processed_262")
    
    # Try to get 262-dim data: naming is {clip}_person{1,2}.npy
    p1_262_path = os.path.join(processed_262_dir, f"{clip_id}_person1.npy")
    p2_262_path = os.path.join(processed_262_dir, f"{clip_id}_person2.npy")
    
    if os.path.exists(p1_262_path) and os.path.exists(p2_262_path):
        m1_262 = np.load(p1_262_path)
        m2_262 = np.load(p2_262_path)
        print(f"  Loaded 262-dim from reconstructed_dataset: p1={m1_262.shape}, p2={m2_262.shape}")
        use_262 = True
    else:
        # Alternative: load from the raw npy dataset (492-dim) and extract positions
        # NOTE: The 492-dim motions_processed positions are in Z-up raw frame
        # (created by save_generated_as_dataset.py decoded_262_to_raw_interhuman),
        # so we extract positions directly — do NOT route through extract_positions_from_262
        # which expects Y-up processed input.
        raw_dir = os.path.join(recon_dir, "motions_processed")
        p1_raw_path = os.path.join(raw_dir, f"person1/{clip_id}.npy")
        p2_raw_path = os.path.join(raw_dir, f"person2/{clip_id}.npy")
        
        if os.path.exists(p1_raw_path) and os.path.exists(p2_raw_path):
            p1_raw = np.load(p1_raw_path)  # (T, 492) — Z-up raw format
            p2_raw = np.load(p2_raw_path)
            # Direct reshape: positions are already in Z-up
            pos_p1_zup = p1_raw[:, :66].reshape(-1, 22, 3).astype(np.float32)
            pos_p2_zup = p2_raw[:, :66].reshape(-1, 22, 3).astype(np.float32)
            print(f"  Loaded raw npy from {raw_dir}: p1={p1_raw.shape}, p2={p2_raw.shape}")
            use_262 = False
        else:
            print(f"  ERROR: Cannot find 262-dim or raw data for clip {clip_id}")
            print(f"    Tried: {p1_262_path}")
            print(f"    Tried: {p1_raw_path}")
            print(f"    Run save_generated_as_dataset.py first!")
            return None

    os.makedirs(output_dir, exist_ok=True)

    # Use neutral betas (generated data uses zeros)
    betas = np.zeros(10, dtype=np.float64)
    model = build_model(betas=betas, device=device)

    results = {}
    for p_idx in range(2):
        print(f"  Person {p_idx + 1}:")
        if use_262:
            m_262 = [m1_262, m2_262][p_idx]
            # 262-dim: positions in Y-up processed frame → convert to Z-up
            positions_zup = extract_positions_from_262(m_262)
        else:
            # 492-dim: positions already in Z-up raw frame
            positions_zup = [pos_p1_zup, pos_p2_zup][p_idx]
        print(f"    Positions (Z-up): {positions_zup.shape}, "
              f"height range: [{positions_zup[:,:,2].min():.3f}, {positions_zup[:,:,2].max():.3f}]m")

        # Run IK
        joint_q, fk_positions = ik_from_positions(model, positions_zup,
                                                    ik_iters=ik_iters, device=device)

        # Save
        out_name = f"{clip_id}_person{p_idx}"
        np.save(os.path.join(output_dir, f"{out_name}_joint_q.npy"), joint_q)
        np.save(os.path.join(output_dir, f"{out_name}_betas.npy"), betas)
        np.save(os.path.join(output_dir, f"{out_name}.npy"), fk_positions)
        print(f"    Saved: {out_name}_joint_q.npy ({joint_q.shape})")

        results[f"person{p_idx}"] = {
            'joint_q': joint_q,
            'positions': fk_positions,
            'ref_positions': positions_zup,
        }

    return results


def process_gt_clip(clip_id, data_dir, output_dir, ik_iters=50, device="cuda:0"):
    """
    Process GT clip: extract positions from raw 492-dim npy → IK → joint_q.

    IMPORTANT: The GT motions_processed files (492-dim) store positions in the
    ORIGINAL Z-up coordinate frame (raw InterHuman format). This is DIFFERENT
    from the 262-dim processed format which stores positions in Y-up processed
    frame (after trans_matrix). So we must NOT apply INV_TRANS_MATRIX here —
    just reshape the first 66 dims directly as Z-up positions.

    See: extract_joint_positions.py which does the same direct reshape.
    See: process_motion_np() in data/utils.py which applies trans_matrix to
         these Z-up positions to get Y-up processed positions.
    """
    raw_dir = os.path.join(data_dir, "motions_processed")
    
    os.makedirs(output_dir, exist_ok=True)
    betas = np.zeros(10, dtype=np.float64)
    model = build_model(betas=betas, device=device)

    results = {}
    for p_idx, person_name in enumerate(["person1", "person2"]):
        raw_path = os.path.join(raw_dir, person_name, f"{clip_id}.npy")
        if not os.path.exists(raw_path):
            print(f"  WARNING: {raw_path} not found")
            continue
        
        raw = np.load(raw_path)  # (T, 492) — positions in Z-up raw frame
        print(f"  GT Person {p_idx + 1}: raw shape={raw.shape}")
        
        # Extract first 22 joints × 3 positions — already in Z-up (raw frame)
        # Do NOT apply INV_TRANS_MATRIX (that's only for 262-dim processed data)
        positions_zup = raw[:, :66].reshape(-1, 22, 3).astype(np.float32)
        print(f"    Positions (Z-up): {positions_zup.shape}")

        # Run IK
        joint_q, fk_positions = ik_from_positions(model, positions_zup,
                                                    ik_iters=ik_iters, device=device)

        out_name = f"{clip_id}_person{p_idx}"
        np.save(os.path.join(output_dir, f"{out_name}_joint_q.npy"), joint_q)
        np.save(os.path.join(output_dir, f"{out_name}_betas.npy"), betas)
        np.save(os.path.join(output_dir, f"{out_name}.npy"), fk_positions)
        print(f"    Saved: {out_name}_joint_q.npy ({joint_q.shape})")

        results[f"person{p_idx}"] = {
            'joint_q': joint_q,
            'positions': fk_positions,
            'ref_positions': positions_zup,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Video rendering: Newton FK positions → MP4 (Z-up, same style as
# prepare2/visualize_skyhook_mp4.py)
# ═══════════════════════════════════════════════════════════════

# 22-joint bone connections (same as prepare2/visualize_skyhook_mp4.py)
BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),   # spine + head
    (0, 1), (1, 4), (4, 7), (7, 10),               # left leg
    (0, 2), (2, 5), (5, 8), (8, 11),               # right leg
    (9, 13), (13, 16), (16, 18), (18, 20),          # left arm
    (9, 14), (14, 17), (17, 19), (19, 21),          # right arm
]

PERSON_COLORS = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
}


def render_newton_videos(clips, gt_dir, gen_dir, output_dir, device="cuda:0"):
    """
    Render MP4 videos from Newton FK positions using matplotlib FuncAnimation.
    Uses Z-up coordinate system (Newton native) with the same camera and style
    as prepare2/visualize_skyhook_mp4.py.

    For each clip, renders:
      - {clip}_gt_newton.mp4   — GT positions → IK → Newton FK → video
      - {clip}_gen_newton.mp4  — Generated positions → IK → Newton FK → video
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    os.makedirs(output_dir, exist_ok=True)

    for clip_id in clips:
        print(f"\n  Rendering videos for clip {clip_id}...")

        for label, data_dir, suffix in [("GT", gt_dir, "gt_newton"),
                                          ("Gen", gen_dir, "gen_newton")]:
            p1_path = os.path.join(data_dir, f"{clip_id}_person0.npy")
            p2_path = os.path.join(data_dir, f"{clip_id}_person1.npy")

            if not os.path.exists(p1_path) or not os.path.exists(p2_path):
                print(f"    {label}: positions not found, skipping")
                continue

            pos_p1 = np.load(p1_path)  # (T, 22, 3) Z-up
            pos_p2 = np.load(p2_path)  # (T, 22, 3) Z-up
            all_positions = {0: pos_p1, 1: pos_p2}
            persons = sorted(all_positions.keys())
            T = min(p.shape[0] for p in all_positions.values())

            # Compute bounds from all positions (Z-up: X,Y horizontal, Z height)
            all_pos = np.concatenate([v[:T] for v in all_positions.values()], axis=0)
            mins = all_pos.reshape(-1, 3).min(axis=0)
            maxs = all_pos.reshape(-1, 3).max(axis=0)
            center = 0.5 * (mins + maxs)
            span = np.maximum(maxs - mins, 1e-6)
            half = float(np.max(span)) * 0.65

            fig = plt.figure(figsize=(10, 8))
            ax3d = fig.add_subplot(111, projection='3d')
            fig.suptitle(f"{label} (Newton IK+FK): clip {clip_id}", fontsize=14)

            # Newton-like camera: Z-up, look toward +Y
            ax3d.view_init(elev=15.0, azim=-90.0)
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z (up)")
            ax3d.set_box_aspect((1.0, 1.0, 1.0))

            ax3d.set_xlim(center[0] - half, center[0] + half)
            ax3d.set_ylim(center[1] - half, center[1] + half)
            ax3d.set_zlim(max(0.0, mins[2] - 0.05), center[2] + half)

            # Ground plane at Z=0
            gx = np.linspace(center[0] - half, center[0] + half, 2)
            gy = np.linspace(center[1] - half, center[1] + half, 2)
            GX, GY = np.meshgrid(gx, gy)
            GZ = np.zeros_like(GX)
            ax3d.plot_surface(GX, GY, GZ, color="#bbbbbb", alpha=0.12, linewidth=0)

            # Create persistent skeleton artists (avoid ax.cla() compatibility issues)
            skeleton_lines = {}
            joint_scatters = {}
            for p in persons:
                color = PERSON_COLORS.get(p, "#333333")
                lines = []
                for _ in BONES:
                    line, = ax3d.plot([], [], [], color=color, linewidth=2.1, alpha=0.9)
                    lines.append(line)
                skeleton_lines[p] = lines
                scatter = ax3d.scatter([], [], [], s=18, color=color, alpha=0.9,
                                       label=f"person{p}")
                joint_scatters[p] = scatter

            ax3d.legend(loc="upper left", fontsize=9)
            frame_text = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes, fontsize=10)

            def _update(frame_idx):
                frame_idx = int(frame_idx)
                for p in persons:
                    pos = all_positions[p][min(frame_idx, all_positions[p].shape[0] - 1)]
                    for li, (i, j) in enumerate(BONES):
                        xs = [float(pos[i, 0]), float(pos[j, 0])]
                        ys = [float(pos[i, 1]), float(pos[j, 1])]
                        zs = [float(pos[i, 2]), float(pos[j, 2])]
                        skeleton_lines[p][li].set_data_3d(xs, ys, zs)
                    joint_scatters[p]._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
                frame_text.set_text(f"Frame {frame_idx}/{T-1}")
                return []

            ani = FuncAnimation(fig, _update, frames=T, interval=1000 / 20, repeat=False)
            save_path = os.path.join(output_dir, f"{clip_id}_{suffix}.mp4")
            ani.save(save_path, fps=20, dpi=100)
            plt.close()
            print(f"    Saved: {save_path} ({T} frames)")


def main():
    parser = argparse.ArgumentParser(
        description="Retarget generated/GT positions → Newton joint_q via IK"
    )
    parser.add_argument("--clips", nargs="+", default=["4659", "3678"])
    parser.add_argument("--gen-dir", default=None,
                        help="Generated data directory (default: data/reconstructed_dataset/interhuman)")
    parser.add_argument("--gt-data-dir", default=None,
                        help="GT InterHuman directory (default: data/InterHuman)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for retargeted data")
    parser.add_argument("--ik-iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-gt", action="store_true",
                        help="Skip GT processing (only do generated)")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip generated processing (only do GT)")
    parser.add_argument("--run-physics", action="store_true",
                        help="Also run physics analysis after retargeting")
    args = parser.parse_args()

    if args.gt_data_dir is None:
        args.gt_data_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    if args.gen_dir is None:
        args.gen_dir = os.path.join(PROJECT_ROOT, "data", "reconstructed_dataset", "interhuman")

    gt_retarget_dir = os.path.join(PROJECT_ROOT, "data", "retargeted_v2", "gt_from_positions")
    gen_retarget_dir = os.path.join(PROJECT_ROOT, "data", "retargeted_v2", "gen_from_positions")

    if args.output_dir:
        gen_retarget_dir = args.output_dir

    for clip_id in args.clips:
        print(f"\n{'='*60}")
        print(f"Clip {clip_id}")
        print(f"{'='*60}")

        # GT: positions → IK → joint_q
        if not args.skip_gt:
            print(f"\n--- GT (from positions via IK) ---")
            gt_result = process_gt_clip(
                clip_id, args.gt_data_dir, gt_retarget_dir,
                ik_iters=args.ik_iters, device=args.device
            )

        # Generated: positions → IK → joint_q
        if not args.skip_gen:
            print(f"\n--- Generated (from positions via IK) ---")
            gen_result = process_generated_clip(
                clip_id, args.gen_dir, gen_retarget_dir,
                ik_iters=args.ik_iters, device=args.device
            )

    # Run visualization
    print(f"\n{'='*60}")
    print("Running GT vs Gen visualization (position-based IK)...")
    print(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')

    vis_dir = os.path.join(PROJECT_ROOT, "physics_analysis", "visualization_from_positions")
    os.makedirs(vis_dir, exist_ok=True)

    # Import and run our visualization
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "physics_analysis"))
    from visualize_gt_vs_gen import load_retargeted, plot_skeleton_comparison, plot_trajectory_comparison

    for clip_id in args.clips:
        print(f"\nVisualizing clip {clip_id}...")
        gt_pos = load_retargeted(gt_retarget_dir, clip_id, args.device)
        gen_pos = load_retargeted(gen_retarget_dir, clip_id, args.device)

        if gt_pos and gen_pos:
            min_T = min(gt_pos[0].shape[0], gen_pos[0].shape[0])
            frame_indices = np.linspace(0, min_T - 1, 6, dtype=int).tolist()
            plot_skeleton_comparison(gt_pos, gen_pos, clip_id, frame_indices, vis_dir)
            plot_trajectory_comparison(gt_pos, gen_pos, clip_id, vis_dir)

    # Optionally run physics analysis
    if args.run_physics:
        print(f"\n{'='*60}")
        print("Running physics analysis on IK-retargeted generated data...")
        print(f"{'='*60}")
        print("  Note: Physics analysis for position-based data requires")
        print("  custom integration. Use the visualization output for comparison.")

    # ── Render MP4 videos (Newton FK positions → InterMask-style stick figures) ──
    print(f"\n{'='*60}")
    print("Rendering Newton FK videos...")
    print(f"{'='*60}")
    render_newton_videos(args.clips, gt_retarget_dir, gen_retarget_dir,
                         vis_dir, args.device)

    print(f"\nDone! Results saved to:")
    print(f"  GT retargeted:  {gt_retarget_dir}")
    print(f"  Gen retargeted: {gen_retarget_dir}")
    print(f"  Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
