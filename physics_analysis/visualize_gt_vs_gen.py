#!/usr/bin/env python
"""
Offline Newton skeleton visualization for GT vs Generated motions.
Renders skeleton keyframes to PNG using matplotlib (no display needed).
Uses retargeted joint_q data with Newton FK.

Usage:
    conda activate mimickit
    python physics_analysis/visualize_gt_vs_gen.py --clips 4659 3678
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warp as wp
import newton
from prepare2.retarget import get_or_create_xml, SMPL_TO_NEWTON, N_SMPL_JOINTS

# SMPL 22-joint skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),     # pelvis → hips, spine1
    (1, 4), (2, 5),             # hip → knee
    (4, 7), (5, 8),             # knee → ankle
    (7, 10), (8, 11),           # ankle → foot
    (3, 6), (6, 9),             # spine chain
    (9, 12), (9, 13), (9, 14),  # spine3 → neck, collarbones
    (12, 15),                    # neck → head
    (13, 16), (14, 17),         # collar → shoulder
    (16, 18), (17, 19),         # shoulder → elbow
    (18, 20), (19, 21),         # elbow → wrist
]


def compute_fk_positions(betas, joint_q, device="cuda:0"):
    """Compute FK positions from joint_q using Newton."""
    xml_path = get_or_create_xml(betas)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(xml_path, enable_self_collisions=False)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    T = joint_q.shape[0]
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
    positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    
    for t in range(T):
        state.joint_q = wp.array(joint_q[t].astype(np.float32), dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)
        for j in range(N_SMPL_JOINTS):
            positions[t, j] = body_q[SMPL_TO_NEWTON[j], :3]

    return positions


def plot_skeleton_comparison(gt_positions_list, gen_positions_list, clip_id,
                             frame_indices, save_dir, n_persons=2):
    """Plot skeleton keyframes: GT (top) vs Generated (bottom)."""
    n_frames = len(frame_indices)
    fig = plt.figure(figsize=(5 * n_frames, 12))

    for row, (label, positions_list, color_set) in enumerate([
        ("GT", gt_positions_list, ['tab:blue', 'tab:orange']),
        ("Generated", gen_positions_list, ['tab:red', 'tab:purple']),
    ]):
        for col, fidx in enumerate(frame_indices):
            ax = fig.add_subplot(2, n_frames, row * n_frames + col + 1, projection='3d')

            for p_idx in range(min(n_persons, len(positions_list))):
                pos = positions_list[p_idx]
                if fidx >= pos.shape[0]:
                    continue
                joints = pos[fidx]  # (22, 3)
                color = color_set[p_idx]

                # Plot joints
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                          c=color, s=20, alpha=0.8)

                # Plot bones
                for i, j in SKELETON_CONNECTIONS:
                    ax.plot([joints[i, 0], joints[j, 0]],
                           [joints[i, 1], joints[j, 1]],
                           [joints[i, 2], joints[j, 2]],
                           c=color, alpha=0.6, linewidth=1.5)

            ax.set_title(f"{label} f={fidx}", fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set consistent axis limits
            all_pos = np.concatenate([p[min(fidx, p.shape[0]-1):min(fidx, p.shape[0]-1)+1]
                                      for p in positions_list if p.shape[0] > 0], axis=1)
            if all_pos.size > 0:
                center = all_pos.mean(axis=(0, 1))
                max_range = max(all_pos.max(axis=(0, 1)) - all_pos.min(axis=(0, 1))) * 0.6
                max_range = max(max_range, 1.0)
                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[1] - max_range, center[1] + max_range)
                ax.set_zlim(0, 2 * max_range)
            ax.view_init(elev=15, azim=45)

    fig.suptitle(f"Clip {clip_id}: GT (top) vs Generated (bottom)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"skeleton_comparison_clip_{clip_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_trajectory_comparison(gt_positions_list, gen_positions_list, clip_id, save_dir):
    """Plot root/COM trajectory comparison: GT vs Gen, top-down view."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (label, positions_list, colors) in zip(axes, [
        ("GT", gt_positions_list, ['tab:blue', 'tab:orange']),
        ("Generated", gen_positions_list, ['tab:red', 'tab:purple']),
    ]):
        for p_idx, pos in enumerate(positions_list):
            # Root joint (pelvis) trajectory
            root = pos[:, 0, :]  # (T, 3)
            ax.plot(root[:, 0], root[:, 1], c=colors[p_idx],
                   label=f'P{p_idx+1} root', linewidth=2, alpha=0.8)
            ax.scatter(root[0, 0], root[0, 1], c=colors[p_idx], marker='o', s=100, zorder=5)
            ax.scatter(root[-1, 0], root[-1, 1], c=colors[p_idx], marker='x', s=100, zorder=5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f"{label} — Clip {clip_id} ({positions_list[0].shape[0]} frames)")
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Root Trajectory — Clip {clip_id}: GT vs Generated (top-down XY)", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"trajectory_comparison_clip_{clip_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def load_retargeted(data_dir, clip_id, device="cuda:0"):
    """Load retargeted joint_q and compute FK positions."""
    positions_list = []
    for p_idx in range(2):
        jq_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
        betas_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_betas.npy")

        if not os.path.exists(jq_path):
            print(f"  WARNING: {jq_path} not found")
            continue

        jq = np.load(jq_path)
        betas = np.load(betas_path)
        positions = compute_fk_positions(betas, jq, device=device)
        positions_list.append(positions)
        print(f"  Loaded P{p_idx+1}: {jq.shape[0]} frames → FK positions {positions.shape}")

    return positions_list


def main():
    parser = argparse.ArgumentParser(description="Offline GT vs Gen skeleton visualization")
    parser.add_argument("--clips", nargs="+", default=["4659", "3678"])
    parser.add_argument("--gt-dir", default=os.path.join(PROJECT_ROOT, "data/retargeted_v2/gt_test"))
    parser.add_argument("--gen-dir", default=os.path.join(PROJECT_ROOT, "data/retargeted_v2/gen_test"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "physics_analysis/visualization_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-keyframes", type=int, default=6, help="Number of keyframes to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for clip_id in args.clips:
        print(f"\n{'='*60}")
        print(f"Clip {clip_id}")
        print(f"{'='*60}")

        print("  Loading GT retargeted data...")
        gt_pos = load_retargeted(args.gt_dir, clip_id, args.device)

        print("  Loading Generated retargeted data...")
        gen_pos = load_retargeted(args.gen_dir, clip_id, args.device)

        if not gt_pos or not gen_pos:
            print(f"  Skipping clip {clip_id}: missing data")
            continue

        # Select keyframes (distributed across the shorter sequence)
        min_T = min(gt_pos[0].shape[0], gen_pos[0].shape[0])
        frame_indices = np.linspace(0, min_T - 1, args.n_keyframes, dtype=int).tolist()
        print(f"  Keyframes: {frame_indices} (min_T={min_T})")

        # Plot skeleton comparison
        plot_skeleton_comparison(gt_pos, gen_pos, clip_id, frame_indices, args.output_dir)

        # Plot trajectory comparison
        plot_trajectory_comparison(gt_pos, gen_pos, clip_id, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
