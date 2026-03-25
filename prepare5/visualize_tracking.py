"""
visualize_tracking.py — Visualize PHC tracker results as MP4/GIF.

Generates side-by-side 3D skeleton animations comparing reference motion
(blue) vs physics-simulated motion (red), with per-frame MPJPE overlay.

Usage:
    # From saved .npz result
    python prepare5/visualize_tracking.py \
        --result output/phc_tracker/clip_1129_gt/phc_result.npz

    # From a fresh tracking run
    python prepare5/visualize_tracking.py --clip-id 1129 --source gt

    # Save as MP4 (requires ffmpeg)
    python prepare5/visualize_tracking.py --clip-id 1129 --source gt --mp4
"""
import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 22-joint kinematic chain (from utils/paramUtil.py)
# Each chain = [joint_indices] forming a connected limb
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],        # Right leg: pelvis → r_hip → r_knee → r_ankle → r_foot
    [0, 1, 4, 7, 10],        # Left leg:  pelvis → l_hip → l_knee → l_ankle → l_foot
    [0, 3, 6, 9, 12, 15],    # Spine:     pelvis → spine1 → spine2 → spine3 → neck → head
    [9, 14, 17, 19, 21],     # Right arm: spine3 → r_collar → r_shoulder → r_elbow → r_wrist
    [9, 13, 16, 18, 20],     # Left arm:  spine3 → l_collar → l_shoulder → l_elbow → l_wrist
]

REF_COLORS = ['#2196F3', '#1976D2', '#0D47A1', '#42A5F5', '#1565C0']  # Blue tones
SIM_COLORS = ['#F44336', '#D32F2F', '#B71C1C', '#EF5350', '#C62828']  # Red tones


def setup_3d_axis(ax, title, positions, z_up=True):
    """Configure 3D axis with consistent limits."""
    all_pos = positions.reshape(-1, 3)
    center = all_pos.mean(axis=0)
    extent = max(np.abs(all_pos - center).max(), 0.5)

    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(max(0, center[2] - extent), center[2] + extent)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=20, azim=-60)


def draw_skeleton(ax, joints, colors, linewidth=2.5, alpha=0.9):
    """Draw a 22-joint skeleton on a 3D axis."""
    lines = []
    for ci, chain in enumerate(KINEMATIC_CHAIN):
        color = colors[ci % len(colors)]
        lw = 3.0 if ci < 3 else 2.0
        line, = ax.plot3D(
            joints[chain, 0], joints[chain, 1], joints[chain, 2],
            color=color, linewidth=lw * linewidth / 2.5, alpha=alpha,
        )
        lines.append(line)
    # Joint dots
    scatter = ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2],
        s=15, c=colors[0], alpha=alpha, zorder=5,
    )
    return lines, scatter


def create_comparison_animation(ref_positions, sim_positions, output_path,
                                 fps=30, title="PHC Tracking", mpjpe_per_frame=None):
    """Create side-by-side animation: reference (blue) vs simulated (red).

    Args:
        ref_positions: (T, 22, 3) reference body positions
        sim_positions: (T, 22, 3) simulated body positions
        output_path: path to save (png sequence or mp4)
        fps: animation framerate
        title: plot title
        mpjpe_per_frame: (T,) optional per-frame MPJPE in mm
    """
    T = min(ref_positions.shape[0], sim_positions.shape[0])
    all_pos = np.concatenate([ref_positions[:T], sim_positions[:T]], axis=0)

    fig = plt.figure(figsize=(16, 7))

    # Left: overlay view
    ax1 = fig.add_subplot(121, projection='3d')
    setup_3d_axis(ax1, "Overlay: Reference (blue) vs Simulated (red)", all_pos)

    # Right: error heatmap or MPJPE plot
    if mpjpe_per_frame is not None:
        ax2 = fig.add_subplot(122)
        ax2.set_xlim(0, T)
        ax2.set_ylim(0, max(mpjpe_per_frame[:T].max() * 1.1, 50))
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('MPJPE (mm)')
        ax2.set_title('Per-Frame Tracking Error')
        ax2.grid(True, alpha=0.3)
        mpjpe_line, = ax2.plot([], [], 'r-', alpha=0.8)
        frame_marker, = ax2.plot([], [], 'ro', markersize=8)
        mean_mpjpe = mpjpe_per_frame[:T].mean()
        ax2.axhline(y=mean_mpjpe, color='gray', linestyle='--', alpha=0.5,
                     label=f'Mean: {mean_mpjpe:.0f} mm')
        ax2.legend(fontsize=10)
    else:
        ax2 = fig.add_subplot(122, projection='3d')
        setup_3d_axis(ax2, "Simulated Motion", sim_positions[:T])

    fig.suptitle(title, fontsize=13, y=0.98)

    def update(frame):
        ax1.cla()
        setup_3d_axis(ax1, f"Frame {frame}/{T}", all_pos)

        # Draw reference skeleton (blue)
        draw_skeleton(ax1, ref_positions[frame], REF_COLORS, alpha=0.6)
        # Draw simulated skeleton (red)
        draw_skeleton(ax1, sim_positions[frame], SIM_COLORS, alpha=0.9)

        if mpjpe_per_frame is not None:
            mpjpe_line.set_data(np.arange(frame + 1), mpjpe_per_frame[:frame + 1])
            frame_marker.set_data([frame], [mpjpe_per_frame[frame]])
        else:
            ax2.cla()
            setup_3d_axis(ax2, f"Simulated (frame {frame})", sim_positions[:T])
            draw_skeleton(ax2, sim_positions[frame], SIM_COLORS)

        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    if output_path.endswith('.mp4'):
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    elif output_path.endswith('.gif'):
        anim.save(output_path, writer='pillow', fps=fps, dpi=80)
    else:
        # Save key frames as PNGs
        for frame_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
            update(frame_idx)
            frame_path = output_path.replace('.png', f'_frame{frame_idx:04d}.png')
            fig.savefig(frame_path, dpi=120, bbox_inches='tight')

    plt.close()
    print(f"  Saved: {output_path}")


def create_snapshot_grid(ref_positions, sim_positions, output_path,
                          n_frames=6, title="PHC Tracking Snapshots"):
    """Create a grid of skeleton snapshots at evenly-spaced frames.

    Single static image showing tracking quality at multiple timepoints.
    """
    T = min(ref_positions.shape[0], sim_positions.shape[0])
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)
    all_pos = np.concatenate([ref_positions[:T], sim_positions[:T]], axis=0)

    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 5),
                              subplot_kw={'projection': '3d'})
    if n_frames == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, y=1.02)

    for i, (ax, fi) in enumerate(zip(axes, frame_indices)):
        setup_3d_axis(ax, f"Frame {fi}", all_pos)
        draw_skeleton(ax, ref_positions[fi], REF_COLORS, alpha=0.5, linewidth=2.0)
        draw_skeleton(ax, sim_positions[fi], SIM_COLORS, alpha=0.9, linewidth=2.5)

        # Per-joint error lines
        for j in range(ref_positions.shape[1]):
            err = np.linalg.norm(ref_positions[fi, j] - sim_positions[fi, j])
            if err > 0.05:  # Only show lines for >5cm error
                ax.plot3D(
                    [ref_positions[fi, j, 0], sim_positions[fi, j, 0]],
                    [ref_positions[fi, j, 1], sim_positions[fi, j, 1]],
                    [ref_positions[fi, j, 2], sim_positions[fi, j, 2]],
                    'k--', alpha=0.3, linewidth=0.8,
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PHC tracking results")
    parser.add_argument("--result", type=str, default=None,
                        help="Path to phc_result.npz from a previous run")
    parser.add_argument("--clip-id", type=int, default=1129,
                        help="Clip ID (if running fresh)")
    parser.add_argument("--source", choices=["gt", "generated"], default="gt",
                        help="Data source (if running fresh)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mp4", action="store_true",
                        help="Save as MP4 (requires ffmpeg)")
    parser.add_argument("--gif", action="store_true",
                        help="Save as GIF")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gain-preset", choices=["phc", "old"], default="phc")

    args = parser.parse_args()

    if args.result:
        # Load from saved .npz
        data = np.load(args.result)
        ref_pos = data['ref_positions']
        sim_pos = data['sim_positions']
        output_dir = os.path.dirname(args.result)
        title = f"PHC Tracking — {os.path.basename(os.path.dirname(args.result))}"

        # Try to load per-frame MPJPE
        mpjpe = None
        metrics_path = os.path.join(output_dir, 'metrics.json')
        if os.path.isfile(metrics_path):
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            title += f" (MPJPE={metrics.get('mpjpe_mm', 0):.0f}mm)"
    else:
        # Run fresh tracking
        from prepare5.run_phc_tracker import load_clip, retarget_person
        from prepare5.phc_tracker import PHCTracker

        persons, text = load_clip(args.clip_id, args.source)
        if persons is None:
            print(f"ERROR: Could not load clip {args.clip_id}")
            return

        joint_q, betas = retarget_person(persons[0], args.source, device=args.device)
        tracker = PHCTracker(device=args.device, gain_preset=args.gain_preset,
                              verbose=True)
        result = tracker.track(joint_q, betas)

        ref_pos = result['ref_positions']
        sim_pos = result['sim_positions']
        output_dir = args.output_dir or f"output/phc_tracker/clip_{args.clip_id}_{args.source}"
        title = f"PHC Tracking — Clip {args.clip_id} ({args.source})\n{text}\nMPJPE={result['mpjpe_mm']:.0f}mm"
        mpjpe = result.get('per_frame_mpjpe_mm')

    os.makedirs(output_dir, exist_ok=True)

    # Static snapshot grid
    create_snapshot_grid(
        ref_pos, sim_pos,
        os.path.join(output_dir, "tracking_snapshots.png"),
        title=title.split('\n')[0],
    )

    # Per-frame MPJPE if available
    if mpjpe is None:
        diff = ref_pos - sim_pos
        mpjpe = np.linalg.norm(diff, axis=-1).mean(axis=1) * 1000

    # Animation
    if args.mp4:
        ext, label = '.mp4', 'MP4'
    elif args.gif:
        ext, label = '.gif', 'GIF'
    else:
        ext, label = '.png', 'PNG keyframes'

    anim_path = os.path.join(output_dir, f"tracking_animation{ext}")
    print(f"\n  Generating {label} animation ({ref_pos.shape[0]} frames)...")
    create_comparison_animation(
        ref_pos, sim_pos, anim_path,
        title=title.split('\n')[0],
        mpjpe_per_frame=mpjpe,
    )


if __name__ == "__main__":
    main()
