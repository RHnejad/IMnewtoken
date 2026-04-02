"""
visualize_comparison.py — 4-way comparison visualization:
  1. GT reference motion (blue)
  2. GT after PPO tracking (red)
  3. Generated reference motion (green)
  4. Generated after PPO tracking (orange)

Saves MP4 animation and snapshot grid PNG.

Usage:
    python prepare6/visualize_comparison.py --clip-id 161

    # Custom timesteps
    python prepare6/visualize_comparison.py --clip-id 161 --total-timesteps 500000
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

# 22-joint kinematic chain
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],        # Right leg
    [0, 1, 4, 7, 10],        # Left leg
    [0, 3, 6, 9, 12, 15],    # Spine
    [9, 14, 17, 19, 21],     # Right arm
    [9, 13, 16, 18, 20],     # Left arm
]

COLOR_SETS = {
    'gt_ref':  ['#2196F3', '#1976D2', '#0D47A1', '#42A5F5', '#1565C0'],  # Blue
    'gt_sim':  ['#F44336', '#D32F2F', '#B71C1C', '#EF5350', '#C62828'],  # Red
    'gen_ref': ['#4CAF50', '#388E3C', '#1B5E20', '#66BB6A', '#2E7D32'],  # Green
    'gen_sim': ['#FF9800', '#F57C00', '#E65100', '#FFA726', '#EF6C00'],  # Orange
}


def setup_ax(ax, title, all_pos):
    center = all_pos.reshape(-1, 3).mean(axis=0)
    extent = max(np.abs(all_pos.reshape(-1, 3) - center).max(), 0.5)
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(max(0, center[2] - extent), center[2] + extent)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=10, pad=5)
    ax.view_init(elev=20, azim=-60)


def draw_skeleton(ax, joints, colors, lw=2.5, alpha=0.9):
    for ci, chain in enumerate(KINEMATIC_CHAIN):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2],
                  color=colors[ci % len(colors)], linewidth=lw, alpha=alpha)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=12, c=colors[0], alpha=alpha, zorder=5)


def run_rl_tracker(clip_id, source, person, device, n_envs, total_timesteps):
    """Run RL tracker and return (sim_positions, ref_positions, mpjpe_mm)."""
    from prepare5.run_phc_tracker import load_clip, retarget_person
    from prepare6.rl_tracker import RLTracker

    persons, text = load_clip(clip_id, source)
    joint_q, betas = retarget_person(persons[person], source, device=device)
    print(f"  {source.upper()}: T={joint_q.shape[0]} frames, text={text[:60]}")

    tracker = RLTracker(
        device=device, n_envs=n_envs,
        total_timesteps=total_timesteps, verbose=True,
    )
    result = tracker.train_and_evaluate(joint_q, betas)
    return result['sim_positions'], result['ref_positions'], result['mpjpe_mm'], result['per_frame_mpjpe_mm']


def create_4way_animation(gt_ref, gt_sim, gen_ref, gen_sim,
                           gt_mpjpe_pf, gen_mpjpe_pf,
                           gt_mpjpe, gen_mpjpe,
                           output_path, fps=30, clip_id=0):
    """Create 4-panel MP4: GT ref | GT sim | Gen ref | Gen sim + MPJPE plot."""
    T = min(gt_ref.shape[0], gen_ref.shape[0])
    all_pos = np.concatenate([gt_ref[:T], gt_sim[:T], gen_ref[:T], gen_sim[:T]])

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Clip {clip_id} — Physics Plausibility Comparison", fontsize=14, y=0.98)

    # 4 skeleton panels (top row)
    ax_gt_ref = fig.add_subplot(231, projection='3d')
    ax_gt_sim = fig.add_subplot(232, projection='3d')
    ax_gen_ref = fig.add_subplot(234, projection='3d')
    ax_gen_sim = fig.add_subplot(235, projection='3d')

    # MPJPE plot (right column, spans both rows)
    ax_mpjpe = fig.add_subplot(133)
    ax_mpjpe.set_xlim(0, T)
    max_mpjpe = max(gt_mpjpe_pf[:T].max(), gen_mpjpe_pf[:T].max()) * 1.1
    ax_mpjpe.set_ylim(0, max(max_mpjpe, 100))
    ax_mpjpe.set_xlabel('Frame')
    ax_mpjpe.set_ylabel('MPJPE (mm)')
    ax_mpjpe.set_title('Per-Frame Tracking Error')
    ax_mpjpe.grid(True, alpha=0.3)
    gt_line, = ax_mpjpe.plot([], [], 'r-', alpha=0.8, label=f'GT sim ({gt_mpjpe:.0f}mm)')
    gen_line, = ax_mpjpe.plot([], [], color='#FF9800', alpha=0.8, label=f'Gen sim ({gen_mpjpe:.0f}mm)')
    gt_marker, = ax_mpjpe.plot([], [], 'ro', markersize=6)
    gen_marker, = ax_mpjpe.plot([], [], 'o', color='#FF9800', markersize=6)
    ax_mpjpe.axhline(gt_mpjpe, color='red', linestyle='--', alpha=0.4)
    ax_mpjpe.axhline(gen_mpjpe, color='#FF9800', linestyle='--', alpha=0.4)
    ax_mpjpe.legend(fontsize=10)

    def update(frame):
        for ax, pos, title, ckey in [
            (ax_gt_ref,  gt_ref,  f"GT Reference (frame {frame})",   'gt_ref'),
            (ax_gt_sim,  gt_sim,  f"GT after PPO (frame {frame})",   'gt_sim'),
            (ax_gen_ref, gen_ref, f"Gen Reference (frame {frame})",  'gen_ref'),
            (ax_gen_sim, gen_sim, f"Gen after PPO (frame {frame})",  'gen_sim'),
        ]:
            ax.cla()
            setup_ax(ax, title, all_pos)
            draw_skeleton(ax, pos[min(frame, pos.shape[0]-1)], COLOR_SETS[ckey])

        gt_line.set_data(np.arange(frame+1), gt_mpjpe_pf[:frame+1])
        gen_line.set_data(np.arange(frame+1), gen_mpjpe_pf[:frame+1])
        gt_marker.set_data([frame], [gt_mpjpe_pf[min(frame, T-1)]])
        gen_marker.set_data([frame], [gen_mpjpe_pf[min(frame, T-1)]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close()
    print(f"  Saved MP4: {output_path}")


def create_4way_snapshot(gt_ref, gt_sim, gen_ref, gen_sim,
                          gt_mpjpe, gen_mpjpe,
                          output_path, clip_id=0, n_frames=6):
    """Create a snapshot grid showing 4 versions at evenly-spaced frames."""
    T = min(gt_ref.shape[0], gen_ref.shape[0])
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)
    all_pos = np.concatenate([gt_ref[:T], gt_sim[:T], gen_ref[:T], gen_sim[:T]])

    fig, axes = plt.subplots(4, n_frames, figsize=(4 * n_frames, 16),
                              subplot_kw={'projection': '3d'})

    row_configs = [
        ("GT Reference",       gt_ref,  'gt_ref'),
        (f"GT after PPO ({gt_mpjpe:.0f}mm)",   gt_sim,  'gt_sim'),
        ("Gen Reference",      gen_ref, 'gen_ref'),
        (f"Gen after PPO ({gen_mpjpe:.0f}mm)", gen_sim, 'gen_sim'),
    ]

    for row, (label, pos, ckey) in enumerate(row_configs):
        for col, fi in enumerate(frame_indices):
            ax = axes[row][col]
            setup_ax(ax, f"{label}\nf{fi}" if col == 0 else f"f{fi}", all_pos)
            draw_skeleton(ax, pos[min(fi, pos.shape[0]-1)], COLOR_SETS[ckey])

    fig.suptitle(f"Clip {clip_id} — 4-Way Physics Comparison\n"
                 f"GT MPJPE: {gt_mpjpe:.0f}mm | Gen MPJPE: {gen_mpjpe:.0f}mm | "
                 f"Gap: {gen_mpjpe - gt_mpjpe:+.0f}mm",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved PNG: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="4-way comparison visualization")
    parser.add_argument("--clip-id", type=int, default=161)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--output-dir", default="output/rl_comparison")
    parser.add_argument("--person", type=int, default=0)
    parser.add_argument("--skip-mp4", action="store_true", help="Only save PNG snapshot")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir, f"clip_{args.clip_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Check if we already have batch results we can reuse
    batch_path = os.path.join(PROJECT_ROOT, "output", "rl_batch_eval", "results.json")
    gt_cached = gen_cached = None

    if os.path.isfile(batch_path):
        import json
        with open(batch_path) as f:
            batch = json.load(f)
        # Check for saved npz files from batch
        for r in batch.get("results", []):
            if r["clip_id"] == args.clip_id and r["person"] == args.person:
                tag = f"clip_{args.clip_id}_{r['source']}_p{args.person}"
                npz = os.path.join(PROJECT_ROOT, "output", "rl_batch_eval", tag, "rl_result.npz")
                if os.path.isfile(npz):
                    if r["source"] == "gt":
                        gt_cached = npz
                    else:
                        gen_cached = npz

    # Run GT tracker
    print(f"\n{'='*60}")
    print(f"Running RL tracker for clip {args.clip_id}")
    print(f"{'='*60}")

    print(f"\n--- GT ---")
    gt_sim, gt_ref, gt_mpjpe, gt_mpjpe_pf = run_rl_tracker(
        args.clip_id, "gt", args.person,
        args.device, args.n_envs, args.total_timesteps,
    )

    # Save GT npz
    np.savez(os.path.join(out_dir, "gt_result.npz"),
             sim_positions=gt_sim, ref_positions=gt_ref,
             per_frame_mpjpe_mm=gt_mpjpe_pf)

    print(f"\n--- Generated ---")
    gen_sim, gen_ref, gen_mpjpe, gen_mpjpe_pf = run_rl_tracker(
        args.clip_id, "generated", args.person,
        args.device, args.n_envs, args.total_timesteps,
    )

    # Save Gen npz
    np.savez(os.path.join(out_dir, "gen_result.npz"),
             sim_positions=gen_sim, ref_positions=gen_ref,
             per_frame_mpjpe_mm=gen_mpjpe_pf)

    # Truncate to same length
    T = min(gt_ref.shape[0], gen_ref.shape[0])
    gt_ref  = gt_ref[:T];  gt_sim  = gt_sim[:T];  gt_mpjpe_pf  = gt_mpjpe_pf[:T]
    gen_ref = gen_ref[:T]; gen_sim = gen_sim[:T]; gen_mpjpe_pf = gen_mpjpe_pf[:T]

    print(f"\n--- Generating visualizations ---")
    print(f"  GT MPJPE:  {gt_mpjpe:.1f} mm")
    print(f"  Gen MPJPE: {gen_mpjpe:.1f} mm")
    print(f"  Gap:       {gen_mpjpe - gt_mpjpe:+.1f} mm")

    # Snapshot grid
    create_4way_snapshot(
        gt_ref, gt_sim, gen_ref, gen_sim,
        gt_mpjpe, gen_mpjpe,
        os.path.join(out_dir, "comparison_snapshot.png"),
        clip_id=args.clip_id,
    )

    # MP4 animation
    if not args.skip_mp4:
        create_4way_animation(
            gt_ref, gt_sim, gen_ref, gen_sim,
            gt_mpjpe_pf, gen_mpjpe_pf,
            gt_mpjpe, gen_mpjpe,
            os.path.join(out_dir, "comparison_animation.mp4"),
            clip_id=args.clip_id,
        )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
