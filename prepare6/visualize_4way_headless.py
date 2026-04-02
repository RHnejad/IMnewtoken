"""
visualize_4way_headless.py — Headless 4-way comparison MP4.

Renders: GT Reference | GT after PPO | Gen Reference | Gen after PPO
using Newton-style colored skeleton visualization (matplotlib + FFMpeg).

Usage:
    # From existing npz (fast, uses saved positions)
    python prepare6/visualize_4way_headless.py --clip-id 161

    # Re-run RL tracker first, then render
    python prepare6/visualize_4way_headless.py --clip-id 161 --run

    # Custom output
    python prepare6/visualize_4way_headless.py --clip-id 880 --output my_video.mp4
"""
import os
import sys
import argparse
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ═══════════════════════════════════════════════════════════════
# 22-joint SMPL skeleton topology (matching positions from RL tracker)
# ═══════════════════════════════════════════════════════════════
SKELETON_BONES = [
    (0, 1), (1, 4), (4, 7), (7, 10),     # Left leg
    (0, 2), (2, 5), (5, 8), (8, 11),     # Right leg
    (0, 3), (3, 6), (6, 9),               # Spine
    (9, 12), (12, 15),                     # Neck/head
    (9, 13), (13, 16), (16, 18), (18, 20), # Left arm
    (9, 14), (14, 17), (17, 19), (19, 21), # Right arm
]

BONE_GROUPS = {
    "spine":     [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)],
    "left_leg":  [(0, 1), (1, 4), (4, 7), (7, 10)],
    "right_leg": [(0, 2), (2, 5), (5, 8), (8, 11)],
    "left_arm":  [(9, 13), (13, 16), (16, 18), (18, 20)],
    "right_arm": [(9, 14), (14, 17), (17, 19), (19, 21)],
}

# Different color schemes per panel
PANEL_COLORS = [
    # GT Reference — blue tones
    {"spine": "#1565C0", "left_leg": "#1976D2", "right_leg": "#2196F3",
     "left_arm": "#42A5F5", "right_arm": "#0D47A1", "joint": "#0D47A1"},
    # GT after PPO — red tones
    {"spine": "#C62828", "left_leg": "#D32F2F", "right_leg": "#F44336",
     "left_arm": "#EF5350", "right_arm": "#B71C1C", "joint": "#B71C1C"},
    # Gen Reference — green tones
    {"spine": "#2E7D32", "left_leg": "#388E3C", "right_leg": "#4CAF50",
     "left_arm": "#66BB6A", "right_arm": "#1B5E20", "joint": "#1B5E20"},
    # Gen after PPO — orange tones
    {"spine": "#E65100", "left_leg": "#F57C00", "right_leg": "#FF9800",
     "left_arm": "#FFA726", "right_arm": "#EF6C00", "joint": "#E65100"},
]

PANEL_LABELS = [
    "GT Reference",
    "GT after PPO",
    "Gen Reference",
    "Gen after PPO",
]

LABEL_COLORS = ["#1565C0", "#C62828", "#2E7D32", "#E65100"]


def draw_skeleton(ax, joints, colors, alpha=0.9, lw=2.5, joint_size=20):
    """Draw skeleton with colored bone groups."""
    for group_name, bones in BONE_GROUPS.items():
        color = colors[group_name]
        for (i, j) in bones:
            if i < len(joints) and j < len(joints):
                ax.plot(
                    [joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]],
                    color=color, alpha=alpha, linewidth=lw,
                )
    ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2],
        c=colors["joint"], s=joint_size, alpha=alpha, depthshade=True, zorder=5,
    )


def load_4way_data(clip_id, person, output_dir, run=False, device="cuda:0",
                   n_envs=64, total_timesteps=200000):
    """Load or compute 4-way position data.

    Returns: (gt_ref, gt_sim, gen_ref, gen_sim, gt_mpjpe, gen_mpjpe, gt_mpjpe_pf, gen_mpjpe_pf, text)
        Each *_ref/*_sim is (T, 22, 3).
    """
    out_dir = os.path.join(output_dir, f"clip_{clip_id}")

    if run:
        from prepare6.visualize_comparison import run_rl_tracker
        print(f"\n--- Running RL tracker for GT ---")
        gt_sim, gt_ref, gt_mpjpe, gt_mpjpe_pf = run_rl_tracker(
            clip_id, "gt", person, device, n_envs, total_timesteps)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "gt_result.npz"),
                 sim_positions=gt_sim, ref_positions=gt_ref,
                 per_frame_mpjpe_mm=gt_mpjpe_pf)

        print(f"\n--- Running RL tracker for Generated ---")
        gen_sim, gen_ref, gen_mpjpe, gen_mpjpe_pf = run_rl_tracker(
            clip_id, "generated", person, device, n_envs, total_timesteps)
        np.savez(os.path.join(out_dir, "gen_result.npz"),
                 sim_positions=gen_sim, ref_positions=gen_ref,
                 per_frame_mpjpe_mm=gen_mpjpe_pf)

        # Load text
        from prepare5.run_phc_tracker import load_clip
        _, text = load_clip(clip_id, "gt")
    else:
        gt_data = np.load(os.path.join(out_dir, "gt_result.npz"))
        gen_data = np.load(os.path.join(out_dir, "gen_result.npz"))
        gt_ref = gt_data['ref_positions']
        gt_sim = gt_data['sim_positions']
        gen_ref = gen_data['ref_positions']
        gen_sim = gen_data['sim_positions']
        gt_mpjpe_pf = gt_data['per_frame_mpjpe_mm']
        gen_mpjpe_pf = gen_data['per_frame_mpjpe_mm']
        gt_mpjpe = gt_mpjpe_pf.mean()
        gen_mpjpe = gen_mpjpe_pf.mean()
        try:
            from prepare5.run_phc_tracker import load_clip
            _, text = load_clip(clip_id, "gt")
        except Exception:
            text = ""

    T = min(gt_ref.shape[0], gen_ref.shape[0])
    return (gt_ref[:T], gt_sim[:T], gen_ref[:T], gen_sim[:T],
            gt_mpjpe, gen_mpjpe, gt_mpjpe_pf[:T], gen_mpjpe_pf[:T], text)


def render_4way_mp4(clip_id, gt_ref, gt_sim, gen_ref, gen_sim,
                    gt_mpjpe, gen_mpjpe, gt_mpjpe_pf, gen_mpjpe_pf,
                    output_path, fps=20, dpi=100, camera="fixed", text=""):
    """Render 4-panel + MPJPE plot MP4."""
    T = gt_ref.shape[0]
    panels = [gt_ref, gt_sim, gen_ref, gen_sim]

    # Global bounds
    all_pos = np.concatenate(panels, axis=0)
    global_min = all_pos.reshape(-1, 3).min(axis=0)
    global_max = all_pos.reshape(-1, 3).max(axis=0)
    center = 0.5 * (global_min + global_max)
    span = max(global_max - global_min) * 0.6
    span = max(span, 1.5)

    fig = plt.figure(figsize=(28, 8))

    # 4 skeleton panels + 1 MPJPE plot
    axes_sk = []
    for i in range(4):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        axes_sk.append(ax)
    ax_mpjpe = fig.add_subplot(1, 5, 5)

    gap = gen_mpjpe - gt_mpjpe
    title = (f"Clip {clip_id} — Physics Plausibility Comparison\n"
             f"GT MPJPE: {gt_mpjpe:.0f}mm | Gen MPJPE: {gen_mpjpe:.0f}mm | "
             f"Gap: {gap:+.0f}mm")
    if text:
        title += f'\n"{text[:80]}"'
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)

    # Setup MPJPE plot (static parts)
    ax_mpjpe.set_xlim(0, T)
    max_err = max(gt_mpjpe_pf.max(), gen_mpjpe_pf.max()) * 1.1
    ax_mpjpe.set_ylim(0, max(max_err, 50))
    ax_mpjpe.set_xlabel("Frame", fontsize=10)
    ax_mpjpe.set_ylabel("MPJPE (mm)", fontsize=10)
    ax_mpjpe.set_title("Per-Frame Tracking Error", fontsize=11)
    ax_mpjpe.grid(True, alpha=0.3)
    gt_line, = ax_mpjpe.plot([], [], color=LABEL_COLORS[1], alpha=0.8,
                              label=f"GT ({gt_mpjpe:.0f}mm)")
    gen_line, = ax_mpjpe.plot([], [], color=LABEL_COLORS[3], alpha=0.8,
                               label=f"Gen ({gen_mpjpe:.0f}mm)")
    gt_marker, = ax_mpjpe.plot([], [], "o", color=LABEL_COLORS[1], markersize=5)
    gen_marker, = ax_mpjpe.plot([], [], "o", color=LABEL_COLORS[3], markersize=5)
    ax_mpjpe.axhline(gt_mpjpe, color=LABEL_COLORS[1], ls="--", alpha=0.3)
    ax_mpjpe.axhline(gen_mpjpe, color=LABEL_COLORS[3], ls="--", alpha=0.3)
    ax_mpjpe.legend(fontsize=9)

    def update(frame):
        for pi, (ax, pos_arr) in enumerate(zip(axes_sk, panels)):
            ax.cla()
            t = min(frame, pos_arr.shape[0] - 1)
            draw_skeleton(ax, pos_arr[t], PANEL_COLORS[pi])

            # Ground plane
            gx = np.linspace(center[0] - span, center[0] + span, 2)
            gy = np.linspace(center[1] - span, center[1] + span, 2)
            GX, GY = np.meshgrid(gx, gy)
            ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.06, color="#999999")

            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(max(0, global_min[2] - 0.1), center[2] + span)
            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)
            ax.set_zlabel("Z", fontsize=8)

            label = f"{PANEL_LABELS[pi]} (f{frame})"
            ax.set_title(label, fontsize=11, fontweight="bold",
                         color=LABEL_COLORS[pi], pad=8)

            if camera == "orbit":
                ax.view_init(elev=18, azim=-90 + frame * 0.5)
            else:
                ax.view_init(elev=18, azim=-90)

        # Update MPJPE traces
        gt_line.set_data(np.arange(frame + 1), gt_mpjpe_pf[:frame + 1])
        gen_line.set_data(np.arange(frame + 1), gen_mpjpe_pf[:frame + 1])
        gt_marker.set_data([frame], [gt_mpjpe_pf[min(frame, T - 1)]])
        gen_marker.set_data([frame], [gen_mpjpe_pf[min(frame, T - 1)]])

        fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.12)

    print(f"Rendering {T} frames to {output_path} ...")
    t0 = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    writer = FFMpegWriter(fps=fps, metadata={"title": f"Clip {clip_id} 4-way"})
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"Saved: {output_path} ({T} frames in {elapsed:.1f}s)")


def render_4way_snapshot(clip_id, gt_ref, gt_sim, gen_ref, gen_sim,
                         gt_mpjpe, gen_mpjpe, output_path, n_frames=6):
    """Save a static PNG grid: 4 rows × n_frames columns."""
    panels = [gt_ref, gt_sim, gen_ref, gen_sim]
    T = gt_ref.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    all_pos = np.concatenate(panels, axis=0)
    global_min = all_pos.reshape(-1, 3).min(axis=0)
    global_max = all_pos.reshape(-1, 3).max(axis=0)
    center = 0.5 * (global_min + global_max)
    span = max(global_max - global_min) * 0.6
    span = max(span, 1.5)

    fig, axes = plt.subplots(4, n_frames, figsize=(4 * n_frames, 16),
                              subplot_kw={"projection": "3d"})

    for row in range(4):
        for col, fi in enumerate(frame_indices):
            ax = axes[row][col]
            t = min(fi, panels[row].shape[0] - 1)
            draw_skeleton(ax, panels[row][t], PANEL_COLORS[row])

            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(max(0, global_min[2] - 0.1), center[2] + span)
            ax.view_init(elev=18, azim=-90)

            title = f"{PANEL_LABELS[row]}\nf{fi}" if col == 0 else f"f{fi}"
            ax.set_title(title, fontsize=9, color=LABEL_COLORS[row])

    gap = gen_mpjpe - gt_mpjpe
    fig.suptitle(
        f"Clip {clip_id} — GT MPJPE: {gt_mpjpe:.0f}mm | "
        f"Gen MPJPE: {gen_mpjpe:.0f}mm | Gap: {gap:+.0f}mm",
        fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Headless 4-way comparison MP4")
    parser.add_argument("--clip-id", type=int, required=True)
    parser.add_argument("--run", action="store_true",
                        help="Run RL tracker first (otherwise load from npz)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", default="output/rl_comparison")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--camera", default="fixed", choices=["fixed", "orbit"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--person", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--skip-mp4", action="store_true", help="Only PNG snapshot")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir, f"clip_{args.clip_id}")

    data = load_4way_data(
        args.clip_id, args.person, os.path.join(PROJECT_ROOT, args.output_dir),
        run=args.run, device=args.device,
        n_envs=args.n_envs, total_timesteps=args.total_timesteps,
    )
    gt_ref, gt_sim, gen_ref, gen_sim, gt_mpjpe, gen_mpjpe, gt_mpjpe_pf, gen_mpjpe_pf, text = data

    print(f"\n  GT MPJPE:  {gt_mpjpe:.1f} mm")
    print(f"  Gen MPJPE: {gen_mpjpe:.1f} mm")
    print(f"  Gap:       {gen_mpjpe - gt_mpjpe:+.1f} mm")

    # Snapshot PNG
    render_4way_snapshot(
        args.clip_id, gt_ref, gt_sim, gen_ref, gen_sim,
        gt_mpjpe, gen_mpjpe,
        os.path.join(out_dir, "newton_snapshot.png"),
    )

    # MP4
    if not args.skip_mp4:
        mp4_path = args.output or os.path.join(out_dir, "newton_4way.mp4")
        render_4way_mp4(
            args.clip_id, gt_ref, gt_sim, gen_ref, gen_sim,
            gt_mpjpe, gen_mpjpe, gt_mpjpe_pf, gen_mpjpe_pf,
            mp4_path, fps=args.fps, dpi=args.dpi, camera=args.camera, text=text,
        )


if __name__ == "__main__":
    main()
