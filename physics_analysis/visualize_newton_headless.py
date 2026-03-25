#!/usr/bin/env python
"""
Headless Newton skeleton visualization → MP4.

Renders retargeted joint_q data on per-subject Newton skeletons to MP4 video,
with text labels, ground plane, and support for comparing multiple versions
(e.g., GT vs Generated) side-by-side in one video.

Usage (conda activate mimickit):
    # Single clip, one version
    python physics_analysis/visualize_newton_headless.py \
        --clip 4659 \
        --data-dir data/retargeted_v2/gt_from_positions \
        --label "GT" \
        --output physics_analysis/visualization_results/4659_gt.mp4

    # Compare GT vs Generated side-by-side (2-panel)
    python physics_analysis/visualize_newton_headless.py \
        --clip 4659 \
        --data-dir data/retargeted_v2/gt_from_positions data/retargeted_v2/gen_from_positions \
        --label "GT (position-IK)" "Generated (position-IK)" \
        --output physics_analysis/visualization_results/4659_compare.mp4

    # Compare 3 versions
    python physics_analysis/visualize_newton_headless.py \
        --clip 3678 \
        --data-dir data/retargeted_v2/gt_from_positions \
                   data/retargeted_v2/gen_from_positions \
                   data/retargeted_v2/interhuman \
        --label "GT (pos-IK)" "Gen (pos-IK)" "GT (rot-transfer)" \
        --output physics_analysis/visualization_results/3678_3way.mp4

    # Options
    --fps 30        Playback framerate (default: 20)
    --dpi 120       Render resolution (default: 100)
    --camera fixed  Fixed camera (default), or 'orbit' for rotating camera
    --annots        Show motion text description from annotations
"""

import os
import sys
import argparse
import warnings
import numpy as np
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter

import warp as wp
wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton

# ═══════════════════════════════════════════════════════════════
# Newton 24-body skeleton topology
# ═══════════════════════════════════════════════════════════════
# Body indices in Newton model (after add_mjcf with SMPL XML):
#   0=Pelvis, 1=L_Hip, 2=L_Knee, 3=L_Ankle, 4=L_Toe,
#   5=R_Hip, 6=R_Knee, 7=R_Ankle, 8=R_Toe,
#   9=Torso, 10=Spine, 11=Chest,
#   12=Neck, 13=Head,
#   14=L_Thorax, 15=L_Shoulder, 16=L_Elbow, 17=L_Wrist, 18=L_Hand,
#   19=R_Thorax, 20=R_Shoulder, 21=R_Elbow, 22=R_Wrist, 23=R_Hand
SKELETON_BONES = [
    # Left leg
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Right leg
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Spine
    (0, 9), (9, 10), (10, 11),
    # Neck/head
    (11, 12), (12, 13),
    # Left arm
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    # Right arm
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23),
]

BONE_GROUPS = {
    "spine":    [(0, 9), (9, 10), (10, 11), (11, 12), (12, 13)],
    "left_leg": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "right_leg":[(0, 5), (5, 6), (6, 7), (7, 8)],
    "left_arm": [(11, 14), (14, 15), (15, 16), (16, 17), (17, 18)],
    "right_arm":[(11, 19), (19, 20), (20, 21), (21, 22), (22, 23)],
}

# Colors per person (for multi-person skeletons within one version)
PERSON_COLORS = [
    {"spine": "#2222CC", "left_leg": "#CC2222", "right_leg": "#22AA22",
     "left_arm": "#DD7700", "right_arm": "#8822AA", "joint": "#333333"},
    {"spine": "#6666FF", "left_leg": "#FF6666", "right_leg": "#66DD66",
     "left_arm": "#FFAA44", "right_arm": "#CC66FF", "joint": "#666666"},
]


def get_or_create_xml(betas):
    """Get or create per-subject SMPL XML. Falls back to generic."""
    try:
        from prepare2.gen_smpl_xml import generate_smpl_xml
        xml_dir = os.path.join(PROJECT_ROOT, "prepare2", "xml_cache")
        os.makedirs(xml_dir, exist_ok=True)
        betas_key = "_".join(f"{b:.4f}" for b in betas.flatten()[:10])
        xml_path = os.path.join(xml_dir, f"smpl_{hash(betas_key) & 0xFFFFFFFF:08x}.xml")
        if not os.path.exists(xml_path):
            xml_path = generate_smpl_xml(betas, output_path=xml_path)
        return xml_path
    except Exception:
        return os.path.join(PROJECT_ROOT, "prepare", "assets", "smpl.xml")


def load_version_data(clip_id, data_dir, device="cuda:0"):
    """Load joint_q + betas for a clip from a data directory.
    
    Returns:
        dict with 'persons': list of {joint_q, betas, xml_path, n_frames}
    """
    persons = []
    for p_idx in [0, 1]:
        jq_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
        betas_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_betas.npy")
        
        if not os.path.exists(jq_path):
            continue
        
        jq = np.load(jq_path).astype(np.float32)
        betas = np.load(betas_path).astype(np.float64) if os.path.exists(betas_path) else np.zeros(10)
        xml_path = get_or_create_xml(betas)
        
        persons.append({
            "joint_q": jq,
            "betas": betas,
            "xml_path": xml_path,
            "n_frames": jq.shape[0],
            "person_idx": p_idx,
        })
    
    if not persons:
        raise FileNotFoundError(f"No joint_q files found for clip {clip_id} in {data_dir}")
    
    return {"persons": persons}


def fk_all_frames(persons, device="cuda:0"):
    """Run Newton FK on all persons, return body positions per frame.
    
    Returns:
        list of np.array (T, 24, 3) per person
    """
    all_positions = []
    
    for pdata in persons:
        jq = pdata["joint_q"]
        T = jq.shape[0]
        
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(pdata["xml_path"], enable_self_collisions=False)
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        
        state = model.state()
        jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
        
        positions = np.zeros((T, 24, 3), dtype=np.float32)
        for t in range(T):
            state.joint_q = wp.array(jq[t], dtype=wp.float32, device=device)
            jqd.zero_()
            newton.eval_fk(model, state.joint_q, jqd, state)
            bq = state.body_q.numpy().reshape(-1, 7)
            # First 24 bodies are the skeleton (exclude ground plane body)
            positions[t] = bq[:24, :3]
        
        all_positions.append(positions)
    
    return all_positions


def load_annotation(clip_id):
    """Try to load motion text description."""
    annots_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman", "annots")
    annots_path = os.path.join(annots_dir, f"{clip_id}.txt")
    if os.path.exists(annots_path):
        with open(annots_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[0] if lines else None
    return None


def draw_skeleton(ax, positions, person_idx=0, alpha=1.0, lw=2.5, joint_size=20):
    """Draw a skeleton on a 3D axis with colored bone groups."""
    colors = PERSON_COLORS[person_idx % len(PERSON_COLORS)]
    
    # Draw bones by group
    for group_name, bones in BONE_GROUPS.items():
        color = colors[group_name]
        for (i, j) in bones:
            if i < len(positions) and j < len(positions):
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]],
                    color=color, alpha=alpha, linewidth=lw,
                )
    
    # Draw joints
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=colors["joint"], s=joint_size, alpha=alpha, depthshade=True, zorder=5,
    )


def render_comparison_mp4(
    clip_id,
    versions,       # list of dicts: {label, positions: [(T,24,3), ...], n_frames}
    output_path,
    fps=20,
    dpi=100,
    camera="fixed",
    annotation=None,
):
    """Render one or more versions side-by-side to MP4."""
    n_versions = len(versions)
    
    # Compute global frame count (min across all versions)
    T = min(v["n_frames"] for v in versions)
    
    # Compute global bounds across all versions for consistent axes
    all_pos = []
    for v in versions:
        for pos in v["positions"]:
            all_pos.append(pos[:T])
    all_pos = np.concatenate(all_pos, axis=0)  # (sumT, 24, 3)
    
    global_min = all_pos.reshape(-1, 3).min(axis=0)
    global_max = all_pos.reshape(-1, 3).max(axis=0)
    center = 0.5 * (global_min + global_max)
    span = max(global_max - global_min) * 0.6
    span = max(span, 1.5)
    
    # Create figure with subplots (one per version)
    fig_width = 7 * n_versions
    fig_height = 7
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    axes = []
    for i in range(n_versions):
        ax = fig.add_subplot(1, n_versions, i + 1, projection="3d")
        axes.append(ax)
    
    # Title
    title_parts = [f"Clip {clip_id}"]
    if annotation:
        title_parts.append(f'"{annotation[:80]}"')
    title_parts.append(f"{T} frames @ {fps} fps ({T/fps:.1f}s)")
    fig.suptitle(" — ".join(title_parts), fontsize=13, fontweight="bold", y=0.98)
    
    # Version label colors for multi-version comparison
    VERSION_COLORS = ["#006600", "#CC0000", "#0000CC", "#CC6600", "#6600CC"]
    
    def update(frame_idx):
        for vi, (ax, version) in enumerate(zip(axes, versions)):
            ax.cla()
            
            # Version label
            label = version["label"]
            color = VERSION_COLORS[vi % len(VERSION_COLORS)]
            ax.set_title(label, fontsize=14, fontweight="bold", color=color, pad=10)
            
            # Draw each person's skeleton
            for pi, pos in enumerate(version["positions"]):
                t = min(frame_idx, pos.shape[0] - 1)
                draw_skeleton(ax, pos[t], person_idx=pi, alpha=0.9 if pi == 0 else 0.6)
            
            # Ground plane
            gx = np.linspace(center[0] - span, center[0] + span, 2)
            gy = np.linspace(center[1] - span, center[1] + span, 2)
            GX, GY = np.meshgrid(gx, gy)
            GZ = np.zeros_like(GX)
            ax.plot_surface(GX, GY, GZ, alpha=0.08, color="#999999")
            
            # Axis limits (consistent across versions)
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(max(0, global_min[2] - 0.1), center[2] + span)
            
            ax.set_xlabel("X", fontsize=9)
            ax.set_ylabel("Y", fontsize=9)
            ax.set_zlabel("Z", fontsize=9)
            
            # Camera
            if camera == "orbit":
                ax.view_init(elev=18, azim=-90 + frame_idx * 0.5)
            else:
                ax.view_init(elev=18, azim=-90)
            
            # Frame counter
            ax.text2D(0.02, 0.02, f"Frame {frame_idx}/{T-1}", transform=ax.transAxes,
                      fontsize=9, color="#444444")
            
            # Person legend
            for pi in range(len(version["positions"])):
                p_color = PERSON_COLORS[pi % len(PERSON_COLORS)]["spine"]
                ax.text2D(0.02, 0.95 - pi * 0.04,
                          f"● person{version['person_indices'][pi]}",
                          transform=ax.transAxes, fontsize=9, color=p_color)
        
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                            wspace=0.05)
    
    # Create animation
    print(f"Rendering {T} frames to {output_path} ...")
    t0 = time.time()
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    writer = FFMpegWriter(fps=fps, metadata={"title": f"Clip {clip_id}"})
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    elapsed = time.time() - t0
    print(f"Saved: {output_path} ({T} frames in {elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Headless Newton skeleton visualization → MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single version
  python physics_analysis/visualize_newton_headless.py \\
      --clip 4659 --data-dir data/retargeted_v2/gt_from_positions \\
      --label "GT" --output 4659_gt.mp4

  # GT vs Gen comparison
  python physics_analysis/visualize_newton_headless.py \\
      --clip 4659 \\
      --data-dir data/retargeted_v2/gt_from_positions \\
                 data/retargeted_v2/gen_from_positions \\
      --label "Ground Truth" "Generated" \\
      --output 4659_compare.mp4
""",
    )
    parser.add_argument("--clip", type=str, required=True, help="Clip ID")
    parser.add_argument("--data-dir", type=str, nargs="+", required=True,
                        help="One or more directories with joint_q/betas .npy files")
    parser.add_argument("--label", type=str, nargs="+", default=None,
                        help="Labels for each data-dir (default: dir basename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 path (default: auto-generated)")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS")
    parser.add_argument("--dpi", type=int, default=100, help="Render resolution")
    parser.add_argument("--camera", type=str, default="fixed",
                        choices=["fixed", "orbit"], help="Camera mode")
    parser.add_argument("--annots", action="store_true",
                        help="Show motion text annotation")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    clip_id = args.clip
    data_dirs = args.data_dir
    labels = args.label or [os.path.basename(d.rstrip("/")) for d in data_dirs]
    
    if len(labels) != len(data_dirs):
        parser.error(f"Number of labels ({len(labels)}) must match number of data-dirs ({len(data_dirs)})")
    
    # Load annotation
    annotation = None
    if args.annots:
        annotation = load_annotation(clip_id)
        if annotation:
            print(f"Annotation: {annotation}")
    
    # Load data and run FK for each version
    versions = []
    for data_dir, label in zip(data_dirs, labels):
        print(f"\n{'='*60}")
        print(f"Loading: {label} ({data_dir})")
        print(f"{'='*60}")
        
        vdata = load_version_data(clip_id, data_dir, device=args.device)
        persons = vdata["persons"]
        
        for p in persons:
            print(f"  person{p['person_idx']}: {p['n_frames']} frames, "
                  f"XML={os.path.basename(p['xml_path'])}")
        
        # Run FK
        print(f"  Running Newton FK ...")
        positions_list = fk_all_frames(persons, device=args.device)
        
        n_frames = min(p["n_frames"] for p in persons)
        person_indices = [p["person_idx"] for p in persons]
        
        versions.append({
            "label": label,
            "positions": positions_list,
            "n_frames": n_frames,
            "person_indices": person_indices,
        })
    
    # Default output path
    if args.output is None:
        out_dir = os.path.join(PROJECT_ROOT, "physics_analysis", "visualization_results")
        if len(data_dirs) > 1:
            args.output = os.path.join(out_dir, f"{clip_id}_comparison.mp4")
        else:
            dir_name = os.path.basename(data_dirs[0].rstrip("/"))
            args.output = os.path.join(out_dir, f"{clip_id}_{dir_name}.mp4")
    
    # Render
    render_comparison_mp4(
        clip_id=clip_id,
        versions=versions,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
        camera=args.camera,
        annotation=annotation,
    )


if __name__ == "__main__":
    main()
