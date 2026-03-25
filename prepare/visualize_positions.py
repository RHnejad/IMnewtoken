"""
Visualize extracted joint positions (T, 22, 3) as a 3D skeleton animation.

Usage:
    python prepare/visualize_positions.py --clip 1000          # both persons
    python prepare/visualize_positions.py --clip 1000 --person 0  # person 0 only
    python prepare/visualize_positions.py --save                # save mp4 instead of showing
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
from pathlib import Path

# ── SMPL 22-joint skeleton ──────────────────────────────────────────────
# Joint names (index → name)
JOINT_NAMES = [
    "Pelvis",        # 0
    "L_Hip",         # 1
    "R_Hip",         # 2
    "Spine1",        # 3
    "L_Knee",        # 4
    "R_Knee",        # 5
    "Spine2",        # 6
    "L_Ankle",       # 7
    "R_Ankle",       # 8
    "Spine3",        # 9
    "L_Foot",        # 10
    "R_Foot",        # 11
    "Neck",          # 12
    "L_Collar",      # 13
    "R_Collar",      # 14
    "Head",          # 15
    "L_Shoulder",    # 16
    "R_Shoulder",    # 17
    "L_Elbow",       # 18
    "R_Elbow",       # 19
    "L_Wrist",       # 20
    "R_Wrist",       # 21
]

# Bone connections (parent → child)
BONES = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
]

# Colors for body parts
BONE_COLORS = {
    "spine": "#4444FF",
    "left_leg": "#FF4444",
    "right_leg": "#44BB44",
    "left_arm": "#FF8800",
    "right_arm": "#AA44AA",
}

def get_bone_color(i, j):
    if (i, j) in [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]:
        return BONE_COLORS["spine"]
    elif (i, j) in [(0, 1), (1, 4), (4, 7), (7, 10)]:
        return BONE_COLORS["left_leg"]
    elif (i, j) in [(0, 2), (2, 5), (5, 8), (8, 11)]:
        return BONE_COLORS["right_leg"]
    elif (i, j) in [(9, 13), (13, 16), (16, 18), (18, 20)]:
        return BONE_COLORS["left_arm"]
    elif (i, j) in [(9, 14), (14, 17), (17, 19), (19, 21)]:
        return BONE_COLORS["right_arm"]
    return "#888888"

PERSON_ALPHA = [1.0, 0.7]  # person0 solid, person1 slightly transparent


def load_clip(data_dir: Path, clip_id: str, person: int | None = None):
    """Load one or both persons for a clip. Returns list of (T,22,3) arrays."""
    persons = []
    labels = []
    if person is not None:
        f = data_dir / f"{clip_id}_person{person}.npy"
        if not f.exists():
            raise FileNotFoundError(f)
        persons.append(np.load(f))
        labels.append(f"person{person}")
    else:
        for p in [0, 1]:
            f = data_dir / f"{clip_id}_person{p}.npy"
            if f.exists():
                persons.append(np.load(f))
                labels.append(f"person{p}")
    if not persons:
        raise FileNotFoundError(f"No files found for clip {clip_id}")
    return persons, labels


def visualize(persons, labels, clip_id, fps=20, save_path=None):
    """Create 3D skeleton animation."""
    if save_path:
        matplotlib.use("Agg")

    T = min(p.shape[0] for p in persons)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"Clip {clip_id} — {T} frames @ {fps} fps", fontsize=14)

    # Compute global bounds across all persons and all frames
    all_pos = np.concatenate(persons, axis=0)
    center = all_pos.mean(axis=(0, 1))  # (3,)
    extent = max(
        all_pos[:, :, 0].max() - all_pos[:, :, 0].min(),
        all_pos[:, :, 1].max() - all_pos[:, :, 1].min(),
        all_pos[:, :, 2].max() - all_pos[:, :, 2].min(),
    ) * 0.6

    # Pre-create line and scatter objects for each person
    bone_lines = []
    joint_scatters = []
    for pi, (pos, label) in enumerate(zip(persons, labels)):
        alpha = PERSON_ALPHA[pi] if pi < len(PERSON_ALPHA) else 0.5
        lines = []
        for bone in BONES:
            color = get_bone_color(*bone)
            (line,) = ax.plot([], [], [], color=color, linewidth=2, alpha=alpha)
            lines.append(line)
        bone_lines.append(lines)

        scatter = ax.scatter([], [], [], s=20, alpha=alpha, label=label)
        joint_scatters.append(scatter)

    ax.legend(loc="upper left")
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)

    def init():
        for lines in bone_lines:
            for line in lines:
                line.set_data_3d([], [], [])
        return []

    def update(frame):
        for pi, pos in enumerate(persons):
            t = min(frame, pos.shape[0] - 1)
            p = pos[t]  # (22, 3)

            # Update bones
            for bi, (i, j) in enumerate(BONES):
                xs = [p[i, 0], p[j, 0]]
                ys = [p[i, 1], p[j, 1]]
                zs = [p[i, 2], p[j, 2]]
                bone_lines[pi][bi].set_data_3d(xs, ys, zs)

            # Update joints
            joint_scatters[pi]._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        frame_text.set_text(f"Frame {frame}/{T-1}")
        return []

    anim = FuncAnimation(
        fig, update, init_func=init, frames=T,
        interval=1000 // fps, blit=False
    )

    if save_path:
        print(f"Saving animation to {save_path} ...")
        anim.save(str(save_path), writer="ffmpeg", fps=fps, dpi=100)
        print(f"Done! Saved {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize extracted joint positions")
    parser.add_argument("--clip", type=str, default="1000", help="Clip ID (e.g. 1000)")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Person index (0 or 1). Omit for both.")
    parser.add_argument("--data-dir", type=str,
                        default="data/extracted_positions/interhuman",
                        help="Directory with .npy files")
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS")
    parser.add_argument("--save", action="store_true",
                        help="Save as MP4 instead of interactive display")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for MP4 (default: vis_<clip>.mp4)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    persons, labels = load_clip(data_dir, args.clip, args.person)

    for p, label in zip(persons, labels):
        print(f"{label}: shape={p.shape}, range=[{p.min():.3f}, {p.max():.3f}]")

    save_path = None
    if args.save:
        save_path = args.out or f"vis_{args.clip}.mp4"

    visualize(persons, labels, args.clip, fps=args.fps, save_path=save_path)


if __name__ == "__main__":
    main()
