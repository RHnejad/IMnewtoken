"""
gen_comparison_mp4.py — Generate side-by-side GT vs Generated MP4 comparison videos.

Uses matplotlib (Agg backend) so it works headlessly over SSH — no OpenGL needed.
Renders GT skeletons on the left, Generated on the right, with shared camera.

Usage:
    conda run -n mimickit --no-capture-output python prepare4/gen_comparison_mp4.py \
        --clips 1129 1147 1187 121

    # Custom output directory:
    conda run -n mimickit --no-capture-output python prepare4/gen_comparison_mp4.py \
        --clips 1129 --output-dir prepare4/newton_videos/
"""
import os
import sys
import pickle
import argparse
import numpy as np
import scipy.ndimage.filters as filters

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

from utils.paramUtil import t2m_kinematic_chain

# Person colors: person1=blue tones, person2=red tones
PERSON_COLORS = [
    ['#1f77b4', '#1f77b4', '#2ca02c', '#1f77b4', '#1f77b4'],  # person 1
    ['#d62728', '#d62728', '#ff7f0e', '#d62728', '#d62728'],   # person 2
]


def load_gen_positions_yup(clip_id):
    """Load generated positions and return in Y-up frame."""
    path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman",
                        f"{clip_id}.pkl")
    if not os.path.isfile(path):
        return None, None

    with open(path, "rb") as f:
        raw = pickle.load(f)

    TRANS_MATRIX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

    joints = []
    for pkey in ["person1", "person2"]:
        if pkey not in raw:
            return None, None
        p = raw[pkey]
        if "positions_zup" not in p:
            return None, None
        pos_zup = p["positions_zup"]
        pos_yup = np.einsum("mn,...n->...m", TRANS_MATRIX, pos_zup)
        joints.append(pos_yup.astype(np.float64))

    text = raw.get("text", f"Clip {clip_id}")
    return joints, text


def load_gt_positions_yup(clip_id):
    """Load GT positions from raw NPY and convert to Y-up frame."""
    from data.utils import trans_matrix as TRANS_MATRIX_TORCH
    TRANS = TRANS_MATRIX_TORCH.numpy().astype(np.float64)

    gt_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    joints = []
    for pidx in [1, 2]:
        npy_path = os.path.join(gt_dir, "motions_processed",
                                "person" + str(pidx), f"{clip_id}.npy")
        if not os.path.isfile(npy_path):
            return None
        raw = np.load(npy_path).astype(np.float64)
        pos_zup = raw[:, :66].reshape(-1, 22, 3)
        pos_yup = np.einsum("mn,...n->...m", TRANS, pos_zup)
        joints.append(pos_yup)

    # Floor correction
    all_y = np.concatenate([j[:, :, 1] for j in joints])
    floor = all_y.min()
    for j in joints:
        j[:, :, 1] -= floor

    return joints


def render_comparison_mp4(save_path, gt_joints, gen_joints, text,
                          fps=30, radius=3.0):
    """
    Render GT (left) and Generated (right) skeletons side-by-side.
    Each panel shows both persons with different colors.
    """
    T_gt = min(j.shape[0] for j in gt_joints)
    T_gen = min(j.shape[0] for j in gen_joints)
    T = min(T_gt, T_gen)

    gt_joints = [j[:T].copy() for j in gt_joints]
    gen_joints = [j[:T].copy() for j in gen_joints]

    # Floor correction for gen
    all_y_gen = np.concatenate([j[:, :, 1] for j in gen_joints])
    floor_gen = all_y_gen.min()
    for j in gen_joints:
        j[:, :, 1] -= floor_gen

    # Compute shared bounds across both GT and Gen for consistent view
    all_pos = np.concatenate(gt_joints + gen_joints, axis=0)
    center_x = (all_pos[:, :, 0].max() + all_pos[:, :, 0].min()) / 2
    center_z = (all_pos[:, :, 2].max() + all_pos[:, :, 2].min()) / 2
    y_max = all_pos[:, :, 1].max()

    fig = plt.figure(figsize=(16, 7))

    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_gen = fig.add_subplot(1, 2, 2, projection='3d')

    title_sp = text.split(' ')
    if len(title_sp) > 15:
        title_text = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    else:
        title_text = text

    def setup_ax(ax, label):
        ax.cla()
        ax.set_xlim3d([center_x - radius, center_x + radius])
        ax.set_ylim3d([0, radius * 2])
        ax.set_zlim3d([center_z - radius, center_z + radius])
        ax.view_init(elev=20, azim=-60)
        ax.set_title(label, fontsize=14, fontweight='bold', pad=0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Ground plane
        gnd = radius * 1.5
        verts = [[center_x - gnd, 0, center_z - gnd],
                 [center_x - gnd, 0, center_z + gnd],
                 [center_x + gnd, 0, center_z + gnd],
                 [center_x + gnd, 0, center_z - gnd]]
        plane = Poly3DCollection([verts])
        plane.set_facecolor((0.85, 0.85, 0.85, 0.5))
        plane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
        ax.add_collection3d(plane)

    def draw_skeleton(ax, joints_frame, person_idx):
        colors = PERSON_COLORS[person_idx]
        for ci, chain in enumerate(t2m_kinematic_chain):
            lw = 3.0 if ci < 3 else 2.0
            color = colors[ci % len(colors)]
            ax.plot3D(joints_frame[chain, 0],
                      joints_frame[chain, 1],
                      joints_frame[chain, 2],
                      linewidth=lw, color=color, alpha=0.9)
        # Draw joint dots
        ax.scatter(joints_frame[:, 0], joints_frame[:, 1], joints_frame[:, 2],
                   s=8, c=PERSON_COLORS[person_idx][0], alpha=0.6)

    def update(frame_idx):
        setup_ax(ax_gt, f'Ground Truth  (frame {frame_idx}/{T})')
        setup_ax(ax_gen, f'Generated  (frame {frame_idx}/{T})')

        for pid in range(2):
            draw_skeleton(ax_gt, gt_joints[pid][frame_idx], pid)
            draw_skeleton(ax_gen, gen_joints[pid][frame_idx], pid)

        fig.suptitle(title_text, fontsize=12, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps, dpi=100,
             savefig_kwargs={'facecolor': 'white'})
    plt.close()
    print(f"  Saved: {save_path}  ({T} frames, {T/fps:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GT vs Generated side-by-side comparison MP4s")
    parser.add_argument("--clips", type=str, nargs="+", required=True,
                        help="Clip IDs to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: prepare4/newton_videos/)")
    parser.add_argument("--smooth", action="store_true", default=True,
                        help="Apply light Gaussian smoothing")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, "prepare4", "newton_videos")
    os.makedirs(out_dir, exist_ok=True)

    for clip_id in args.clips:
        print(f"\n{'='*60}")
        print(f" Clip {clip_id}")
        print(f"{'='*60}")

        gen_joints, text = load_gen_positions_yup(clip_id)
        if gen_joints is None:
            print(f"  No generated data for clip {clip_id}, skipping")
            continue

        gt_joints = load_gt_positions_yup(clip_id)
        if gt_joints is None:
            print(f"  No GT data for clip {clip_id}, skipping")
            continue

        if args.smooth:
            gen_joints = [filters.gaussian_filter1d(j, 1, axis=0, mode='nearest')
                          for j in gen_joints]
            gt_joints = [filters.gaussian_filter1d(j, 1, axis=0, mode='nearest')
                         for j in gt_joints]

        save_path = os.path.join(out_dir, f"{clip_id}_comparison.mp4")
        print(f"  Text: '{text}'")
        render_comparison_mp4(save_path, gt_joints, gen_joints, text,
                              fps=args.fps)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
