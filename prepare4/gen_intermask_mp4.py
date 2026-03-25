"""
gen_intermask_mp4.py — Generate InterMask-style MP4 visualizations from generated PKLs.

This uses the EXACT same visualization code as InterMask (plot_3d_motion_2views)
so you can compare with the MP4s in data/reconstructed_dataset/interhuman/visualizations/.

Usage:
    # Generate MP4 for specific clips
    conda run -n mimickit --no-capture-output python prepare4/gen_intermask_mp4.py --clips 3678 4659

    # Generate for clip 1 with GT comparison
    conda run -n mimickit --no-capture-output python prepare4/gen_intermask_mp4.py --clips 1 --with-gt
"""
import os
import sys
import pickle
import argparse
import numpy as np
import scipy.ndimage.filters as filters

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.paramUtil import t2m_kinematic_chain


def plot_3d_motion_2views_compat(save_path, kinematic_tree, mp_joints,
                                  title, figsize=(20, 10), fps=30, radius=8):
    """
    InterMask-style 3D skeleton visualization (compatible with newer matplotlib).
    Renders two views: front + side (rotated 110deg around Y).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.animation import FuncAnimation
    from scipy.spatial.transform import Rotation as R

    colors = ['orange', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]),
                           ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    rot = R.from_euler('y', 110, degrees=True)
    frame_number = min(j.shape[0] for j in mp_joints)

    mp_data = []
    for i, joints in enumerate(mp_joints):
        data = joints[:frame_number].copy().reshape(frame_number, -1, 3)
        MINS = data.min(axis=0).min(axis=0)
        data[:, :, 1] -= MINS[1]  # put on floor
        data_rot = rot.apply(data.reshape(-1, 3)).reshape(-1, 22, 3)
        mp_data.append({"joints": data, "joints_rot": data_rot})

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax1, ax2]

    mp_colors = [[colors[i]] * 15 for i in range(len(mp_joints))]

    def update(index):
        for ax in axs:
            ax.cla()
            ax.set_xlim3d([-radius / 4, radius / 4])
            ax.set_ylim3d([0, radius / 4])
            ax.set_zlim3d([0, radius / 4])
            ax.view_init(elev=120, azim=270)
            ax.axis('off')

            # Ground plane
            verts = [[-3, 0, -3], [-3, 0, 3], [3, 0, 3], [3, 0, -3]]
            plane = Poly3DCollection([verts])
            plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(plane)

        fig.suptitle(title, fontsize=20)

        for pid, data in enumerate(mp_data):
            for ci, (chain, color) in enumerate(zip(kinematic_tree,
                                                     mp_colors[pid])):
                lw = 2.0 if ci < 5 else 1.0
                j = data["joints"][index]
                axs[0].plot3D(j[chain, 0], j[chain, 1], j[chain, 2],
                              linewidth=lw, color=color)
                jr = data["joints_rot"][index]
                axs[1].plot3D(jr[chain, 0], jr[chain, 1], jr[chain, 2],
                              linewidth=lw, color=color)

    ani = FuncAnimation(fig, update, frames=frame_number,
                        interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
    print(f"    Saved: {save_path}")


def load_gen_positions_yup(clip_id):
    """
    Load generated positions and return in Y-up processed frame
    (same frame as InterMask's plot_3d_motion_2views expects).
    """
    path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman",
                        f"{clip_id}.pkl")
    if not os.path.isfile(path):
        return None, None

    with open(path, "rb") as f:
        raw = pickle.load(f)

    # The positions_zup are in Z-up world frame after INV_TRANS from Y-up proc
    # Convert back to Y-up processed: (x, y, z)_zup → (x, z, -y)_yup
    TRANS_MATRIX = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

    joints = []
    for pkey in ["person1", "person2"]:
        if pkey not in raw:
            return None, None
        p = raw[pkey]
        if "positions_zup" in p:
            pos_zup = p["positions_zup"]
            pos_yup = np.einsum("mn,...n->...m", TRANS_MATRIX, pos_zup)
        else:
            # Fall back: decode from the 262-dim representation stored
            # or just skip
            print(f"  WARNING: {pkey} has no positions_zup")
            return None, None
        joints.append(pos_yup.astype(np.float64))

    text = raw.get("text", f"Clip {clip_id}")
    return joints, text


def load_gt_positions_yup(clip_id):
    """
    Load GT positions from raw NPY and convert to Y-up frame.
    Uses raw positions (not process_motion_np) to preserve relative
    world positions between the two persons.
    Only applies trans_matrix (Z-up → Y-up) and floor correction.
    """
    from data.utils import trans_matrix as TRANS_MATRIX_TORCH
    TRANS = TRANS_MATRIX_TORCH.numpy().astype(np.float64)  # (x,y,z)_zup → (x,z,-y)_yup

    gt_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    joints = []
    for pidx in [1, 2]:
        npy_path = os.path.join(gt_dir, "motions_processed",
                                f"person{pidx}", f"{clip_id}.npy")
        if not os.path.isfile(npy_path):
            return None

        raw = np.load(npy_path).astype(np.float64)
        # First 66 cols are positions in Z-up world frame
        pos_zup = raw[:, :66].reshape(-1, 22, 3)
        pos_yup = np.einsum("mn,...n->...m", TRANS, pos_zup)
        joints.append(pos_yup)

    # Floor correction: shift Y so min foot height = 0
    all_y = np.concatenate([j[:, :, 1] for j in joints])
    floor = all_y.min()
    for j in joints:
        j[:, :, 1] -= floor

    return joints


def main():
    parser = argparse.ArgumentParser(
        description="Generate InterMask-style MP4 visualizations")
    parser.add_argument("--clips", type=str, nargs="+", required=True,
                        help="Clip IDs to visualize")
    parser.add_argument("--with-gt", action="store_true",
                        help="Also generate GT comparison MP4s")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: prepare4/vis_compare/)")
    parser.add_argument("--smooth", action="store_true", default=True,
                        help="Apply Gaussian smoothing (same as InterMask default)")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, "prepare4", "vis_compare")
    os.makedirs(out_dir, exist_ok=True)

    for clip_id in args.clips:
        print(f"\n{'='*60}")
        print(f" Clip {clip_id}")
        print(f"{'='*60}")

        # ── Generated MP4 ──
        gen_joints, text = load_gen_positions_yup(clip_id)
        if gen_joints is None:
            print(f"  No generated data for clip {clip_id}")
            continue

        T = min(j.shape[0] for j in gen_joints)
        gen_joints = [j[:T] for j in gen_joints]

        if args.smooth:
            gen_joints = [filters.gaussian_filter1d(j, 1, axis=0, mode='nearest')
                          for j in gen_joints]

        gen_path = os.path.join(out_dir, f"{clip_id}_gen.mp4")
        print(f"  Gen: {T} frames, text: '{text}'")
        print(f"  Saving: {gen_path}")
        plot_3d_motion_2views_compat(
            gen_path, t2m_kinematic_chain,
            gen_joints,
            title=f"Gen: {text}", fps=30,
        )

        # ── GT MP4 ──
        if args.with_gt:
            gt_joints = load_gt_positions_yup(clip_id)
            if gt_joints is None:
                print(f"  No GT data for clip {clip_id}")
                continue

            T_gt = min(j.shape[0] for j in gt_joints)
            gt_joints = [j[:T_gt] for j in gt_joints]

            if args.smooth:
                gt_joints = [filters.gaussian_filter1d(j, 1, axis=0, mode='nearest')
                             for j in gt_joints]

            gt_path = os.path.join(out_dir, f"{clip_id}_gt.mp4")
            print(f"  GT: {T_gt} frames")
            print(f"  Saving: {gt_path}")
            plot_3d_motion_2views_compat(
                gt_path, t2m_kinematic_chain,
                gt_joints,
                title=f"GT: {text}", fps=30,
            )

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
