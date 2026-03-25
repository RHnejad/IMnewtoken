"""
retarget_rotation.py — Direct rotation transfer from SMPL-X to Newton.

Instead of IK (position → joint angles), this script DIRECTLY converts
SMPL-X axis-angle rotations to Newton joint angles (Euler XYZ).

Advantages over IK:
  - Exact rotation transfer (no approximation error)
  - No iterative solving (fast, deterministic)
  - No temporal jitter from IK convergence

Caveat:
  - Bone proportions differ between SMPL-X and Newton's model,
    so absolute end-effector positions won't match exactly,
    but the motion quality (rotation fidelity) is perfect.

Coordinate Systems:
  - SMPL-X: X-left, Y-up, Z-forward
  - Newton:  X-forward, Y-left, Z-up
  - R_ROT  (rotations):  [[0,0,1],[1,0,0],[0,1,0]]    (body-local frame mapping)
  - Position: direct mapping (InterHuman data is Z-up, Newton up_axis=Z)

Newton joint_q layout (76 values):
  [0:7]   = Pelvis freejoint: tx, ty, tz, qx, qy, qz, qw  (xyzw)
  [7:10]  = L_Hip:   extrinsic Euler XYZ (3 hinges)
  [10:13] = L_Knee:  Euler XYZ
  [13:16] = L_Ankle: Euler XYZ
  [16:19] = L_Toe:   Euler XYZ
  [19:22] = R_Hip:   Euler XYZ
  ... (depth-first order, 3 hinges per body)

Usage:
    # Single clip (InterHuman)
    python prepare/retarget_rotation.py --dataset interhuman --clip 1000

    # Batch (InterHuman)
    python prepare/retarget_rotation.py --dataset interhuman

    # Single clip (Inter-X)
    python prepare/retarget_rotation.py --dataset interx --clip S001C001A001

    # Batch (Inter-X)
    python prepare/retarget_rotation.py --dataset interx
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# SMPL-X joint → Newton body index
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}
N_SMPL_JOINTS = 22
N_NEWTON_BODIES = 24
N_JOINT_Q = 76  # 7 (freejoint) + 23 * 3 (hinges)

# Coordinate transforms (SMPL-X ↔ Newton):
#   R_ROT:   maps SMPL-X body-local → Newton body-local frame
#   Position: direct mapping (InterHuman trans is Z-up, Newton up_axis=Z)

R_ROT = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def newton_body_q_index(body_idx):
    """Get the starting index in joint_q for a non-root Newton body."""
    assert body_idx >= 1, "Root (body 0) uses freejoint, not hinges"
    return 7 + (body_idx - 1) * 3


# SMPL-X body offset: rest-pose pelvis position (v_template + J_regressor).
# Must be added to trans to get the actual pelvis world position.
# Computed once and cached.
_SMPLX_BODY_OFFSET = None


def _get_smplx_body_offset(smplx_model_path="data/body_model/smplx/SMPLX_NEUTRAL.npz"):
    """Get SMPL-X rest-pose pelvis offset (computed once, cached)."""
    global _SMPLX_BODY_OFFSET
    if _SMPLX_BODY_OFFSET is None:
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data', 'body_model'))
        from body_model import BodyModel
        bm = BodyModel(smplx_model_path)
        with torch.no_grad():
            rest = bm()
        _SMPLX_BODY_OFFSET = rest.Jtr[0, 0].numpy().astype(np.float64)
    return _SMPLX_BODY_OFFSET


def smplx_to_joint_q(root_orient, pose_body, trans, smplx_model_path=None):
    """
    Convert SMPL-X parameters to Newton joint_q.

    InterHuman data is in Z-up world frame (root_orient encodes Y→Z).
    With Newton up_axis=Z, positions map directly. R_ROT is used
    for body-joint rotation conjugation.

    Args:
        root_orient: (T, 3) axis-angle for root joint (in Z-up world)
        pose_body: (T, 63) axis-angle for 21 body joints
        trans: (T, 3) global translation (in Z-up world)

    Returns:
        joint_q: (T, 76) Newton joint coordinates
    """
    T = root_orient.shape[0]
    joint_q = np.zeros((T, N_JOINT_Q), dtype=np.float32)

    # Get SMPL-X body offset (rest-pose pelvis position)
    body_offset = _get_smplx_body_offset(
        smplx_model_path or "data/body_model/smplx/SMPLX_NEUTRAL.npz"
    )

    # ── Root (Pelvis) ──────────────────────────────────────────
    # Position: direct sum (both trans and body_offset give pelvis world pos)
    trans_with_offset = trans + body_offset[None, :]  # (T, 3)
    joint_q[:, 0:3] = trans_with_offset

    # Orientation: R_newton_root = R(root_orient) @ R_ROT^T
    R_smplx_root = Rotation.from_rotvec(root_orient)
    R_ROT_inv = Rotation.from_matrix(R_ROT.T)
    root_newton = R_smplx_root * R_ROT_inv
    root_quat_xyzw = root_newton.as_quat()  # scipy: [x, y, z, w]
    # Newton joint_q free joint: [px,py,pz, qx,qy,qz,qw] (xyzw)
    joint_q[:, 3:7] = root_quat_xyzw

    # ── Body joints (1-21) ─────────────────────────────────────
    pose_body_reshaped = pose_body.reshape(T, 21, 3)

    for smpl_j in range(1, N_SMPL_JOINTS):
        newton_body = SMPL_TO_NEWTON[smpl_j]
        q_start = newton_body_q_index(newton_body)

        # Get axis-angle for this joint
        aa_smplx = pose_body_reshaped[:, smpl_j - 1, :]  # (T, 3)

        # Transform to Newton rotation frame via R_ROT
        aa_newton = (R_ROT @ aa_smplx.T).T  # (T, 3)

        # Convert to rotation matrix, then decompose to extrinsic XYZ Euler
        rot = Rotation.from_rotvec(aa_newton)
        euler_xyz = rot.as_euler('XYZ')  # extrinsic XYZ, in radians

        joint_q[:, q_start:q_start + 3] = euler_xyz

    # Newton bodies L_Hand(18) and R_Hand(23) have no SMPL-X counterpart → remain 0

    return joint_q


def extract_positions_from_fk(model, joint_q, device="cuda:0"):
    """
    Run Newton FK on joint_q to get body positions.

    Args:
        model: Newton model
        joint_q: (T, 76) joint coordinates
        device: CUDA device

    Returns:
        positions: (T, 22, 3) joint positions
    """
    T = joint_q.shape[0]
    positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    for t in range(T):
        state.joint_q = wp.array(joint_q[t], dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)  # (24, 7)
        for j in range(N_SMPL_JOINTS):
            positions[t, j] = body_q[SMPL_TO_NEWTON[j], :3]

    return positions


def build_newton_model(device="cuda:0"):
    """Build Newton model for FK."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf("prepare/assets/smpl.xml", enable_self_collisions=False)
    return builder.finalize(device=device)


# ═══════════════════════════════════════════════════════════════
# InterHuman dataset loading
# ═══════════════════════════════════════════════════════════════

def load_interhuman_clip(data_dir, clip_id):
    """
    Load SMPL-X params from InterHuman raw pkl.

    Returns list of dicts (one per person), each with:
        root_orient: (T, 3)
        pose_body: (T, 63)
        trans: (T, 3)
    """
    pkl_path = os.path.join(data_dir, "motions", f"{clip_id}.pkl")
    if not os.path.isfile(pkl_path):
        return None

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)

    results = []
    for person_key in ['person1', 'person2']:
        if person_key not in raw:
            continue
        p = raw[person_key]
        results.append({
            'root_orient': p['root_orient'].astype(np.float64),
            'pose_body': p['pose_body'].astype(np.float64),
            'trans': p['trans'].astype(np.float64),
        })

    return results


def list_interhuman_clips(data_dir):
    """List all available InterHuman clip IDs."""
    motion_dir = os.path.join(data_dir, "motions")
    clips = []
    for f in sorted(os.listdir(motion_dir)):
        if f.endswith('.pkl'):
            clips.append(os.path.splitext(f)[0])
    return clips


# ═══════════════════════════════════════════════════════════════
# Inter-X dataset loading
# ═══════════════════════════════════════════════════════════════

def load_interx_clip(data_dir, clip_id):
    """
    Load SMPL-X params from Inter-X H5 file.

    H5 format per clip: (T, 56, 6) where dim6 = [person1(3), person2(3)]
    Layout: [root_orient(1), pose_body(21), jaw+eyes(3), pose_hand(30), trans(1)]

    InterX data is in Y-up world frame. We convert to Z-up (InterHuman
    convention) so that the Z-up retarget formulas apply correctly.

    Returns list of dicts (one per person).
    """
    import h5py

    h5_path = os.path.join(data_dir, "processed", "motions", "inter-x.h5")
    if not os.path.isfile(h5_path):
        h5_path = os.path.join(data_dir, "processed", "inter-x.h5")
    if not os.path.isfile(h5_path):
        return None

    with h5py.File(h5_path, 'r') as mf:
        if clip_id not in mf:
            return None
        motion = mf[clip_id][:].astype(np.float64)  # (T, 56, 6)

    T = motion.shape[0]
    results = []

    # Y-up → Z-up: Rx(90°) = [[1,0,0],[0,0,-1],[0,1,0]]
    R_yup_to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    R_conv = Rotation.from_matrix(R_yup_to_zup)

    for pidx in range(2):
        p = motion[:, :, pidx * 3:(pidx + 1) * 3]  # (T, 56, 3)

        root_orient = p[:, 0, :]                             # (T, 3)
        pose_body = p[:, 1:22, :].reshape(T, 63)             # (T, 63)
        trans = p[:, 55, :]                                   # (T, 3)

        # Convert world frame Y-up → Z-up
        trans = (R_yup_to_zup @ trans.T).T
        root_orient = (R_conv * Rotation.from_rotvec(root_orient)).as_rotvec()

        results.append({
            'root_orient': root_orient,
            'pose_body': pose_body,
            'trans': trans,
        })

    return results


def list_interx_clips(data_dir):
    """List all available Inter-X clip IDs."""
    import h5py
    h5_path = os.path.join(data_dir, "processed", "motions", "inter-x.h5")
    if not os.path.isfile(h5_path):
        h5_path = os.path.join(data_dir, "processed", "inter-x.h5")
    if not os.path.isfile(h5_path):
        return []
    with h5py.File(h5_path, 'r') as mf:
        return sorted(mf.keys())


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def process_clip(model, persons_data, device="cuda:0"):
    """
    Process one clip: convert rotations → joint_q → FK positions.

    Args:
        model: Newton model
        persons_data: list of dicts with root_orient, pose_body, trans

    Returns:
        list of dicts with 'joint_q' (T, 76) and 'positions' (T, 22, 3)
    """
    results = []
    for p in persons_data:
        joint_q = smplx_to_joint_q(p['root_orient'], p['pose_body'], p['trans'])
        positions = extract_positions_from_fk(model, joint_q, device=device)
        results.append({
            'joint_q': joint_q,
            'positions': positions,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Direct rotation transfer: SMPL-X → Newton")
    parser.add_argument("--dataset", choices=["interhuman", "interx"], required=True)
    parser.add_argument("--data_dir", default=None, help="Dataset root directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--clip", default=None, help="Single clip ID (e.g., 1000 or S001C001A001)")
    parser.add_argument("--gpu", default="cuda:0", help="GPU device")
    parser.add_argument("--save_joint_q", action="store_true",
                        help="Also save joint_q arrays (for visualization)")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = "data/InterHuman" if args.dataset == "interhuman" else "data/Inter-X_Dataset"
    if args.output_dir is None:
        args.output_dir = f"data/retargeted_rotation/{args.dataset}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Build Newton model
    print(f"Building Newton model on {args.gpu}...")
    model = build_newton_model(device=args.gpu)

    # Load / list clips
    if args.dataset == "interhuman":
        load_fn = lambda cid: load_interhuman_clip(args.data_dir, cid)
        if args.clip:
            clips = [args.clip]
        else:
            clips = list_interhuman_clips(args.data_dir)
    else:
        load_fn = lambda cid: load_interx_clip(args.data_dir, cid)
        if args.clip:
            clips = [args.clip]
        else:
            clips = list_interx_clips(args.data_dir)

    print(f"Processing {len(clips)} clips...")
    total_time = 0
    processed = 0
    skipped = 0

    for clip_id in tqdm(clips, desc="Retarget (rotation)"):
        persons_data = load_fn(clip_id)
        if persons_data is None:
            skipped += 1
            continue

        t0 = time.time()
        results = process_clip(model, persons_data, device=args.gpu)
        dt = time.time() - t0
        total_time += dt

        for pidx, res in enumerate(results):
            out_name = f"{clip_id}_person{pidx}"
            np.save(os.path.join(args.output_dir, f"{out_name}.npy"), res['positions'])

            if args.save_joint_q:
                jq_dir = os.path.join(args.output_dir, "joint_q")
                os.makedirs(jq_dir, exist_ok=True)
                np.save(os.path.join(jq_dir, f"{out_name}.npy"), res['joint_q'])

        processed += 1

        if processed == 1 or processed % 100 == 0:
            T = results[0]['positions'].shape[0]
            print(f"  Clip {clip_id}: {T} frames, {dt:.3f}s")

    print(f"\nDone: {processed} clips processed, {skipped} skipped")
    if processed > 0:
        print(f"Total time: {total_time:.1f}s, avg: {total_time/processed:.3f}s/clip")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
