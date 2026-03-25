"""
extract_joint_positions.py

Extract 22-joint positions from InterHuman and Inter-X datasets.
Each output: (T, 22, 3) float32 global joint positions in meters.

InterHuman: motions_processed already contains pre-computed positions in
  the first 66 values of the 492-dim vector. Direct reshape.

Inter-X: H5 file stores SMPL-X parameters (axis-angle rotations + translation),
  NOT positions. We must run SMPL-X forward kinematics to get joint positions.
  H5 shape per clip: (T, 56, 6) where:
    dim 6 = [person1(3), person2(3)]
    56 "joints": [root_orient(1), pose_body(21), jaw+eyes(3), pose_hand(30), trans(1)]
  We extract root_orient, pose_body, pose_hand, trans → run BodyModel FK → get Jtr.

Usage:
    python prepare/extract_joint_positions.py --dataset interhuman
    python prepare/extract_joint_positions.py --dataset interx
    python prepare/extract_joint_positions.py --dataset interx --smplx_model data/body_model/smplx/SMPLX_NEUTRAL.npz
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm


def extract_interhuman(data_dir, output_dir):
    """
    InterHuman motions_processed/person{1,2}/*.npy files.
    Each file shape: (T, 492) where first 22*3=66 values are joint positions.
    Extract and reshape to (T, 22, 3).
    """
    N_JOINTS = 22
    motion_dir = os.path.join(data_dir, "motions_processed")
    os.makedirs(output_dir, exist_ok=True)

    person1_dir = os.path.join(motion_dir, "person1")
    person2_dir = os.path.join(motion_dir, "person2")
    if not os.path.isdir(person1_dir):
        print(f"ERROR: expected directory {person1_dir} not found")
        return

    files = sorted([f for f in os.listdir(person1_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} InterHuman motion files")

    for fname in tqdm(files):
        clip_id = os.path.splitext(fname)[0]
        for pid, pdir in enumerate([person1_dir, person2_dir], start=0):
            fpath = os.path.join(pdir, fname)
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath).astype(np.float32)  # (T, 492)
            pos = data[:, :N_JOINTS * 3].reshape(-1, N_JOINTS, 3)  # (T, 22, 3)
            np.save(os.path.join(output_dir, f"{clip_id}_person{pid}.npy"), pos)


def extract_interx(data_dir, output_dir, smplx_model_path):
    """
    Inter-X: H5 file with shape (T, 56, 6) per clip.

    The 6-dim last axis = [person1_params(3), person2_params(3)].
    The 56 "joints" are SMPL-X parameters (axis-angle rotations), NOT positions:
        joint  0:      root_orient  (3D axis-angle)
        joints 1-21:   pose_body    (21 × 3 = 63D)
        joints 22-24:  jaw + eyes   (3 × 3 = 9D, skipped)
        joints 25-54:  pose_hand    (30 × 3 = 90D)
        joint  55:     translation  (3D)

    We run SMPL-X forward kinematics to obtain actual joint positions.
    Output: first 22 joints of Jtr = (T, 22, 3) global positions.
    """
    import h5py
    import torch

    # Add InterMask root to path for BodyModel import
    intermask_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if intermask_path not in sys.path:
        sys.path.insert(0, intermask_path)

    from data.body_model.body_model import BodyModel

    N_JOINTS = 22

    h5_path = os.path.join(data_dir, "processed", "motions", "inter-x.h5")
    if not os.path.isfile(h5_path):
        # Fallback: try without "motions" subdirectory
        h5_path = os.path.join(data_dir, "processed", "inter-x.h5")
    if not os.path.isfile(h5_path):
        print(f"ERROR: Inter-X H5 file not found. Tried:")
        print(f"  {os.path.join(data_dir, 'processed', 'motions', 'inter-x.h5')}")
        print(f"  {h5_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load SMPL-X body model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SMPL-X body model from {smplx_model_path} on {device}...")
    body_model = BodyModel(
        bm_fname=smplx_model_path,
        num_betas=10,
        num_expressions=10,
        dtype=torch.float32,
    ).to(device)

    batch_size = 64  # process frames in batches for GPU efficiency

    with h5py.File(h5_path, 'r') as mf:
        keys = sorted(mf.keys())
        print(f"Found {len(keys)} Inter-X motion clips")

        for clip_id in tqdm(keys):
            motion = mf[clip_id][:].astype(np.float32)  # (T, 56, 6)
            T = motion.shape[0]

            # Split two persons: each (T, 56, 3)
            persons_raw = [motion[:, :, :3], motion[:, :, 3:]]

            for pidx, p in enumerate(persons_raw):
                out_path = os.path.join(output_dir, f"{clip_id}_person{pidx}.npy")

                # Parse SMPL-X parameters from the 56-joint layout
                root_orient = torch.from_numpy(p[:, 0, :]).float().to(device)                  # (T, 3)
                pose_body   = torch.from_numpy(p[:, 1:22, :].reshape(T, 63)).float().to(device) # (T, 63)
                pose_hand   = torch.from_numpy(p[:, 25:55, :].reshape(T, 90)).float().to(device) # (T, 90)
                trans       = torch.from_numpy(p[:, 55, :]).float().to(device)                  # (T, 3)
                betas       = torch.zeros(1, 10, device=device)

                # Run FK in batches
                all_joints = []
                for start in range(0, T, batch_size):
                    end = min(start + batch_size, T)
                    bs = end - start
                    with torch.no_grad():
                        out = body_model(
                            root_orient=root_orient[start:end],
                            pose_body=pose_body[start:end],
                            pose_hand=pose_hand[start:end],
                            betas=betas.expand(bs, -1),
                            trans=trans[start:end],
                        )
                    # out.Jtr: (bs, n_joints, 3) — joint positions
                    # Take only first 22 joints (SMPL body joints)
                    joints = out.Jtr[:, :N_JOINTS, :].cpu().numpy()
                    all_joints.append(joints)

                positions = np.concatenate(all_joints, axis=0)  # (T, 22, 3)

                # InterX FK output is Y-up; convert to Z-up (InterHuman convention)
                # Rx(90°): (x, y, z) → (x, -z, y)
                positions = np.stack([
                    positions[:, :, 0],
                    -positions[:, :, 2],
                    positions[:, :, 1],
                ], axis=2)

                np.save(out_path, positions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 22-joint positions from InterHuman / Inter-X"
    )
    parser.add_argument("--dataset", choices=["interhuman", "interx"], required=True)
    parser.add_argument("--data_dir", default=None,
                        help="Dataset root directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for .npy files")
    parser.add_argument("--smplx_model", default="data/body_model/smplx/SMPLX_NEUTRAL.npz",
                        help="Path to SMPL-X neutral model .npz file")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = "data/InterHuman" if args.dataset == "interhuman" else "data/Inter-X_Dataset"
    if args.output_dir is None:
        args.output_dir = f"data/extracted_positions/{args.dataset}"

    if args.dataset == "interhuman":
        extract_interhuman(args.data_dir, args.output_dir)
    else:
        extract_interx(args.data_dir, args.output_dir, args.smplx_model)