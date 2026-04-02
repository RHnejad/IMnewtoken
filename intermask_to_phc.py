#!/usr/bin/env python3
"""
intermask_to_phc.py — Convert InterMask .npy output to PHC SMPL-X .pkl format.

Usage (from repo root /home/tomvan/IMnewtoken):
  PYTHONPATH=PHC .venv/phc/bin/python intermask_to_phc.py \
      --npy  checkpoints/interx/<name>/animation_infer/<file>.npy \
      --out  output/my_motion_phc.pkl

Then run PHC evaluation (from PHC/ subdirectory):
  python phc/run_hydra.py \
      learning=im_pnn_big exp_name=phc_x_pnn env=env_im_x_pnn \
      robot=smplx_humanoid env.motion_file=../output/my_motion_phc.pkl \
      env.training_prim=0 epoch=-1 test=True env.num_envs=1 headless=False

InterMask .npy layout (after reshape to (T, 2, 56, 6)):
  axis 2, slots 0-54  : SMPL-X joint 6D rotations in SMPLX_BONE_ORDER_NAMES order
  axis 2, slot 55     : root translation encoded as pseudo-joint (xyz = first 3 values)

PHC SMPL-X uses SMPLH_BONE_ORDER_NAMES (52 joints = SMPL-X body without face joints
Jaw/L_Eye/R_Eye) reindexed to SMPLH_MUJOCO_NAMES order for the Mujoco articulation tree.
Coordinate system: Z-up (Isaac Gym), no upright_start rotation for SMPL-X.
"""

import sys
import os
import argparse
import numpy as np
import torch
import joblib
from scipy.spatial.transform import Rotation as sRot

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_PHC  = os.path.join(_ROOT, "PHC")
sys.path.insert(0, _ROOT)   # for data.rotation_conversions
sys.path.insert(0, _PHC)    # for poselib.*  (PHC/poselib/poselib/…)

from data.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_joint_names import (
    SMPLX_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# ── Joint-index constants ─────────────────────────────────────────────────────

# Indices within SMPLX_BONE_ORDER_NAMES[0:55] that PHC skips (face joints)
_SMPLX_FACE_IDX = {22, 23, 24}   # Jaw, L_Eye_Smplhf, R_Eye_Smplhf

# After removing face joints, InterMask joints map to SMPLH_BONE_ORDER_NAMES.
# Verify this at import time (mirrors the comment in smpl_parser.py):
assert (SMPLX_BONE_ORDER_NAMES[:22] + SMPLX_BONE_ORDER_NAMES[25:55]) == list(SMPLH_BONE_ORDER_NAMES), (
    "SMPLX→SMPLH joint mapping mismatch — check smpl_joint_names version"
)

# Index permutation: SMPLH_BONE_ORDER → SMPLH_MUJOCO (PHC's internal DoF ordering)
_SMPLH_TO_MUJOCO = [SMPLH_BONE_ORDER_NAMES.index(j) for j in SMPLH_MUJOCO_NAMES]

# Slots from InterMask's 55-joint block to keep (remove face idx 22,23,24)
_KEEP_SMPLX = [i for i in range(55) if i not in _SMPLX_FACE_IDX]

# ── Skeleton builder ──────────────────────────────────────────────────────────

def build_skeleton_tree(smpl_data_dir: str, gender: int = 0) -> SkeletonTree:
    """
    Build an SMPL-X SkeletonTree for PHC by:
      1. Generating a zero-beta neutral SMPL-X Mujoco XML via LocalRobot
      2. Parsing the XML into a SkeletonTree (poselib)

    Uses upright_start=False — the standard PHC SMPL-X configuration (smplx_humanoid.yaml).
    Produces a 52-joint tree (SMPLH_MUJOCO_NAMES ordering).
    """
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": False,      # PHC smplx_humanoid.yaml: has_upright_start: False
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smplx",
        "ball_joint": False,
        "create_vel_sensors": False,
        "sim": "isaacgym",
    }
    robot = LocalRobot(robot_cfg, data_dir=smpl_data_dir)
    robot.load_from_skeleton(betas=torch.zeros(1, 16), objs_info=None, gender=[gender])

    xml_path = f"/tmp/smplx_{gender}_phc_bridge.xml"
    robot.write_xml(xml_path)
    sk_tree = SkeletonTree.from_mjcf(xml_path)

    assert sk_tree.num_joints == 52, (
        f"Expected 52-joint SMPL-X skeleton, got {sk_tree.num_joints}. "
        "Check the SMPL-X body model files."
    )
    return sk_tree


# ── Per-person conversion ─────────────────────────────────────────────────────

def convert_person(
    person_data: np.ndarray,     # (T, 56, 6)
    skeleton_tree: SkeletonTree,
    fps: int = 30,
) -> dict:
    """
    Convert one person's InterMask slice to a PHC motion dict entry.

    InterMask joint layout (SMPLX_BONE_ORDER_NAMES, slots 0-55):
      0-54  : SMPL-X joint 6D rotations  (6D = first two columns of rotation matrix)
      55    : root translation pseudo-joint  [:3] = (tx, ty, tz) in SMPL Y-up frame

    Coordinate system:
      InterMask stores SMPL-X parameters in SMPL native Y-up convention:
        - Y axis = height (positive Y = up)
        - Root orient identity = T-pose standing in Y-up

      PHC (Isaac Gym, upright_start=False) expects Z-up:
        - Z axis = height (positive Z = up)
        - Root orient for standing = Rx(+90°) applied to Y-up T-pose

      Evidence: standing_x.pkl root quat ≈ [0.689, -0.044, -0.051, 0.722]
                which corresponds to Rx(+87°) ≈ Rx(+90°) from identity.

      Transform applied here: R_yup2zup = Rx(+90°)
        - trans:       (x, y_height, z) → (x, -z, y_height) i.e. Z gains height ✓
        - root orient: R_phc = R_yup2zup * R_intermask

    Returns a dict with exactly the keys present in standing_x.pkl:
      pose_quat_global  : (T, 52, 4) global joint quats   dtype float64
      pose_quat         : (T, 52, 4) local  joint quats   dtype float64
      trans_orig        : (T, 3)     original translation  dtype float64
      root_trans_offset : torch.Tensor (T, 3) double — trans + skeleton rest-pose offset
      pose_aa           : (T, 156)   flat axis-angle       dtype float64
      beta              : (16,)      shape params (zeros)  dtype float64
      gender            : numpy str  "neutral"
      fps               : int
    """
    T = person_data.shape[0]

    # Coordinate transform: SMPL Y-up → PHC Z-up (Rx +90 degrees)
    R_yup2zup = sRot.from_euler('x', 90, degrees=True)

    # 1. Root translation from pseudo-joint slot 55 (Y-up: Y is height)
    trans_yup = person_data[:, 55, :3].astype(np.float64)   # (T, 3)  Y-up
    trans = R_yup2zup.apply(trans_yup)                       # (T, 3)  Z-up

    # 2. 6D rotations → rotation matrices → axis-angle for joints 0-54
    rot_6d  = torch.tensor(person_data[:, :55, :], dtype=torch.float32)   # (T, 55, 6)
    rot_mat = rotation_6d_to_matrix(rot_6d)                               # (T, 55, 3, 3)
    pose_aa_full = matrix_to_axis_angle(rot_mat).numpy().astype(np.float64)  # (T, 55, 3)

    # 3. Remove face joints (Jaw=22, L_Eye_Smplhf=23, R_Eye_Smplhf=24)
    #    Remaining 52 joints are in SMPLH_BONE_ORDER_NAMES order.
    pose_aa_smplh = pose_aa_full[:, _KEEP_SMPLX, :]   # (T, 52, 3)

    # 4. Reindex from SMPLH bone order → SMPLH Mujoco order (PHC's DoF ordering)
    pose_aa_mj = pose_aa_smplh[:, _SMPLH_TO_MUJOCO, :]   # (T, 52, 3)

    # 5. Apply Y-up → Z-up rotation to root orientation only (joint index 0 = Pelvis).
    #    Body joint rotations are relative to their parent → frame-independent → no change.
    root_orient_yup = sRot.from_rotvec(pose_aa_mj[:, 0, :])   # (T,)
    root_orient_zup = R_yup2zup * root_orient_yup              # pre-multiply
    pose_aa_mj[:, 0, :] = root_orient_zup.as_rotvec()

    # 6. Axis-angle → local quaternions [x, y, z, w]  (scipy convention)
    pose_quat_local = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(T, 52, 4)
    )   # (T, 52, 4)

    # 7. root_trans_offset = trans (Z-up) + skeleton rest-pose origin offset
    #    (skeleton_tree.local_translation[0] is the Pelvis body-frame offset in the MJCF)
    root_trans_offset = (
        torch.from_numpy(trans) + skeleton_tree.local_translation[0]
    ).double()   # torch.Tensor (T, 3)

    # 8. Forward kinematics: local quats + root translation → global quats
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_local).float(),
        root_trans_offset.float(),
        is_local=True,
    )
    pose_quat_global = sk_state.global_rotation.numpy().astype(np.float64)  # (T, 52, 4)

    return {
        "pose_quat_global":  pose_quat_global,
        "pose_quat":         pose_quat_local.astype(np.float64),
        "trans_orig":        trans,
        "root_trans_offset": root_trans_offset,
        "pose_aa":           pose_aa_mj.reshape(T, -1),           # (T, 156)
        "beta":              np.zeros(16, dtype=np.float64),
        "gender":            np.str_("neutral"),
        "fps":               fps,
    }


# ── Sanity checks ─────────────────────────────────────────────────────────────

def sanity_check(motion_dict: dict) -> None:
    """Print a quick validation report similar to standing_x.pkl inspection."""
    for name, entry in motion_dict.items():
        T = entry["pose_quat_global"].shape[0]
        root_z = entry["root_trans_offset"][:, 2]
        print(f"\n  [{name}]  T={T} frames @ {entry['fps']} fps")
        print(f"    pose_quat_global : {entry['pose_quat_global'].shape}  "
              f"dtype={entry['pose_quat_global'].dtype}")
        print(f"    pose_aa          : {entry['pose_aa'].shape}  "
              f"dtype={entry['pose_aa'].dtype}")
        print(f"    root Z (height)  : "
              f"min={root_z.min():.3f}m  max={root_z.max():.3f}m  "
              f"mean={root_z.mean():.3f}m")
        if root_z.min() < -0.3:
            print("    ⚠  WARNING: root_trans_offset Z < -0.3 m — "
                  "humanoid may be below ground. Check coordinate system.")
        if root_z.mean() < 0.5:
            print("    ⚠  WARNING: mean root height < 0.5 m — "
                  "check that the input .npy uses Z-up (height in Z).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert InterMask .npy → PHC SMPL-X .pkl (Z-up, 52 joints)"
    )
    p.add_argument(
        "--npy", required=True,
        help="Path to InterMask output .npy  (expected raw shape: (T*56*6,) or (T,2,56,6))"
    )
    p.add_argument(
        "--out", required=True,
        help="Output .pkl path, e.g. output/my_motion_phc.pkl"
    )
    p.add_argument(
        "--smpl_data_dir", default="PHC/data/smpl",
        help="Directory with SMPLX_NEUTRAL.npz and friends (default: PHC/data/smpl)"
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--person", type=int, choices=[0, 1, -1], default=-1,
        help="Which person to convert: 0, 1, or -1 for both (default: -1 = both)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading InterMask output: {args.npy}")
    raw = np.load(args.npy)
    print(f"      Raw shape: {raw.shape}")
    # InterMask infer.py saves (T, 2*56*6) or similar flat layouts; reshape robustly.
    total = raw.size
    # Must be divisible by (2 * 56 * 6)
    T = total // (2 * 56 * 6)
    if T * 2 * 56 * 6 != total:
        raise ValueError(
            f"Cannot reshape array of size {total} into (T, 2, 56, 6). "
            f"Expected total elements divisible by {2*56*6}. "
            "Check that this is an InterX-format output."
        )
    data = raw.reshape(T, 2, 56, 6)
    print(f"      Reshaped to: {data.shape}  (T={T}, 2 persons, 56 joints, 6D)")

    # ── Build skeleton ──────────────────────────────────────────────────────
    print(f"\n[2/4] Building SMPL-X SkeletonTree (upright_start=False)…")
    sk_tree = build_skeleton_tree(args.smpl_data_dir, gender=0)
    print(f"      {sk_tree.num_joints} joints  |  "
          f"local_translation[0] = {sk_tree.local_translation[0].numpy()}")

    # ── Convert ─────────────────────────────────────────────────────────────
    persons = [0, 1] if args.person == -1 else [args.person]
    motion_dict = {}

    print(f"\n[3/4] Converting {len(persons)} person(s)…")
    for p in persons:
        name = f"intermask_p{p}"
        print(f"      person {p} → key '{name}'")
        motion_dict[name] = convert_person(data[:, p], sk_tree, args.fps)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[4/4] Saving to {args.out}…")
    joblib.dump(motion_dict, args.out)
    print(f"      {len(motion_dict)} motion(s) saved.")

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\nSanity check:")
    sanity_check(motion_dict)

    # ── PHC command ──────────────────────────────────────────────────────────
    n_envs = len(motion_dict)
    rel_out = os.path.relpath(os.path.abspath(args.out), os.path.join(_ROOT, "PHC"))
    print(f"""
PHC evaluation command (run from PHC/ directory):

  # Single-motion visual test
  python phc/run_hydra.py \\
      learning=im_pnn_big exp_name=phc_x_pnn env=env_im_x_pnn \\
      robot=smplx_humanoid \\
      env.motion_file={rel_out} \\
      env.training_prim=0 epoch=-1 test=True \\
      env.num_envs=1 headless=False

  # Batch eval with success-rate metrics
  python phc/run_hydra.py \\
      learning=im_pnn_big exp_name=phc_x_pnn env=env_im_x_pnn \\
      robot=smplx_humanoid \\
      env.motion_file={rel_out} \\
      env.training_prim=0 epoch=-1 im_eval=True \\
      env.num_envs={n_envs} headless=True
""")


if __name__ == "__main__":
    main()
