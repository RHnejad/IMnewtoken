#!/usr/bin/env python3
"""
interhuman_to_phc.py — Convert InterHuman .pkl motions to PHC SMPL .pkl format.

InterHuman motion .pkl layout (one file, two persons):
    person1 / person2:
        trans        (T, 3)   float32  root translation in meters, Z-up (mocap world frame)
        root_orient  (T, 3)   float32  root orientation axis-angle
        pose_body    (T, 63)  float32  21 body joints × 3 axis-angle (SMPL body, no hands)
        betas        (10,)    float32  SMPL shape parameters
        gender       str      'neutral'
    mocap_framerate  float    59.94
    frames           int      T

Joints: root_orient (1) + pose_body (21) = 22 joints.
        SMPL has 24 joints; joints 22-23 (L/R hand) are missing → zero-padded.

Coordinate system:
    InterHuman uses a Z-up world frame (mocap convention), unlike raw SMPL Y-up.
    No coordinate transform needed — trans and root_orient are already in Z-up.

PHC SMPL expects (robot: smpl_humanoid):
    pose_quat_global  (T, J, 4)    global quaternions [x,y,z,w]
    pose_quat         (T, J, 4)    local  quaternions
    trans_orig        (T, 3)       root translation (meters, Z-up)
    root_trans_offset (T, 3)       trans + skeleton rest-pose pelvis offset [torch double]
    pose_aa           (T, J*3)     flat axis-angle
    beta              (10,)        SMPL shape params
    gender            str          'neutral'
    fps               int

Usage (from repo root):
    python interhuman_to_phc.py \\
        --motion  InterHuman_dataset/motions/1000.pkl \\
        --out     output/1000_phc.pkl \\
        --smpl_data_dir PHC/data/smpl

    # single person only (0 or 1):
    python interhuman_to_phc.py \\
        --motion InterHuman_dataset/motions/1000.pkl \\
        --out    output/1000_phc.pkl \\
        --person 0

Then run PHC (from PHC/ directory):
    python phc/run_hydra.py \\
        learning=im_pnn_big exp_name=phc_pnn env=env_im_pnn \\
        robot=smpl_humanoid \\
        env.motion_file=../output/1000_phc.pkl \\
        env.training_prim=0 epoch=-1 test=True \\
        env.num_envs=1 headless=False
"""

import sys
import os
import argparse
import pickle

import numpy as np
import torch
import joblib
from scipy.spatial.transform import Rotation as sRot

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PHC  = os.path.join(_ROOT, "PHC")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _PHC)

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# InterHuman has 22 joints (root + 21 body); SMPL has 24.
# Pad joints 22-23 (L/R hand) with identity (zeros in axis-angle).
_N_INTERHUMAN_JOINTS = 22
_N_SMPL_JOINTS       = len(SMPL_MUJOCO_NAMES)   # 24

# Reindex: SMPL anatomical order → MuJoCo articulation order
_SMPL_TO_MUJOCO = [SMPL_BONE_ORDER_NAMES.index(j) for j in SMPL_MUJOCO_NAMES]


# ── Skeleton builder ──────────────────────────────────────────────────────────

def build_skeleton_tree(smpl_data_dir: str, betas: np.ndarray, gender: str = "neutral") -> SkeletonTree:
    """
    Build a subject-specific SMPL SkeletonTree using real betas.
    upright_start=False → matches smpl_humanoid.yaml in PHC.
    """
    gender_id = {"neutral": 0, "male": 1, "female": 2}.get(gender, 0)
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": False,
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
        "model": "smpl",
        "ball_joint": False,
        "create_vel_sensors": False,
        "sim": "isaacgym",
    }
    robot = LocalRobot(robot_cfg, data_dir=smpl_data_dir)
    robot.load_from_skeleton(
        betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
        objs_info=None,
        gender=[gender_id],
    )

    xml_path = f"/tmp/smpl_{gender_id}_phc_bridge.xml"
    robot.write_xml(xml_path)
    sk_tree = SkeletonTree.from_mjcf(xml_path)

    assert sk_tree.num_joints == _N_SMPL_JOINTS, (
        f"Expected {_N_SMPL_JOINTS}-joint SMPL skeleton, got {sk_tree.num_joints}."
    )
    return sk_tree


# ── Per-person conversion ─────────────────────────────────────────────────────

def convert_person(person: dict, smpl_data_dir: str, fps: int) -> dict:
    """
    Convert one InterHuman person dict to a PHC motion entry.

    Coordinate system note:
        InterHuman stores mocap data in Z-up world frame.
        No Rx(+90°) needed (unlike raw SMPL Y-up from fitting).
        The trans values (-1..+1 m) and root_orient are already Z-up.
    """
    trans      = person["trans"].astype(np.float64)       # (T, 3)  Z-up, meters
    root_aa    = person["root_orient"].astype(np.float64)  # (T, 3)
    body_aa    = person["pose_body"].astype(np.float64)    # (T, 63) = 21 joints × 3
    betas      = person["betas"].astype(np.float64)        # (10,)
    gender     = person.get("gender", "neutral")
    T = trans.shape[0]

    # Build subject-specific skeleton
    sk_tree = build_skeleton_tree(smpl_data_dir, betas, gender)

    # Assemble full 22-joint pose: root_orient + 21 body joints
    pose_aa_22 = np.concatenate(
        [root_aa[:, None, :], body_aa.reshape(T, 21, 3)], axis=1
    )  # (T, 22, 3)

    # Zero-pad to 24 joints (add L/R hand as identity)
    pose_aa_24 = np.concatenate(
        [pose_aa_22, np.zeros((T, 2, 3), dtype=np.float64)], axis=1
    )  # (T, 24, 3)

    # Reindex from SMPL anatomical order → MuJoCo articulation order
    pose_aa_mj = pose_aa_24[:, _SMPL_TO_MUJOCO, :]  # (T, 24, 3)

    # Axis-angle → local quaternions [x, y, z, w]
    pose_quat_local = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(T, _N_SMPL_JOINTS, 4)
    )

    # root_trans_offset = trans + rest-pose pelvis offset from skeleton
    root_trans_offset = (
        torch.from_numpy(trans) + sk_tree.local_translation[0]
    ).double()

    # Forward kinematics: local quats + root translation → global quats
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree,
        torch.from_numpy(pose_quat_local).float(),
        root_trans_offset.float(),
        is_local=True,
    )
    pose_quat_global = sk_state.global_rotation.numpy().astype(np.float64)

    return {
        "pose_quat_global":  pose_quat_global,
        "pose_quat":         pose_quat_local.astype(np.float64),
        "trans_orig":        trans,
        "root_trans_offset": root_trans_offset,
        "pose_aa":           pose_aa_mj.reshape(T, -1),
        "beta":              betas,
        "gender":            np.str_(gender),
        "fps":               fps,
    }


# ── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check(motion_dict: dict) -> None:
    for name, entry in motion_dict.items():
        T   = entry["pose_quat_global"].shape[0]
        rz  = entry["root_trans_offset"][:, 2]
        print(f"\n  [{name}]  T={T} frames @ {entry['fps']} fps")
        print(f"    pose_quat_global : {entry['pose_quat_global'].shape}  dtype={entry['pose_quat_global'].dtype}")
        print(f"    root Z (height)  : min={rz.min():.3f}m  max={rz.max():.3f}m  mean={rz.mean():.3f}m")
        if rz.min() < -0.3:
            print("    WARNING: root Z < -0.3m — umanoid potrebbe essere sotto il piano.")
        if rz.mean() < 0.4:
            print("    WARNING: media root height < 0.4m — controlla sistema di coordinate.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert InterHuman .pkl → PHC SMPL .pkl"
    )
    p.add_argument("--motion", required=True,
                   help="Path to InterHuman motion .pkl (es. InterHuman_dataset/motions/1000.pkl)")
    p.add_argument("--out", required=True,
                   help="Output .pkl path (es. output/1000_phc.pkl)")
    p.add_argument("--smpl_data_dir", default="PHC/data/smpl",
                   help="Directory con SMPL_NEUTRAL.pkl etc. (default: PHC/data/smpl)")
    p.add_argument("--person", type=int, choices=[0, 1, -1], default=-1,
                   help="Persona da convertire: 0, 1, -1=entrambe (default: -1)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n[1/4] Loading {args.motion}")
    with open(args.motion, "rb") as f:
        raw = pickle.load(f)

    fps_raw = float(raw.get("mocap_framerate", 59.94))
    # Resample to 30 fps for PHC (drop frames)
    step = round(fps_raw / 30)
    fps  = 30
    print(f"      mocap fps={fps_raw:.2f}  →  downsampling step={step}  →  {fps} fps")

    persons_raw = {}
    if args.person in (-1, 0):
        persons_raw["p0"] = {k: v[::step] if hasattr(v, '__len__') and len(np.shape(v)) >= 1 and np.shape(v)[0] > 1 else v
                             for k, v in raw["person1"].items()}
    if args.person in (-1, 1):
        persons_raw["p1"] = {k: v[::step] if hasattr(v, '__len__') and len(np.shape(v)) >= 1 and np.shape(v)[0] > 1 else v
                             for k, v in raw["person2"].items()}

    print(f"      {len(persons_raw)} persona/e  |  T_raw={raw['frames']}  T_out={raw['frames']//step}")

    motion_dict = {}
    for i, (key, person) in enumerate(persons_raw.items()):
        print(f"\n[{i+2}/4] Converting {key}…")
        motion_dict[f"interhuman_{key}"] = convert_person(person, args.smpl_data_dir, fps)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    step_n = len(persons_raw) + 2
    print(f"\n[{step_n}/4] Saving → {args.out}")
    joblib.dump(motion_dict, args.out)
    print(f"      {len(motion_dict)} motion/i salvate.")

    print("\nSanity check:")
    sanity_check(motion_dict)

    rel_out = os.path.relpath(os.path.abspath(args.out), _PHC)
    n_envs  = len(motion_dict)
    print(f"""
PHC command (da PHC/ directory):

  python phc/run_hydra.py \\
      learning=im_pnn_big exp_name=phc_pnn env=env_im_pnn \\
      robot=smpl_humanoid \\
      env.motion_file={rel_out} \\
      env.training_prim=0 epoch=-1 test=True \\
      env.num_envs={n_envs} headless=False
""")


if __name__ == "__main__":
    main()
