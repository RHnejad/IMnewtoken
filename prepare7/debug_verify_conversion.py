#!/usr/bin/env python3
"""Verify InterHuman conversion by analyzing .motion positions vs raw pkl.

Run from repo root:
    python prepare7/debug_verify_conversion.py
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import torch

PROTO_ROOT = Path(__file__).resolve().parent / "ProtoMotions"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROTO_ROOT))

from scipy.spatial.transform import Rotation as sRot
from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion
from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES


def main():
    # Load raw InterHuman pkl
    pkl_path = PROJECT_ROOT / "data/InterHuman/motions/1.pkl"
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    p = raw["person1"]
    trans_ih = p["trans"].astype(np.float64)
    ro_ih = p["root_orient"].astype(np.float64)

    print("=== Raw InterHuman pkl (person1, frame 0) ===")
    print(f"  trans[0]: {trans_ih[0]}")
    print(f"  root_orient[0]: {ro_ih[0]}")
    R_ih = sRot.from_rotvec(ro_ih[0])
    print(f"  root_orient euler XYZ (deg): {R_ih.as_euler('XYZ', degrees=True)}")

    # Our undo: R_x(-90) left-multiply
    R_x_neg90 = sRot.from_euler("X", -90, degrees=True)
    trans_yup = (R_x_neg90.as_matrix() @ trans_ih.T).T
    R_yup = R_x_neg90 * R_ih
    print(f"\n=== After R_x(-90) undo ===")
    print(f"  trans_yup[0]: {trans_yup[0]}")
    print(f"  root_orient_yup euler XYZ (deg): {R_yup.as_euler('XYZ', degrees=True)}")

    # Load ProtoMotions converted motion
    motion_path = PROJECT_ROOT / "prepare7/data/interhuman_test/1_person1.motion"
    if not motion_path.exists():
        motion_path = PROJECT_ROOT / "prepare7/data/interhuman_gt_motions/1_person1.motion"

    d = torch.load(motion_path, map_location="cpu", weights_only=False)
    motion = RobotState.from_dict(d, state_conversion=StateConversion.COMMON)

    pos = motion.rigid_body_pos.numpy()  # (T, 24, 3)
    rot = motion.rigid_body_rot.numpy()  # (T, 24, 4) xyzw

    print(f"\n=== ProtoMotions .motion (frame 0) ===")
    print(f"  Num frames: {pos.shape[0]}, Num bodies: {pos.shape[1]}")

    # MuJoCo body names
    print(f"\n  Joint positions (frame 0):")
    for i, name in enumerate(SMPL_MUJOCO_NAMES):
        p = pos[0, i]
        print(f"    {i:2d} {name:20s}: [{p[0]:8.4f}, {p[1]:8.4f}, {p[2]:8.4f}]")

    # Check pelvis quat
    pelvis_q = rot[0, 0]  # xyzw
    R_pelvis = sRot.from_quat(pelvis_q)
    print(f"\n  Pelvis quat (xyzw): {pelvis_q}")
    print(f"  Pelvis euler XYZ (deg): {R_pelvis.as_euler('XYZ', degrees=True)}")

    # What does rot1 look like?
    rot1 = sRot.from_euler("xyz", [-np.pi/2, -np.pi/2, 0])
    print(f"\n  rot1 quat (xyzw): {rot1.as_quat()}")
    print(f"  rot1 euler XYZ (deg): {rot1.as_euler('XYZ', degrees=True)}")

    # Expected pelvis: R_yup * rot1
    expected_pelvis = R_yup * rot1
    print(f"\n  Expected pelvis (R_yup * rot1) quat: {expected_pelvis.as_quat()}")
    print(f"  Actual pelvis quat:                   {pelvis_q}")
    print(f"  Match: {np.allclose(expected_pelvis.as_quat(), pelvis_q, atol=0.01)}")

    # What about AMASS reference?
    amass_path = PROJECT_ROOT / "prepare7/ProtoMotions/examples/data/smpl_humanoid_sit_armchair.motion"
    d2 = torch.load(amass_path, map_location="cpu", weights_only=False)
    motion2 = RobotState.from_dict(d2, state_conversion=StateConversion.COMMON)
    pos2 = motion2.rigid_body_pos.numpy()
    rot2 = motion2.rigid_body_rot.numpy()

    print(f"\n=== AMASS reference (frame 0) ===")
    for i, name in enumerate(SMPL_MUJOCO_NAMES):
        p = pos2[0, i]
        print(f"    {i:2d} {name:20s}: [{p[0]:8.4f}, {p[1]:8.4f}, {p[2]:8.4f}]")

    # Skeleton geometry check: compute distances between parent-child pairs
    print(f"\n=== Skeleton geometry check (frame 0) ===")
    print("  Checking if bone lengths match between InterHuman and AMASS:")
    parent_child_pairs = [
        (0, 1, "Pelvis->L_Hip"),
        (0, 2, "Pelvis->R_Hip"),
        (0, 3, "Pelvis->Torso"),
        (3, 6, "Torso->Spine"),
    ]
    for pi, ci, name in parent_child_pairs:
        d_ih = np.linalg.norm(pos[0, ci] - pos[0, pi])
        d_amass = np.linalg.norm(pos2[0, ci] - pos2[0, pi])
        print(f"    {name}: IH={d_ih:.4f}, AMASS={d_amass:.4f}")


if __name__ == "__main__":
    main()
