#!/usr/bin/env python3
"""Convert InterHuman .pkl motions to ProtoMotions .motion format.

InterHuman .pkl data is Z-up (root_orient and trans pre-rotated via R_x(+90)).
The ProtoMotions MJCF body model uses Z-up body-local offsets, so InterHuman
data can be fed directly into the standard AMASS conversion pipeline (including
rot1) without any coordinate undo.

Each pkl contains person1/person2 with SMPL-X body params (21 body joints).
We pad with 2 zero hand joints to get SMPL 24-joint format, reorder to
MuJoCo joint order, run FK, and save in ProtoMotions .motion format.

Usage:
    cd prepare7/ProtoMotions
    python ../../prepare7/convert_interhuman_to_proto.py \
        --interhuman-dir ../../data/InterHuman \
        --output-dir ../data/interhuman_motions \
        --output-fps 30
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

logger = logging.getLogger("convert_interhuman")

# Add ProtoMotions to path
PROTO_ROOT = Path(__file__).resolve().parent / "ProtoMotions"
sys.path.insert(0, str(PROTO_ROOT))
sys.path.insert(0, str(PROTO_ROOT / "data" / "scripts"))

from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quaternion_to_matrix,
)
from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
    compute_angular_velocity,
    compute_forward_kinematics_from_transforms,
    compute_joint_rot_mats_from_global_mats,
)
from contact_detection import compute_contact_labels_from_pos_and_vel

FOOT_HEIGHT_OFFSET = 0.015  # SMPL foot sole thickness


def closest_divisor_larger_than_target(rounded_fps, target_fps):
    divisors = [i for i in range(1, rounded_fps + 1) if rounded_fps % i == 0]
    larger_divisors = [d for d in divisors if d >= target_fps]
    return min(larger_divisors) if larger_divisors else None


def load_interhuman_pkl(pkl_path):
    """Load InterHuman pkl, return list of per-person dicts."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    mocap_fr = raw.get("mocap_framerate", 30)
    results = []
    for person_key in ["person1", "person2"]:
        if person_key not in raw:
            continue
        p = raw[person_key]
        results.append(
            {
                "root_orient": p["root_orient"].astype(np.float64),  # (T, 3)
                "pose_body": p["pose_body"].astype(np.float64),  # (T, 63)
                "trans": p["trans"].astype(np.float64),  # (T, 3)
            }
        )
    return results, mocap_fr


def convert_person_to_motion(
    root_orient,
    pose_body,
    trans,
    mocap_fr,
    output_fps,
    kinematic_info,
    smpl_2_mujoco,
    device,
    dtype,
):
    """Convert a single person's SMPL-X body params to ProtoMotions .motion format.

    InterHuman .pkl data has root_orient and trans in Z-up coordinates.
    The ProtoMotions MJCF body model also uses Z-up body-local offsets.
    We feed the InterHuman data directly into the standard AMASS conversion
    pipeline (including rot1), which produces correctly Z-up output because
    InterHuman's root_orient already encodes the Z-up world orientation.
    No coordinate undo is needed.
    """
    # Downsample to target FPS
    largest_divisor = closest_divisor_larger_than_target(mocap_fr, output_fps)
    if largest_divisor is not None:
        downsample_factor = mocap_fr // largest_divisor
        root_orient = root_orient[::downsample_factor]
        pose_body = pose_body[::downsample_factor]
        trans = trans[::downsample_factor]
        current_fps = largest_divisor
    else:
        current_fps = mocap_fr

    T = root_orient.shape[0]
    if T < 2:
        return None, current_fps

    # ── Standard AMASS → ProtoMotions pipeline (matches convert_amass_to_proto.py) ──
    # InterHuman Z-up root_orient + trans work directly with this pipeline because
    # the MJCF body model has Z-up offsets and rot1 preserves the Z-up convention.

    # Construct SMPL 24-joint pose: root(3) + body(63) + hand_zeros(6) = 72
    pose_aa = np.concatenate(
        [root_orient, pose_body, np.zeros((T, 6))], axis=1
    )  # (T, 72)

    # Reorder from SMPL bone order to MuJoCo order
    pose_aa_mj = pose_aa.reshape(T, 24, 3)[:, smpl_2_mujoco]  # (T, 24, 3)

    # Axis-angle → quaternion (scipy outputs xyzw) → rotation matrices
    pose_quat = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(T, 24, 4)
    )

    amass_trans = torch.from_numpy(trans).to(device, dtype)
    pose_quat = torch.from_numpy(pose_quat).to(device, dtype)
    local_rot_mats = quaternion_to_matrix(pose_quat, w_last=True)

    # FK → world rotations
    _, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, amass_trans, local_rot_mats
    )
    global_quat = matrix_to_quaternion(world_rot_mat, w_last=True)

    # Apply rot1: standard body-frame alignment rotation
    rot1 = sRot.from_euler("xyz", np.array([-np.pi / 2, -np.pi / 2, 0]), degrees=False)
    rot1_quat = (
        torch.from_numpy(rot1.as_quat())
        .to(device, dtype)
        .expand(T, -1)
    )
    n_j = 23  # SMPL: 24 bodies - 1 root
    for i in range(0, n_j + 1):
        global_quat[:, i, :] = quat_mul(global_quat[:, i, :], rot1_quat, w_last=True)

    # Recompute local rotations from rotated global rotations
    local_rot_mats_rotated = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    # FK with velocities → RobotState
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        fps=current_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    # Store local quaternions for MotionLib interpolation
    motion.local_rigid_body_rot = matrix_to_quaternion(
        local_rot_mats_rotated, w_last=True
    ).clone()

    # DOF positions (exp_map, skip root 7 DOFs)
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]

    # DOF velocities
    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats_rotated[:, 1:, :, :],
        fps=current_fps,
    )
    motion.dof_vel = local_angular_vels.reshape(-1, n_j * 3)

    # Fix height so minimum body sits at ground + offset
    motion.fix_height(height_offset=FOOT_HEIGHT_OFFSET)

    # Contact labels
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    return motion, current_fps


def main():
    parser = argparse.ArgumentParser(
        description="Convert InterHuman motions to ProtoMotions format"
    )
    parser.add_argument(
        "--interhuman-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "InterHuman",
        help="Path to InterHuman dataset root (contains motions/ folder)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "interhuman_motions",
        help="Output directory for .motion files",
    )
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument(
        "--clip-ids",
        type=str,
        default=None,
        help="Comma-separated clip IDs to convert (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # --- Logging setup ---
    log_path = args.output_dir / "convert.log"
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("=" * 60)
    logger.info("InterHuman → ProtoMotions Conversion")
    logger.info("=" * 60)
    logger.info(f"  InterHuman dir: {args.interhuman_dir}")
    logger.info(f"  Output dir:     {args.output_dir}")
    logger.info(f"  Output FPS:     {args.output_fps}")
    logger.info(f"  Force:          {args.force}")
    logger.info(f"  Log file:       {log_path}")

    device = torch.device("cpu")
    dtype = torch.float32

    # Load kinematic info from SMPL MJCF
    mjcf_path = str(PROTO_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "smpl_humanoid.xml")
    logger.info(f"Loading kinematic info from {mjcf_path}")
    kinematic_info = extract_kinematic_info(mjcf_path)

    # Build joint reordering map
    smpl_2_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(q)
        for q in SMPL_MUJOCO_NAMES
        if q in SMPL_BONE_ORDER_NAMES
    ]

    # Discover clips — look in motions/ subdir first, then flat directory
    motions_dir = args.interhuman_dir / "motions"
    if not motions_dir.is_dir():
        motions_dir = args.interhuman_dir  # flat directory of .pkl files
    if args.clip_ids:
        clip_ids = args.clip_ids.split(",")
    else:
        clip_ids = sorted(
            [f.stem for f in motions_dir.glob("*.pkl")],
            key=lambda x: int(x.replace("(1)", "")) if x.replace("(1)", "").isdigit() else x,
        )

    logger.info(f"Found {len(clip_ids)} clips to convert")

    skipped = 0
    converted = 0
    errors = 0
    error_clips = []
    t_start = time.time()

    for clip_id in tqdm(clip_ids, desc="Converting"):
        pkl_path = motions_dir / f"{clip_id}.pkl"
        if not pkl_path.exists():
            logger.warning(f"Missing pkl: {pkl_path}")
            errors += 1
            error_clips.append((clip_id, "missing_pkl"))
            continue

        try:
            persons, mocap_fr = load_interhuman_pkl(pkl_path)
        except Exception as e:
            logger.error(f"Error loading {clip_id}: {e}")
            errors += 1
            error_clips.append((clip_id, str(e)))
            continue

        mocap_fr = int(np.round(mocap_fr))

        for pidx, person_data in enumerate(persons):
            person_tag = f"person{pidx + 1}"
            outpath = args.output_dir / f"{clip_id}_{person_tag}.motion"

            if outpath.exists() and not args.force:
                skipped += 1
                continue

            try:
                motion, fps = convert_person_to_motion(
                    root_orient=person_data["root_orient"],
                    pose_body=person_data["pose_body"],
                    trans=person_data["trans"],
                    mocap_fr=mocap_fr,
                    output_fps=args.output_fps,
                    kinematic_info=kinematic_info,
                    smpl_2_mujoco=smpl_2_mujoco,
                    device=device,
                    dtype=dtype,
                )
                if motion is None:
                    logger.warning(f"Skipping {clip_id} {person_tag}: too few frames")
                    continue

                os.makedirs(outpath.parent, exist_ok=True)
                torch.save(motion.to_dict(), str(outpath))
                converted += 1
            except Exception as e:
                logger.error(f"Error converting {clip_id} {person_tag}: {e}")
                errors += 1
                error_clips.append((f"{clip_id}_{person_tag}", str(e)))

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Converted: {converted}")
    logger.info(f"  Skipped:   {skipped}")
    logger.info(f"  Errors:    {errors}")
    logger.info(f"  Time:      {elapsed:.1f}s")
    if error_clips:
        logger.info(f"  Failed clips:")
        for clip, reason in error_clips:
            logger.info(f"    {clip}: {reason}")
    logger.info("=" * 60)


if __name__ == "__main__":
    with torch.no_grad():
        main()
