#!/usr/bin/env python3
"""
visualize_interhuman.py — Kinematic visualization of InterHuman motions in Isaac Gym.

No physics-driven motion — root + DOF states are overwritten every frame.

Accepts:
  - Raw InterHuman .pkl   (keys: 'person1', 'person2')
  - Converted PHC  .pkl   (output of interhuman_to_phc.py)

Usage (from repo root, inside Docker):
    # Raw InterHuman pkl (converts on-the-fly):
    python visualize_interhuman.py \\
        --motion InterHuman_dataset/motions/10.pkl \\
        --smpl_data_dir PHC/data/smpl

    # Already-converted PKL:
    python visualize_interhuman.py \\
        --motion PHC/output/interhuman/10_phc.pkl \\
        --smpl_data_dir PHC/data/smpl

Controls (Isaac Gym viewer window):
    Space     — pause / resume
    →  /  ←   — step +1 / -1 frame
    R         — reset to frame 0
    Q / Esc   — quit
"""

import sys
import os
import argparse
import pickle
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PHC  = os.path.join(_ROOT, "PHC")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _PHC)

# Isaac Gym must be imported before torch
from isaacgym import gymapi, gymutil, gymtorch

import numpy as np
import torch
import joblib
from scipy.spatial.transform import Rotation as sRot

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

_N_JOINTS    = len(SMPL_MUJOCO_NAMES)                                  # 24
_NUM_DOFS    = (_N_JOINTS - 1) * 3                                     # 69  (non-root × 3)
_SMPL_TO_MJ  = [SMPL_BONE_ORDER_NAMES.index(j) for j in SMPL_MUJOCO_NAMES]


# ── Skeleton / XML helpers ────────────────────────────────────────────────────

def build_robot_xml(smpl_data_dir: str, betas: np.ndarray,
                    gender: str = "neutral", tag: str = "vis") -> tuple:
    """Build subject-specific SMPL skeleton + MuJoCo XML.  Returns (sk_tree, xml_path)."""
    gender_id = {"neutral": 0, "male": 1, "female": 2}.get(str(gender), 0)
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": True,    # must match smpl_humanoid.yaml (has_upright_start: True)
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "body_params": {}, "joint_params": {},
        "geom_params": {}, "actuator_params": {},
        "model": "smpl",
        "ball_joint": False,
        "create_vel_sensors": False,
        "sim": "isaacgym",
    }
    robot = LocalRobot(robot_cfg, data_dir=smpl_data_dir)
    betas_16 = np.zeros(16, dtype=np.float32)
    n = min(len(betas), 16)
    betas_16[:n] = betas[:n]
    robot.load_from_skeleton(
        betas=torch.tensor(betas_16, dtype=torch.float32).unsqueeze(0),
        objs_info=None,
        gender=[gender_id],
    )
    xml_path = f"/tmp/smpl_vis_{tag}_{gender_id}.xml"
    robot.write_xml(xml_path)
    sk_tree = SkeletonTree.from_mjcf(xml_path)
    return sk_tree, xml_path


# ── Per-person data extraction ────────────────────────────────────────────────

def extract_from_raw(pdata: dict, smpl_data_dir: str, step: int, tag: str) -> dict:
    """Convert raw InterHuman person dict → dict with root_pos / root_rot / dof_pos."""
    trans   = pdata["trans"][::step].astype(np.float64)
    root_aa = pdata["root_orient"][::step].astype(np.float64)
    body_aa = pdata["pose_body"][::step].astype(np.float64)
    betas   = pdata["betas"].astype(np.float64)
    gender  = pdata.get("gender", "neutral")
    T       = trans.shape[0]

    sk_tree, xml_path = build_robot_xml(smpl_data_dir, betas, gender, tag)

    # Build 24-joint axis-angle in MuJoCo order
    pose_aa_22 = np.concatenate([root_aa[:, None, :], body_aa.reshape(T, 21, 3)], axis=1)
    pose_aa_24 = np.concatenate([pose_aa_22, np.zeros((T, 2, 3))], axis=1)
    pose_aa_mj = pose_aa_24[:, _SMPL_TO_MJ, :]   # (T, 24, 3)

    # Local quaternions for FK
    local_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(T, _N_JOINTS, 4)
    root_trans_offset = (torch.from_numpy(trans) + sk_tree.local_translation[0]).numpy().astype(np.float32)

    # FK → global rotations
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree,
        torch.from_numpy(local_quat).float(),
        torch.from_numpy(root_trans_offset).float(),
        is_local=True,
    )
    pose_quat_global = sk_state.global_rotation.numpy()  # (T, J, 4)

    # Apply upright_start correction (same as convert_amass_isaac.py)
    R_correction_inv = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    pose_quat_global = (
        sRot.from_quat(pose_quat_global.reshape(-1, 4)) * R_correction_inv
    ).as_quat().reshape(T, _N_JOINTS, 4)

    # Re-derive local rotations from corrected global
    sk_state2 = SkeletonState.from_rotation_and_root_translation(
        sk_tree,
        torch.from_numpy(pose_quat_global).float(),
        torch.from_numpy(root_trans_offset).float(),
        is_local=False,
    )
    local_quat_corrected = sk_state2.local_rotation.numpy()

    root_rot = pose_quat_global[:, 0, :].astype(np.float32)                         # (T, 4)
    dof_pos  = sRot.from_quat(local_quat_corrected[:, 1:].reshape(-1, 4)) \
                   .as_rotvec().reshape(T, -1).astype(np.float32)                   # (T, 69)

    return {
        "key":      tag,
        "xml_path": xml_path,
        "root_pos": root_trans_offset,  # (T, 3)
        "root_rot": root_rot,           # (T, 4)
        "dof_pos":  dof_pos,            # (T, 69)
        "fps":      30,
    }


def extract_from_phc(entry: dict, smpl_data_dir: str, key: str) -> dict:
    """Extract visualisation arrays from a converted PHC pkl entry."""
    betas   = np.array(entry["beta"])
    gender  = str(entry.get("gender", "neutral"))
    fps     = int(entry.get("fps", 30))

    pose_quat_global  = entry["pose_quat_global"].astype(np.float32)   # (T, 24, 4) MuJoCo order
    pose_quat_local   = entry["pose_quat"].astype(np.float32)          # (T, 24, 4) MuJoCo order
    root_trans_offset = entry["root_trans_offset"]
    if hasattr(root_trans_offset, "numpy"):
        root_trans_offset = root_trans_offset.numpy()
    root_trans_offset = np.array(root_trans_offset, dtype=np.float32)  # (T, 3)
    T = root_trans_offset.shape[0]

    tag = key.replace("/", "_").replace(" ", "_")
    _, xml_path = build_robot_xml(smpl_data_dir, betas, gender, tag)

    # DOF positions: axis-angle of non-root local joints, in MuJoCo order.
    # pose_aa in the pkl is in SMPL anatomical order (for the mesh parser) — do NOT use it
    # for DOFs.  Use pose_quat (local quats in MuJoCo order) instead.
    dof_pos = sRot.from_quat(pose_quat_local[:, 1:].reshape(-1, 4)) \
                  .as_rotvec().reshape(T, -1).astype(np.float32)  # (T, 69)

    return {
        "key":      key,
        "xml_path": xml_path,
        "root_pos": root_trans_offset,          # (T, 3)
        "root_rot": pose_quat_global[:, 0, :],  # (T, 4) root global quat, MuJoCo order
        "dof_pos":  dof_pos,                    # (T, 69)
        "fps":      fps,
    }


# ── Isaac Gym visualiser ──────────────────────────────────────────────────────

def run_viewer(entries: list, playback_fps: float) -> None:
    n_persons = len(entries)
    T         = min(e["root_pos"].shape[0] for e in entries)

    gym = gymapi.acquire_gym()

    # Sim params
    sim_params = gymapi.SimParams()
    sim_params.dt                    = 1.0 / 60.0
    sim_params.up_axis               = gymapi.UP_AXIS_Z
    sim_params.gravity               = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type     = 1
    sim_params.physx.num_threads     = 4
    sim_params.physx.use_gpu         = True
    sim_params.use_gpu_pipeline      = False   # CPU tensors → simpler

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    assert sim is not None, "create_sim failed"

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    assert viewer is not None, "create_viewer failed"

    # Camera: look at the centre of all persons
    cx = (n_persons - 1) * 0.5
    gym.viewer_camera_look_at(viewer, None,
        gymapi.Vec3(cx + 3.0, 0.0, 1.5),
        gymapi.Vec3(cx,       0.0, 1.0))

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE,  "pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT,  "step_fwd")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT,   "step_bwd")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,      "reset")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q,      "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")

    # One env per person, placed side-by-side
    spacing   = 2.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3( spacing,  spacing, spacing * 2)

    asset_opts = gymapi.AssetOptions()
    asset_opts.angular_damping            = 0.0
    asset_opts.linear_damping             = 0.0
    asset_opts.default_dof_drive_mode     = gymapi.DOF_MODE_NONE
    asset_opts.collapse_fixed_joints      = False
    asset_opts.replace_cylinder_with_capsule = True

    envs   = []
    actors = []
    for i, e in enumerate(entries):
        env = gym.create_env(sim, env_lower, env_upper, n_persons)
        init_pose   = gymapi.Transform()
        init_pose.p = gymapi.Vec3(i * 1.0, 0.0, 0.0)
        init_pose.r = gymapi.Quat(0, 0, 0, 1)
        asset = gym.load_asset(sim, "/", e["xml_path"], asset_opts)
        assert asset is not None, f"load_asset failed for {e['xml_path']}"
        actor = gym.create_actor(env, asset, init_pose, e["key"], i, 1)
        envs.append(env)
        actors.append(actor)
        n_dofs = gym.get_actor_dof_count(env, actor)
        print(f"  [{e['key']}]  T={e['root_pos'].shape[0]}  dofs={n_dofs}")
        assert n_dofs == _NUM_DOFS, f"Expected {_NUM_DOFS} DOFs, got {n_dofs}"

    gym.prepare_sim(sim)

    # Acquire state tensors (CPU, since use_gpu_pipeline=False)
    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))  # (n_persons, 13)
    dof_states  = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))          # (n_persons*69, 2)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    env_ids = torch.arange(n_persons, dtype=torch.int32)  # CPU int32

    # Pre-convert motion arrays to CPU tensors
    motion_tensors = [{
        "root_pos": torch.from_numpy(e["root_pos"]),  # (T, 3)
        "root_rot": torch.from_numpy(e["root_rot"]),  # (T, 4)
        "dof_pos":  torch.from_numpy(e["dof_pos"]),   # (T, 69)
    } for e in entries]

    frame    = 0
    paused   = False
    frame_dt = 1.0 / playback_fps
    last_t   = time.perf_counter()

    print(f"\nPlaying {T} frames @ {playback_fps:.0f} fps")
    print("Controls: Space=pause  ←/→=step  R=reset  Q/Esc=quit\n")

    while not gym.query_viewer_has_closed(viewer):
        # Handle keyboard
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value <= 0:
                continue
            if evt.action == "pause":
                paused = not paused
                print(f"  {'paused' if paused else 'playing'}  frame={frame}/{T}")
            elif evt.action == "step_fwd":
                frame = (frame + 1) % T
            elif evt.action == "step_bwd":
                frame = (frame - 1) % T
            elif evt.action == "reset":
                frame = 0
                print(f"  reset  frame=0")
            elif evt.action == "quit":
                gym.destroy_viewer(viewer)
                gym.destroy_sim(sim)
                return

        # Auto-advance
        now = time.perf_counter()
        if not paused and (now - last_t) >= frame_dt:
            frame  = (frame + 1) % T
            last_t = now

        # Write root states: [pos(3), rot(4), lin_vel(3), ang_vel(3)]
        for i, mt in enumerate(motion_tensors):
            root_states[i, 0:3] = mt["root_pos"][frame]
            root_states[i, 3:7] = mt["root_rot"][frame]
            root_states[i, 7:]  = 0.0

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(env_ids),
            n_persons,
        )

        # Write DOF states: [pos, vel] per DOF
        for i, mt in enumerate(motion_tensors):
            s = i * _NUM_DOFS
            dof_states[s:s + _NUM_DOFS, 0] = mt["dof_pos"][frame]
            dof_states[s:s + _NUM_DOFS, 1] = 0.0

        gym.set_dof_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(dof_states),
            gymtorch.unwrap_tensor(env_ids),
            n_persons,
        )

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Kinematic visualisation of InterHuman motions in Isaac Gym"
    )
    p.add_argument("--motion",        required=True,
                   help="Raw InterHuman .pkl or converted PHC .pkl")
    p.add_argument("--smpl_data_dir", default="PHC/data/smpl")
    p.add_argument("--person",        type=int, choices=[0, 1, -1], default=-1,
                   help="Person to show (raw pkl only): 0, 1, -1=both (default -1)")
    p.add_argument("--fps",           type=float, default=30.0,
                   help="Playback speed in fps (default 30)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        probe = joblib.load(args.motion)
    except Exception:
        with open(args.motion, "rb") as f:
            probe = pickle.load(f)
    is_raw = "person1" in probe

    print(f"\nLoading {'raw InterHuman' if is_raw else 'converted PHC'} pkl: {args.motion}")

    entries = []
    if is_raw:
        fps_raw = float(probe.get("mocap_framerate", 59.94))
        step    = round(fps_raw / 30)
        if args.person in (-1, 0):
            entries.append(extract_from_raw(probe["person1"], args.smpl_data_dir, step, "p0"))
        if args.person in (-1, 1):
            entries.append(extract_from_raw(probe["person2"], args.smpl_data_dir, step, "p1"))
    else:
        for key, entry in probe.items():
            entries.append(extract_from_phc(entry, args.smpl_data_dir, key))

    print(f"  {len(entries)} person(s) loaded")
    run_viewer(entries, args.fps)


if __name__ == "__main__":
    main()
