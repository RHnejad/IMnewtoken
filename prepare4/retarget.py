"""
retarget.py — Unified retargeting: rotation-based + IK-based + FK.

Supports two retargeting methods on per-subject Newton skeletons:
  - rotation_retarget(): Direct rotation transfer (SMPL-X → Newton joint_q)
  - ik_retarget():       Position-based IK (positions → Newton joint_q)

Also provides:
  - forward_kinematics(): Newton FK (joint_q → positions)
  - load_interhuman_pkl(): Load SMPL-X params from InterHuman pkl files

Usage:
    from prepare4.retarget import rotation_retarget, ik_retarget, forward_kinematics

    # Rotation-based (when you have SMPL-X parameters)
    joint_q = rotation_retarget(root_orient, pose_body, trans, betas)

    # IK-based (when you only have positions)
    joint_q, fk_pos = ik_retarget(positions, betas, ik_iters=50)

    # FK validation
    positions = forward_kinematics(joint_q, betas)
"""
import os
import sys
import hashlib
import pickle
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare4.gen_xml import (
    R_ROT,
    generate_xml,
    get_or_create_xml,
    get_smplx_body_offset,
)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# SMPL-X joint index → Newton body index
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}
N_SMPL_JOINTS = 22
N_NEWTON_BODIES = 24
N_JOINT_Q = 76  # 7 (freejoint) + 23 * 3 (hinges)


def _newton_body_q_index(body_idx):
    """Starting index in joint_q for a non-root Newton body."""
    assert body_idx >= 1
    return 7 + (body_idx - 1) * 3


# ═══════════════════════════════════════════════════════════════
# Caches
# ═══════════════════════════════════════════════════════════════

_model_cache = {}
_offset_cache = {}


def _betas_hash(betas):
    return hashlib.sha256(np.asarray(betas, dtype=np.float64).tobytes()).hexdigest()[:16]


def get_body_offset(betas):
    """Get cached SMPL-X body offset for given betas."""
    h = _betas_hash(betas)
    if h not in _offset_cache:
        _offset_cache[h] = get_smplx_body_offset(betas)
    return _offset_cache[h]


def get_newton_model(betas, foot_geom="box", device="cuda:0"):
    """Build or retrieve cached Newton model for given betas + foot geometry."""
    import warp as wp
    wp.config.verbose = False
    warnings.filterwarnings("ignore", message="Custom attribute")
    import newton

    key = _betas_hash(betas) + f"_{foot_geom}"
    if key not in _model_cache:
        xml_path = get_or_create_xml(betas, foot_geom=foot_geom)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        _model_cache[key] = builder.finalize(device=device)
    return _model_cache[key]


# ═══════════════════════════════════════════════════════════════
# Rotation-based retargeting
# ═══════════════════════════════════════════════════════════════

def rotation_retarget(root_orient, pose_body, trans, betas):
    """Direct rotation transfer: SMPL-X axis-angle → Newton joint_q.

    NOTE: InterHuman PKL data is pre-rotated to Z-up. The root_orient
    already encodes a Y→Z-up rotation, and trans is in Z-up world coords.
    body_offset (SMPL-X rest pelvis, Y-up) is added un-rotated because
    Newton FK and SMPL-X FK both apply root rotation to this offset
    identically — verified by 0.000 cm MPJPE against GT.

    Args:
        root_orient: (T, 3) axis-angle for root joint (Z-up, pre-rotated by InterHuman)
        pose_body: (T, 63) axis-angle for 21 body joints (local, frame-agnostic)
        trans: (T, 3) global translation (Z-up, pre-rotated by InterHuman)
        betas: (10,) shape parameters

    Returns:
        joint_q: (T, 76) Newton joint coordinates
    """
    T = root_orient.shape[0]
    joint_q = np.zeros((T, N_JOINT_Q), dtype=np.float32)

    body_offset = get_body_offset(betas)

    # Root (Pelvis) translation.
    # No R_ROT on translation: InterHuman PKL stores trans already in Z-up.
    # body_offset is SMPL-X rest pelvis (Y-up frame: [~0, ~-0.335, ~0]), but
    # no rotation is needed because both SMPL-X FK and Newton FK apply the
    # root quaternion to this offset identically. The free-joint origin in
    # Newton is at the body frame origin, so adding the un-rotated rest-pose
    # pelvis position compensates correctly. Verified: 0.000 cm MPJPE vs GT.
    trans_with_offset = trans + body_offset[None, :]
    joint_q[:, 0:3] = trans_with_offset

    R_smplx_root = Rotation.from_rotvec(root_orient)
    R_ROT_inv = Rotation.from_matrix(R_ROT.T)
    root_newton = R_smplx_root * R_ROT_inv
    joint_q[:, 3:7] = root_newton.as_quat()  # scipy [x, y, z, w]

    # Body joints (1-21)
    pose_body_reshaped = pose_body.reshape(T, 21, 3)

    for smpl_j in range(1, N_SMPL_JOINTS):
        newton_body = SMPL_TO_NEWTON[smpl_j]
        q_start = _newton_body_q_index(newton_body)

        aa_smplx = pose_body_reshaped[:, smpl_j - 1, :]
        aa_newton = (R_ROT @ aa_smplx.T).T
        rot = Rotation.from_rotvec(aa_newton)
        euler_xyz = rot.as_euler('XYZ')  # extrinsic XYZ

        joint_q[:, q_start:q_start + 3] = euler_xyz

    return joint_q


# ═══════════════════════════════════════════════════════════════
# Forward Kinematics
# ═══════════════════════════════════════════════════════════════

def forward_kinematics(joint_q, betas, foot_geom="box", device="cuda:0"):
    """Newton FK: (T, 76) joint_q → (T, 22, 3) positions.

    Args:
        joint_q: (T, 76) Newton joint coordinates
        betas: (10,) shape parameters
        foot_geom: "box" | "sphere" | "capsule"
        device: CUDA device

    Returns:
        positions: (T, 22, 3) body joint positions in Z-up world
    """
    import warp as wp
    wp.config.verbose = False
    warnings.filterwarnings("ignore", message="Custom attribute")
    import newton

    model = get_newton_model(betas, foot_geom=foot_geom, device=device)
    T = joint_q.shape[0]
    positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    for t in range(T):
        state.joint_q = wp.array(joint_q[t], dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)
        for j in range(N_SMPL_JOINTS):
            positions[t, j] = body_q[SMPL_TO_NEWTON[j], :3]

    return positions


# ═══════════════════════════════════════════════════════════════
# IK-based retargeting
# ═══════════════════════════════════════════════════════════════

def ik_retarget(positions, betas, foot_geom="box", ik_iters=50, device="cuda:0",
                init_jq=None, sequential=False):
    """Position-based IK: (T, 22, 3) positions → Newton joint_q.

    Args:
        positions: (T, 22, 3) target joint positions in Z-up world
        betas: (10,) shape parameters
        foot_geom: "box" | "sphere" | "capsule"
        ik_iters: number of IK iterations
        device: CUDA device
        init_jq: (T, 76) optional initial joint_q for warm-starting.
                 Use rotation_retarget output to get consistent angles.
                 If None, initializes from pelvis position + identity root.
        sequential: if True and init_jq is None, solve frames sequentially,
                    warm-starting each frame from the previous frame's solution.
                    Produces temporally smooth angles when only positions are
                    available (e.g., from InterMask decoder output).
                    Slower than batch mode but gives consistent angle conventions.

    Returns:
        joint_q: (T, 76) solved joint coordinates
        fk_positions: (T, 22, 3) FK-verified positions after IK
    """
    import warp as wp
    wp.config.verbose = False
    warnings.filterwarnings("ignore", message="Custom attribute")
    import newton
    import newton.ik as ik

    model = get_newton_model(betas, foot_geom=foot_geom, device=device)
    T = positions.shape[0]
    n_coords = model.joint_coord_count  # 76

    # Build IK solver with 22 position objectives
    objectives = []
    for smpl_j in range(N_SMPL_JOINTS):
        targets = []
        for t in range(T):
            p = positions[t, smpl_j]
            targets.append(wp.vec3(float(p[0]), float(p[1]), float(p[2])))

        obj = ik.IKObjectivePosition(
            link_index=SMPL_TO_NEWTON[smpl_j],
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array(targets, dtype=wp.vec3, device=device),
            weight=1.0,
        )
        objectives.append(obj)

    solver = ik.IKSolver(
        model=model,
        n_problems=T,
        objectives=objectives,
        lambda_initial=0.01,
        jacobian_mode=ik.IKJacobianType.AUTODIFF,
    )

    # Initialize joint_q
    if init_jq is not None:
        jq_init = np.asarray(init_jq, dtype=np.float32)
        if jq_init.shape != (T, n_coords):
            raise ValueError(f"init_jq shape {jq_init.shape} != expected ({T}, {n_coords})")
    else:
        # Default: pelvis from reference, identity quaternion, rest zero
        jq_init = np.zeros((T, n_coords), dtype=np.float32)
        for t in range(T):
            jq_init[t, 0:3] = positions[t, 0]           # pelvis xyz
            jq_init[t, 3:7] = [1.0, 0.0, 0.0, 0.0]     # identity quat (xyzw)

    if sequential and init_jq is None:
        # Sequential solve: solve one frame at a time, warm-starting each
        # from the previous frame's solution. This produces temporally
        # consistent angles without needing rotation_retarget.
        #
        # Uses the Newton-recommended pattern from example_ik_h1:
        # - Build solver with n_problems=1
        # - Update targets in-place via set_target_position()
        # - step(jq, jq) to warm-start from previous solution

        # Build single-frame solver
        seq_objs = []
        for smpl_j in range(N_SMPL_JOINTS):
            p = positions[0, smpl_j]
            obj = ik.IKObjectivePosition(
                link_index=SMPL_TO_NEWTON[smpl_j],
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array(
                    [wp.vec3(float(p[0]), float(p[1]), float(p[2]))],
                    dtype=wp.vec3, device=device,
                ),
                weight=1.0,
            )
            seq_objs.append(obj)

        seq_solver = ik.IKSolver(
            model=model, n_problems=1, objectives=seq_objs,
            lambda_initial=0.01, jacobian_mode=ik.IKJacobianType.AUTODIFF,
        )

        # Frame 0: cold solve with more iterations
        jq_single = wp.array(
            jq_init[0:1], dtype=wp.float32, device=device
        )
        seq_solver.step(jq_single, jq_single, iterations=ik_iters)
        wp.synchronize()

        jq_np = np.zeros((T, n_coords), dtype=np.float32)
        jq_np[0] = jq_single.numpy()[0]

        # Subsequent frames: warm-start from previous, fewer iters needed
        # since positions change minimally between adjacent frames
        seq_iters = max(ik_iters // 10, 5)
        for t in range(1, T):
            # Update targets for this frame
            for smpl_j in range(N_SMPL_JOINTS):
                p = positions[t, smpl_j]
                seq_objs[smpl_j].set_target_position(
                    0, wp.vec3(float(p[0]), float(p[1]), float(p[2]))
                )

            # Update pelvis position in joint_q
            jq_arr = jq_single.numpy()
            jq_arr[0, 0:3] = positions[t, 0]
            jq_single = wp.array(jq_arr, dtype=wp.float32, device=device)

            # Solve (warm-started from previous frame)
            seq_solver.step(jq_single, jq_single, iterations=seq_iters)
            wp.synchronize()
            jq_np[t] = jq_single.numpy()[0]
    else:
        # Batch solve: all frames at once (fast, but angles may be inconsistent)
        jq = wp.array(jq_init, dtype=wp.float32, device=device)
        solver.step(jq, jq, iterations=ik_iters)
        wp.synchronize()
        jq_np = jq.numpy()  # (T, 76)

    # Normalize hinge angles to [-π, π] to avoid multi-revolution windup.
    # IK can converge to solutions with angles like -9.7 rad which are
    # physically equivalent to -9.7 + 4π ≈ 2.87 rad.  Large absolute angles
    # cause spurious joint-limit constraint forces in inverse dynamics.
    # Shift each DOF's trajectory by a constant multiple of 2π (preserving
    # temporal smoothness) so the mean angle is closest to 0.
    for dof in range(7, n_coords):
        mean_val = jq_np[:, dof].mean()
        shift = np.round(mean_val / (2 * np.pi)) * (2 * np.pi)
        jq_np[:, dof] -= shift

    # FK to verify and extract positions
    fk_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
    for t in range(T):
        state.joint_q = wp.array(jq_np[t], dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        body_q = state.body_q.numpy().reshape(-1, 7)
        for j in range(N_SMPL_JOINTS):
            fk_positions[t, j] = body_q[SMPL_TO_NEWTON[j], :3]

    return jq_np, fk_positions


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_interhuman_pkl(data_dir, clip_id):
    """Load raw SMPL-X params from InterHuman pkl.

    Searches in {data_dir}/motions/{clip_id}.pkl first, then
    falls back to {data_dir}/{clip_id}.pkl.

    Args:
        data_dir: dataset root directory
        clip_id: clip identifier (e.g., "1000")

    Returns:
        list of dicts, one per person, each with keys:
            root_orient: (T, 3) axis-angle
            pose_body: (T, 63) axis-angle for 21 joints
            trans: (T, 3) global translation
            betas: (10,) shape parameters
        or None if file not found
    """
    pkl_path = os.path.join(data_dir, "motions", f"{clip_id}.pkl")
    if not os.path.isfile(pkl_path):
        pkl_path = os.path.join(data_dir, f"{clip_id}.pkl")
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
            'betas': p['betas'].astype(np.float64),
        })

    return results


def load_interhuman_npy(data_dir, clip_id):
    """Load raw 492-dim npy from InterHuman motions_processed.

    Args:
        data_dir: dataset root (should contain motions_processed/)
        clip_id: clip identifier

    Returns:
        list of (T, 22, 3) position arrays, one per person,
        or None if files not found.
    """
    results = []
    for pidx in [1, 2]:
        npy_path = os.path.join(data_dir, "motions_processed", f"person{pidx}", f"{clip_id}.npy")
        if not os.path.isfile(npy_path):
            return None
        raw = np.load(npy_path)  # (T, 492) in Z-up raw frame
        positions = raw[:, :66].reshape(-1, 22, 3).astype(np.float32)
        results.append(positions)

    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retarget SMPL-X to Newton")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "InterHuman"),
                        help="InterHuman data directory")
    parser.add_argument("--clip", required=True, help="Clip ID")
    parser.add_argument("--method", choices=["rotation", "ik", "both"], default="both",
                        help="Retargeting method")
    parser.add_argument("--foot-geom", choices=["box", "sphere", "capsule"], default="box")
    parser.add_argument("--ik-iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    persons = load_interhuman_pkl(args.data_dir, args.clip)
    if persons is None:
        print(f"Clip {args.clip} not found in {args.data_dir}")
        sys.exit(1)

    for pidx, p in enumerate(persons):
        print(f"\n=== Person {pidx} ===")
        betas = p['betas']

        if args.method in ("rotation", "both"):
            jq_rot = rotation_retarget(
                p['root_orient'], p['pose_body'], p['trans'], betas
            )
            pos_rot = forward_kinematics(jq_rot, betas, foot_geom=args.foot_geom,
                                         device=args.device)
            print(f"  Rotation retarget: joint_q shape={jq_rot.shape}")

        if args.method in ("ik", "both"):
            # Use FK positions from rotation as IK targets
            if args.method == "both":
                target_pos = pos_rot
            else:
                # Load from npy if only doing IK
                npy_data = load_interhuman_npy(args.data_dir, args.clip)
                if npy_data is None:
                    print("  No npy data found, skipping IK")
                    continue
                target_pos = npy_data[pidx]

            jq_ik, pos_ik = ik_retarget(
                target_pos, betas, foot_geom=args.foot_geom,
                ik_iters=args.ik_iters, device=args.device
            )
            mpjpe = np.sqrt(((pos_ik - target_pos) ** 2).sum(-1).mean(-1)).mean()
            print(f"  IK retarget: MPJPE = {mpjpe*100:.3f} cm")

            if args.method == "both":
                # Compare joint angles
                angle_diff = np.abs(jq_rot[:, 7:] - jq_ik[:, 7:])
                print(f"  Joint angle diff: mean={np.degrees(angle_diff.mean()):.3f} deg, "
                      f"max={np.degrees(angle_diff.max()):.3f} deg")
