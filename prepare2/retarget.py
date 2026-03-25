"""
retarget.py — Per-subject rotation retargeting: SMPL-X → Newton.

For each clip, generates a per-subject SMPL XML (matching betas),
then directly maps axis-angle rotations to Newton joint angles.

Since the Newton skeleton has the same proportions as the source
SMPL-X body, position error is near-zero (only pose blend shape
effects remain, typically < 2cm).

Pipeline per clip:
  1. Load raw SMPL-X params (root_orient, pose_body, trans, betas)
  2. Generate per-subject smpl.xml using gen_smpl_xml.py
  3. Build Newton model from that XML
  4. Convert axis-angle → Newton joint_q (using R_ROT for rotations,
     direct position mapping, plus per-subject body offset)
  5. Run FK to get positions
  6. Save joint_q and/or positions

Usage:
    # Single clip (InterHuman)
    python prepare2/retarget.py --dataset interhuman --clip 1000

    # Batch
    python prepare2/retarget.py --dataset interhuman

    # With evaluation against SMPL-X FK
    python prepare2/retarget.py --dataset interhuman --clip 1000 --eval
"""
import os
import sys
import time
import tempfile
import hashlib
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.gen_smpl_xml import generate_smpl_xml, get_smplx_body_offset, R_ROT

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


def newton_body_q_index(body_idx):
    """Get the starting index in joint_q for a non-root Newton body."""
    assert body_idx >= 1
    return 7 + (body_idx - 1) * 3


# ═══════════════════════════════════════════════════════════════
# Per-subject XML cache
# ═══════════════════════════════════════════════════════════════

_xml_cache_dir = None
_body_offset_cache = {}


def _betas_hash(betas):
    """Stable hash of betas for caching."""
    return hashlib.md5(betas.tobytes()).hexdigest()[:12]


def get_or_create_xml(betas, cache_dir=None):
    """
    Get (or generate and cache) the per-subject XML.

    Args:
        betas: (10,) numpy array
        cache_dir: directory to cache XMLs

    Returns:
        xml_path: path to the generated XML
    """
    global _xml_cache_dir
    if cache_dir is None:
        if _xml_cache_dir is None:
            _xml_cache_dir = os.path.join(PROJECT_ROOT, "prepare2", "xml_cache")
        cache_dir = _xml_cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    h = _betas_hash(betas)
    xml_path = os.path.join(cache_dir, f"smpl_{h}.xml")

    if not os.path.exists(xml_path):
        generate_smpl_xml(betas, output_path=xml_path)

    return xml_path


def get_body_offset(betas):
    """Get cached SMPL-X body offset for given betas."""
    h = _betas_hash(betas)
    if h not in _body_offset_cache:
        _body_offset_cache[h] = get_smplx_body_offset(betas)
    return _body_offset_cache[h]


# ═══════════════════════════════════════════════════════════════
# Newton model cache (per unique betas)
# ═══════════════════════════════════════════════════════════════

_model_cache = {}


def get_newton_model(betas, device="cuda:0"):
    """Build or retrieve cached Newton model for given betas."""
    h = _betas_hash(betas)
    if h not in _model_cache:
        xml_path = get_or_create_xml(betas)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        _model_cache[h] = builder.finalize(device=device)
    return _model_cache[h]


# ═══════════════════════════════════════════════════════════════
# Rotation transfer
# ═══════════════════════════════════════════════════════════════

def smplx_to_joint_q(root_orient, pose_body, trans, betas):
    """
    Convert SMPL-X parameters to Newton joint_q.

    InterHuman data is in Z-up world frame (root_orient already encodes the
    SMPL-X Y-up → Z-up rotation). With Newton up_axis=Z, positions and
    orientations map directly — no coordinate rotation needed.

    Uses per-subject body offset and coordinate transforms:
      - R_ROT for body-joint rotation conjugation (axis-angle)
      - Direct position mapping (both InterHuman and Newton are Z-up)

    Args:
        root_orient: (T, 3) axis-angle for root joint (in Z-up world)
        pose_body: (T, 63) axis-angle for 21 body joints
        trans: (T, 3) global translation (in Z-up world)
        betas: (10,) shape parameters

    Returns:
        joint_q: (T, 76) Newton joint coordinates
    """
    T = root_orient.shape[0]
    joint_q = np.zeros((T, N_JOINT_Q), dtype=np.float32)

    # Per-subject body offset (rest-pose pelvis position in SMPL-X frame)
    body_offset = get_body_offset(betas)

    # ── Root (Pelvis) ──────────────────────────────────────────
    # Position: direct sum — both trans and body_offset produce the
    # pelvis world position as output by the SMPL-X body model.
    # With up_axis=Z the Newton world frame matches InterHuman's Z-up.
    trans_with_offset = trans + body_offset[None, :]
    joint_q[:, 0:3] = trans_with_offset

    # Orientation: R_newton_root = R(root_orient) @ R_ROT^T
    #   R(root_orient) maps SMPL-X body-local → Z-up world
    #   R_ROT maps SMPL-X body-local → Newton body-local
    #   So R_newton_root = R(root_orient) @ R_ROT^{-1}
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

        aa_smplx = pose_body_reshaped[:, smpl_j - 1, :]
        aa_newton = (R_ROT @ aa_smplx.T).T
        rot = Rotation.from_rotvec(aa_newton)
        # Newton D6 joint composes as: rot = Rz(θ2) @ Ry(θ1) @ Rx(θ0)
        # = extrinsic XYZ (scipy uppercase 'XYZ')
        euler_xyz = rot.as_euler('XYZ')

        joint_q[:, q_start:q_start + 3] = euler_xyz

    return joint_q


def extract_positions_from_fk(model, joint_q, device="cuda:0"):
    """Run Newton FK on joint_q to get body positions."""
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
# Dataset loading
# ═══════════════════════════════════════════════════════════════

def load_interhuman_clip(data_dir, clip_id):
    """Load raw SMPL-X params from InterHuman pkl (includes betas).

    Searches in {data_dir}/motions/{clip_id}.pkl first, then falls back to
    {data_dir}/{clip_id}.pkl (for generated data from generate_and_save.py).
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
            'betas': p['betas'].astype(np.float64),  # (10,)
        })

    return results


def list_interhuman_clips(data_dir):
    """List all available InterHuman clip IDs."""
    motion_dir = os.path.join(data_dir, "motions")
    clips = []
    # Check {data_dir}/motions/ first (standard InterHuman layout)
    if os.path.isdir(motion_dir):
        for f in sorted(os.listdir(motion_dir)):
            if f.endswith('.pkl'):
                clips.append(os.path.splitext(f)[0])
    # Fall back to {data_dir}/ directly (generated data)
    if not clips:
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.pkl'):
                clips.append(os.path.splitext(f)[0])
    return clips


def load_interx_clip(data_dir, clip_id):
    """Load SMPL-X params from Inter-X H5 file (no per-subject betas).

    InterX data is in Y-up world frame. We convert to Z-up (InterHuman
    convention) so that the retarget pipeline's Z-up formulas apply correctly.

    Searches for H5 files at:
        {data_dir}/processed/motions/inter-x.h5  (standard Inter-X layout)
        {data_dir}/processed/inter-x.h5
        {data_dir}/generated.h5                   (generated data)
    """
    import h5py

    h5_candidates = [
        os.path.join(data_dir, "processed", "motions", "inter-x.h5"),
        os.path.join(data_dir, "processed", "inter-x.h5"),
        os.path.join(data_dir, "generated.h5"),
    ]
    h5_path = None
    for cand in h5_candidates:
        if os.path.isfile(cand):
            h5_path = cand
            break
    if h5_path is None:
        return None

    with h5py.File(h5_path, 'r') as mf:
        if clip_id not in mf:
            return None
        motion = mf[clip_id][:].astype(np.float64)

    T = motion.shape[0]
    results = []

    # Y-up → Z-up: Rx(90°) = [[1,0,0],[0,0,-1],[0,1,0]]
    R_yup_to_zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    R_conv = Rotation.from_matrix(R_yup_to_zup)

    for pidx in range(2):
        p = motion[:, :, pidx * 3:(pidx + 1) * 3]
        root_orient = p[:, 0, :]
        pose_body = p[:, 1:22, :].reshape(T, 63)
        trans = p[:, 55, :]

        # Convert world frame Y-up → Z-up
        trans = (R_yup_to_zup @ trans.T).T
        root_orient = (R_conv * Rotation.from_rotvec(root_orient)).as_rotvec()

        results.append({
            'root_orient': root_orient,
            'pose_body': pose_body,
            'trans': trans,
            'betas': np.zeros(10, dtype=np.float64),  # Inter-X: neutral betas
        })

    return results


def list_interx_clips(data_dir):
    """List all available Inter-X clip IDs."""
    import h5py
    h5_candidates = [
        os.path.join(data_dir, "processed", "motions", "inter-x.h5"),
        os.path.join(data_dir, "processed", "inter-x.h5"),
        os.path.join(data_dir, "generated.h5"),
    ]
    for h5_path in h5_candidates:
        if os.path.isfile(h5_path):
            with h5py.File(h5_path, 'r') as mf:
                return sorted(mf.keys())
    return []


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_clip(persons_data, results):
    """
    Compare Newton FK positions against SMPL-X FK positions.

    Returns per-joint MPJPE in meters.
    """
    import torch
    body_model_dir = os.path.join(PROJECT_ROOT, "data", "body_model")
    if body_model_dir not in sys.path:
        sys.path.insert(0, body_model_dir)
    from body_model import BodyModel

    bm = BodyModel(
        os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
        num_betas=10,
    )

    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
        'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    ]

    all_errors = []
    for pidx, (p, res) in enumerate(zip(persons_data, results)):
        T = p['root_orient'].shape[0]
        betas_t = torch.tensor(p['betas'], dtype=torch.float32).unsqueeze(0).expand(T, -1)

        with torch.no_grad():
            out = bm(
                root_orient=torch.tensor(p['root_orient'], dtype=torch.float32),
                pose_body=torch.tensor(p['pose_body'], dtype=torch.float32),
                betas=betas_t,
                trans=torch.tensor(p['trans'], dtype=torch.float32),
            )
            smplx_pos = out.Jtr[:, :22].numpy()  # (T, 22, 3)

        # Both Newton FK and SMPL-X FK positions are in Z-up world frame
        # — compare directly (no coordinate transform needed)
        errors = np.linalg.norm(res['positions'] - smplx_pos, axis=-1)  # (T, 22)
        all_errors.append(errors)

        mean_per_joint = errors.mean(axis=0) * 100
        print(f"\n  Person {pidx+1}: Mean MPJPE = {errors.mean()*100:.2f} cm, "
              f"Max = {errors.max()*100:.2f} cm")
        for j in range(22):
            print(f"    {joint_names[j]:12s}: {mean_per_joint[j]:6.2f} cm")

    return all_errors


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def process_clip(persons_data, device="cuda:0"):
    """
    Process one clip: per-subject XML → rotation transfer → FK.

    Returns list of dicts with 'joint_q' and 'positions'.
    """
    results = []
    for p in persons_data:
        betas = p['betas']

        # Get per-subject Newton model
        model = get_newton_model(betas, device=device)

        # Convert rotations
        joint_q = smplx_to_joint_q(
            p['root_orient'], p['pose_body'], p['trans'], betas
        )

        # FK to positions
        positions = extract_positions_from_fk(model, joint_q, device=device)

        results.append({
            'joint_q': joint_q,
            'positions': positions,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Per-subject rotation retarget: SMPL-X → Newton"
    )
    parser.add_argument("--dataset", choices=["interhuman", "interx"], required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--clip", default=None, help="Single clip ID")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate MPJPE against SMPL-X FK")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete xml_cache/ before running (forces XML regeneration)")
    args = parser.parse_args()

    # Clear XML cache if requested
    if args.clear_cache:
        import shutil
        cache_dir = os.path.join(PROJECT_ROOT, "prepare2", "xml_cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared XML cache: {cache_dir}")
        # Also clear in-memory caches
        _model_cache.clear()
        _body_offset_cache.clear()

    if args.data_dir is None:
        args.data_dir = ("data/InterHuman" if args.dataset == "interhuman"
                         else "data/Inter-X_Dataset")
    if args.output_dir is None:
        args.output_dir = f"data/retargeted_v2/{args.dataset}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load / list clips
    if args.dataset == "interhuman":
        load_fn = lambda cid: load_interhuman_clip(args.data_dir, cid)
        clips = [args.clip] if args.clip else list_interhuman_clips(args.data_dir)
    else:
        load_fn = lambda cid: load_interx_clip(args.data_dir, cid)
        clips = [args.clip] if args.clip else list_interx_clips(args.data_dir)

    print(f"Processing {len(clips)} clips...")
    total_time = 0
    processed = 0
    skipped = 0

    for clip_id in tqdm(clips, desc="Retarget (per-subject)"):
        persons_data = load_fn(clip_id)
        if persons_data is None:
            skipped += 1
            continue

        t0 = time.time()
        results = process_clip(persons_data, device=args.gpu)
        dt = time.time() - t0
        total_time += dt

        # Save positions, joint_q, betas, and xml path
        for pidx, res in enumerate(results):
            out_name = f"{clip_id}_person{pidx}"
            np.save(os.path.join(args.output_dir, f"{out_name}.npy"),
                    res['positions'])
            np.save(os.path.join(args.output_dir, f"{out_name}_joint_q.npy"),
                    res['joint_q'])
            np.save(os.path.join(args.output_dir, f"{out_name}_betas.npy"),
                    persons_data[pidx]['betas'])

        # Evaluate if requested
        if args.eval:
            print(f"\n--- Clip {clip_id} ---")
            evaluate_clip(persons_data, results)

        processed += 1
        if processed == 1 or processed % 100 == 0:
            T = results[0]['positions'].shape[0]
            print(f"  Clip {clip_id}: {T} frames, {dt:.3f}s")

    print(f"\nDone: {processed} clips, {skipped} skipped")
    if processed > 0:
        print(f"Total: {total_time:.1f}s, avg: {total_time/processed:.3f}s/clip")
    print(f"Output: {args.output_dir}")
    print(f"XML cache: {_xml_cache_dir}")


if __name__ == "__main__":
    main()
