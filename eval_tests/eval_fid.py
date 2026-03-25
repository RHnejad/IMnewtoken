#!/usr/bin/env python3
"""
FID Evaluation for Motion Retargeting and Simulation Quality.

Computes Fréchet Inception Distance (FID) using InterMask's InterCLIP evaluator
with identical settings (emb_scale=6, interhuman dataset).

Cases:
  1) GT FID:       GT distribution vs itself → sanity check (should be ~0)
  2) Retarget FID: retargeted positions vs GT → measures retarget quality
  3) Solo-torque FID: torque-simulated positions vs GT → measures simulation quality

All cases process motions through a unified pipeline:
  - Extract 22-joint positions (T, 22, 3)
  - Apply InterHuman coordinate transform (Y↔Z swap, floor, origin, facing Z+)
  - Compute velocities, foot contacts
  - Rotations set to population mean (zeros after normalization) for fair comparison
  - Normalize with InterHuman global_mean / global_std
  - Embed via InterCLIP evaluator
  - Compute FID between test and reference distributions

Usage:
  cd /path/to/InterMask
  python eval_tests/eval_fid.py --case 1          # GT self-FID
  python eval_tests/eval_fid.py --case 2          # retarget FID
  python eval_tests/eval_fid.py --case 3          # solo-torque FID
  python eval_tests/eval_fid.py --case all        # all cases
  python eval_tests/eval_fid.py --case 1 2        # specific cases
"""

import argparse
import copy
import os
import sys
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict

# ── Project root setup ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data.utils import MotionNormalizer, process_motion_np, load_motion
from data.quaternion import qmul_np, qinv_np, qrot_np, qbetween_np
from utils.metrics import calculate_frechet_distance, calculate_activation_statistics
from utils.get_opt import get_opt

# ── Constants matching InterMask eval.py (interhuman) ───────────
EMB_SCALE = 6
N_JOINTS = 22
MAX_GT_LENGTH = 300      # max frames per motion (30fps)
MIN_LENGTH = 15          # minimum motion length
FPS_GT = 30              # InterHuman ground truth FPS
FPS_RETARGET = 60        # retargeted data FPS (2x GT)
DOWNSAMPLE_FACTOR = 2    # FPS_RETARGET / FPS_GT

face_joint_indx = [2, 1, 17, 16]
fid_l = [7, 10]
fid_r = [8, 11]

trans_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  Feature Computation (positions → InterHuman 262-dim features)
# ═══════════════════════════════════════════════════════════════

def positions_to_features(positions, feet_thre=0.001):
    """
    Convert (T, 22, 3) positions → (T-1, 262) InterHuman features.

    Applies the same coordinate transforms as process_motion_np:
    - Y↔Z swap via trans_matrix
    - Put on floor
    - XZ at origin
    - Face Z+ direction
    - Compute velocities and foot contacts
    - Rotations filled with zeros (= population mean after normalization)

    Returns:
        features: (T-1, 262) array
        root_quat_init: (1, 4) initial root quaternion
        root_pos_init_xz: (1, 3) initial root XZ position
    """
    T = positions.shape[0]

    # Coordinate transform (Y↔Z swap)
    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    # Put on floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # XZ at origin
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # Face Z+ direction
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init_for_all, positions)

    # Foot contacts
    def foot_detect(pos, thres):
        velfactor = np.array([thres, thres])
        heightfactor = np.array([0.12, 0.05])

        feet_l_x = (pos[1:, fid_l, 0] - pos[:-1, fid_l, 0]) ** 2
        feet_l_y = (pos[1:, fid_l, 1] - pos[:-1, fid_l, 1]) ** 2
        feet_l_z = (pos[1:, fid_l, 2] - pos[:-1, fid_l, 2]) ** 2
        feet_l_h = pos[:-1, fid_l, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (pos[1:, fid_r, 0] - pos[:-1, fid_r, 0]) ** 2
        feet_r_y = (pos[1:, fid_r, 1] - pos[:-1, fid_r, 1]) ** 2
        feet_r_z = (pos[1:, fid_r, 2] - pos[:-1, fid_r, 2]) ** 2
        feet_r_h = pos[:-1, fid_r, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Joint positions (T-1, 66)
    joint_positions = positions[:-1].reshape(T - 1, -1)

    # Joint velocities (T-1, 66)
    joint_vels = (positions[1:] - positions[:-1]).reshape(T - 1, -1)

    # Rotations: zeros (126 dims = 21 joints × 6D rotation)
    # After normalization, zeros → population mean → neutral/uninformative
    rot_data = np.zeros((T - 1, 21 * 6), dtype=np.float32)

    # Concatenate: pos(66) + vel(66) + rot(126) + feet(4) = 262
    data = np.concatenate([joint_positions, joint_vels, rot_data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]


def rigid_transform_positions(relative, positions):
    """
    Apply rigid transform to positions (T, 22, 3) given relative pose.
    Mirrors data.utils.rigid_transform but operates on raw positions.
    """
    relative_rot = relative[0]
    relative_t = relative[1:3]

    T = positions.shape[0]
    relative_r_rot_quat = np.zeros((T, N_JOINTS, 4))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)

    positions = qrot_np(qinv_np(relative_r_rot_quat), positions)
    positions[..., [0, 2]] += relative_t

    return positions


def process_position_pair(pos1, pos2, normalizer):
    """
    Process a pair of position arrays (T, 22, 3) into normalized features.

    Returns:
        feat1: (L, 262) normalized features for person 1
        feat2: (L, 262) normalized features for person 2
        length: actual feature length (before padding)
    """
    # Process person 1: get features and initial transform
    feat1, root_quat_init1, root_pos_init1 = positions_to_features(pos1)

    # Process person 2 with relative transform
    feat2_raw, root_quat_init2, root_pos_init2 = positions_to_features(pos2)

    # Compute relative transform (person2 relative to person1)
    r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
    angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

    xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
    relative = np.concatenate([angle, xz], axis=-1)[0]

    # Re-process person 2 with relative transform applied to positions
    # (matches InterHumanDataset processing)
    pos2_txf = pos2.copy()
    # Apply the coord transform first
    pos2_txf_t = np.einsum("mn, tjn->tjm", trans_matrix, pos2_txf)
    floor_height2 = pos2_txf_t.min(axis=0).min(axis=0)[1]
    pos2_txf_t[:, :, 1] -= floor_height2
    root_pos_init2_t = pos2_txf_t[0]
    root_pose_init_xz2 = root_pos_init2_t[0] * np.array([1, 0, 1])
    pos2_txf_t = pos2_txf_t - root_pose_init_xz2

    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across2 = root_pos_init2_t[r_hip] - root_pos_init2_t[l_hip]
    across2 = across2 / np.sqrt((across2 ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init2 = np.cross(np.array([[0, 1, 0]]), across2, axis=-1)
    forward_init2 = forward_init2 / np.sqrt((forward_init2 ** 2).sum(axis=-1))[..., np.newaxis]
    target2 = np.array([[0, 0, 1]])
    rqi2 = qbetween_np(forward_init2, target2)
    rqi2_all = np.ones(pos2_txf_t.shape[:-1] + (4,)) * rqi2
    pos2_txf_t = qrot_np(rqi2_all, pos2_txf_t)

    # Now apply the rigid_transform to the processed features
    # The features after positions_to_features contain pos(66)+vel(66)+rot(126)+feet(4)
    # We need to apply rigid_transform to positions and velocities
    # Rebuild features for person2 using the rigid transform logic
    T2 = pos2_txf_t.shape[0]
    relative_rot = relative[0]
    relative_t_vec = relative[1:3]

    # Apply rotation and translation to positions
    rq = np.zeros((T2, N_JOINTS, 4))
    rq[..., 0] = np.cos(relative_rot)
    rq[..., 2] = np.sin(relative_rot)
    pos2_final = qrot_np(qinv_np(rq), pos2_txf_t)
    pos2_final[..., [0, 2]] += relative_t_vec

    # Compute features from transformed positions
    T = pos2_final.shape[0]
    joint_pos2 = pos2_final[:-1].reshape(T - 1, -1)
    joint_vel2 = (pos2_final[1:] - pos2_final[:-1]).reshape(T - 1, -1)
    rot2 = np.zeros((T - 1, 21 * 6), dtype=np.float32)

    # Foot contacts
    def foot_detect(pos, thres=0.001):
        vf = np.array([thres, thres])
        hf = np.array([0.12, 0.05])
        fl_x = (pos[1:, fid_l, 0] - pos[:-1, fid_l, 0]) ** 2
        fl_y = (pos[1:, fid_l, 1] - pos[:-1, fid_l, 1]) ** 2
        fl_z = (pos[1:, fid_l, 2] - pos[:-1, fid_l, 2]) ** 2
        fl_h = pos[:-1, fid_l, 1]
        fl = (((fl_x + fl_y + fl_z) < vf) & (fl_h < hf)).astype(np.float32)
        fr_x = (pos[1:, fid_r, 0] - pos[:-1, fid_r, 0]) ** 2
        fr_y = (pos[1:, fid_r, 1] - pos[:-1, fid_r, 1]) ** 2
        fr_z = (pos[1:, fid_r, 2] - pos[:-1, fid_r, 2]) ** 2
        fr_h = pos[:-1, fid_r, 1]
        fr = (((fr_x + fr_y + fr_z) < vf) & (fr_h < hf)).astype(np.float32)
        return fl, fr

    feet_l2, feet_r2 = foot_detect(pos2_final)
    feat2 = np.concatenate([joint_pos2, joint_vel2, rot2, feet_l2, feet_r2], axis=-1)

    # Normalize
    feat1 = normalizer.forward(feat1)
    feat2 = normalizer.forward(feat2)

    # Match lengths (take minimum)
    L = min(feat1.shape[0], feat2.shape[0])
    feat1 = feat1[:L]
    feat2 = feat2[:L]

    # Random person swap (50% chance, matching InterHumanDataset)
    if np.random.rand() > 0.5:
        feat1, feat2 = feat2, feat1

    return feat1, feat2, L


# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_gt_positions(data_root, clip_id):
    """
    Load GT positions from InterHuman motions_processed.
    Returns (positions_person1, positions_person2), each (T, 22, 3).
    """
    p1_path = os.path.join(data_root, "motions_processed", "person1", f"{clip_id}.npy")
    p2_path = os.path.join(data_root, "motions_processed", "person2", f"{clip_id}.npy")

    raw1 = np.load(p1_path).astype(np.float32)
    raw2 = np.load(p2_path).astype(np.float32)

    # Extract 22-joint positions (first 66 features)
    pos1 = raw1[:, :N_JOINTS * 3].reshape(-1, N_JOINTS, 3)
    pos2 = raw2[:, :N_JOINTS * 3].reshape(-1, N_JOINTS, 3)

    return pos1, pos2


def load_retarget_positions(retarget_root, clip_id):
    """
    Load retargeted positions. Shape (T, 22, 3) at 60fps.
    Downsampled to 30fps to match GT.
    """
    p0_path = os.path.join(retarget_root, f"{clip_id}_person0.npy")
    p1_path = os.path.join(retarget_root, f"{clip_id}_person1.npy")

    pos0 = np.load(p0_path).astype(np.float32)  # (T_60fps, 22, 3)
    pos1 = np.load(p1_path).astype(np.float32)

    # Downsample from 60fps to 30fps
    pos0 = pos0[::DOWNSAMPLE_FACTOR]
    pos1 = pos1[::DOWNSAMPLE_FACTOR]

    return pos0, pos1


def load_sim_positions(retarget_root, clip_id, sim_type="two_person"):
    """
    Load simulation positions if available.
    Returns (pos_person0, pos_person1) each (T, 22, 3) at 60fps,
    downsampled to 30fps.
    """
    if sim_type == "two_person":
        path = os.path.join(retarget_root, f"{clip_id}_two_person_sim_positions.npy")
        if not os.path.exists(path):
            return None, None
        data = np.load(path).astype(np.float32)  # (T, 2, 22, 3)
        pos0 = data[::DOWNSAMPLE_FACTOR, 0]  # (T/2, 22, 3)
        pos1 = data[::DOWNSAMPLE_FACTOR, 1]
        return pos0, pos1
    else:
        # Future: solo simulation positions
        return None, None


def get_test_clip_ids(data_root):
    """Load test split clip IDs."""
    test_path = os.path.join(data_root, "split", "test.txt")
    with open(test_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ═══════════════════════════════════════════════════════════════
#  Embedding Collection
# ═══════════════════════════════════════════════════════════════

def collect_embeddings(eval_wrapper, feat_pairs, batch_size, device):
    """
    Collect motion embeddings from processed feature pairs.

    Args:
        eval_wrapper: InterCLIP EvaluatorModelWrapper
        feat_pairs: list of (feat1, feat2, length) tuples
        batch_size: batch size for evaluation
        device: torch device

    Returns:
        embeddings: (N, embed_dim) numpy array
    """
    all_embeddings = []
    N = len(feat_pairs)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_feats = feat_pairs[start:end]
        B = len(batch_feats)

        # Pad all to MAX_GT_LENGTH
        m1_list, m2_list, lens_list = [], [], []
        for f1, f2, L in batch_feats:
            if L < MAX_GT_LENGTH:
                pad = np.zeros((MAX_GT_LENGTH - L, 262), dtype=np.float32)
                f1 = np.concatenate([f1, pad], axis=0)
                f2 = np.concatenate([f2, pad], axis=0)
            elif L > MAX_GT_LENGTH:
                # Random crop
                idx = random.randint(0, L - MAX_GT_LENGTH)
                f1 = f1[idx:idx + MAX_GT_LENGTH]
                f2 = f2[idx:idx + MAX_GT_LENGTH]
                L = MAX_GT_LENGTH
            m1_list.append(f1)
            m2_list.append(f2)
            lens_list.append(L)

        m1 = torch.tensor(np.array(m1_list), dtype=torch.float32)
        m2 = torch.tensor(np.array(m2_list), dtype=torch.float32)
        lens = torch.tensor(lens_list)

        batch_data = ("eval", [""] * B, m1, m2, lens)

        with torch.no_grad():
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch_data)
            all_embeddings.append(motion_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ═══════════════════════════════════════════════════════════════
#  FID Computation
# ═══════════════════════════════════════════════════════════════

def compute_fid(embeddings_ref, embeddings_test):
    """Compute FID between reference and test embedding distributions."""
    embed_dim = embeddings_ref.shape[1]
    min_samples = embed_dim + 1  # need > dim samples for non-degenerate cov
    if len(embeddings_ref) < min_samples or len(embeddings_test) < min_samples:
        print(f"\n  WARNING: Too few samples for reliable FID "
              f"(ref={len(embeddings_ref)}, test={len(embeddings_test)}, "
              f"need ≥{min_samples})")
        print(f"  Run batch_sim_solo.py to generate more simulation data first.")
        return float('nan')
    mu_ref, cov_ref = calculate_activation_statistics(embeddings_ref, EMB_SCALE)
    mu_test, cov_test = calculate_activation_statistics(embeddings_test, EMB_SCALE)
    fid = calculate_frechet_distance(mu_ref, cov_ref, mu_test, cov_test)
    return fid


# ═══════════════════════════════════════════════════════════════
#  Main Evaluation Cases
# ═══════════════════════════════════════════════════════════════

def run_case_1(eval_wrapper, data_root, batch_size, device):
    """
    Case 1: GT FID — split GT into two halves, compute FID between them.
    Sanity check: should be close to 0 (only sampling variance).
    """
    print("\n" + "=" * 60)
    print("Case 1: GT Self-FID (sanity check)")
    print("=" * 60)

    normalizer = MotionNormalizer()
    clip_ids = get_test_clip_ids(data_root)
    random.shuffle(clip_ids)

    print(f"Processing {len(clip_ids)} GT test clips...")
    feat_pairs = []
    skipped = 0

    for cid in tqdm(clip_ids, desc="GT positions"):
        try:
            pos1, pos2 = load_gt_positions(data_root, cid)
            if pos1.shape[0] < MIN_LENGTH:
                skipped += 1
                continue
            f1, f2, L = process_position_pair(pos1, pos2, normalizer)
            feat_pairs.append((f1, f2, L))
        except Exception as e:
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} clips (too short or errors)")

    print(f"Collecting embeddings for {len(feat_pairs)} motions...")
    all_embeddings = collect_embeddings(eval_wrapper, feat_pairs, batch_size, device)

    # Split in half
    N = len(all_embeddings)
    half = N // 2
    emb_a = all_embeddings[:half]
    emb_b = all_embeddings[half:2 * half]

    fid = compute_fid(emb_a, emb_b)
    print(f"\n>>> Case 1 — GT Self-FID: {fid:.4f}")
    print(f"    (Split {N} samples into 2 × {half})")
    return fid


def run_case_2(eval_wrapper, data_root, retarget_root, batch_size, device):
    """
    Case 2: GT after retarget FID.
    Compares retargeted positions against GT positions.
    """
    print("\n" + "=" * 60)
    print("Case 2: Retarget FID")
    print("=" * 60)

    normalizer = MotionNormalizer()
    clip_ids = get_test_clip_ids(data_root)

    print(f"Processing {len(clip_ids)} clips...")
    gt_feats = []
    rt_feats = []
    skipped = 0

    for cid in tqdm(clip_ids, desc="GT + Retarget"):
        try:
            # GT positions
            gt_pos1, gt_pos2 = load_gt_positions(data_root, cid)
            if gt_pos1.shape[0] < MIN_LENGTH:
                skipped += 1
                continue

            # Retargeted positions
            rt_pos0, rt_pos1 = load_retarget_positions(retarget_root, cid)
            if rt_pos0.shape[0] < MIN_LENGTH:
                skipped += 1
                continue

            # Process both pairs
            gt_f1, gt_f2, gt_L = process_position_pair(gt_pos1, gt_pos2, normalizer)
            rt_f1, rt_f2, rt_L = process_position_pair(rt_pos0, rt_pos1, normalizer)

            gt_feats.append((gt_f1, gt_f2, gt_L))
            rt_feats.append((rt_f1, rt_f2, rt_L))

        except FileNotFoundError:
            skipped += 1
            continue
        except Exception as e:
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} clips")

    print(f"Collecting GT embeddings ({len(gt_feats)} motions)...")
    gt_embeddings = collect_embeddings(eval_wrapper, gt_feats, batch_size, device)

    print(f"Collecting retarget embeddings ({len(rt_feats)} motions)...")
    rt_embeddings = collect_embeddings(eval_wrapper, rt_feats, batch_size, device)

    fid = compute_fid(gt_embeddings, rt_embeddings)
    print(f"\n>>> Case 2 — Retarget FID: {fid:.4f}")
    print(f"    (GT: {len(gt_embeddings)} samples, Retarget: {len(rt_embeddings)} samples)")
    return fid


def run_case_3(eval_wrapper, data_root, retarget_root, batch_size, device):
    """
    Case 3: GT torques single person FID.
    Compares torque-simulated positions against GT positions.
    Requires pre-computed simulation positions as
    {clip_id}_two_person_sim_positions.npy or similar.
    """
    print("\n" + "=" * 60)
    print("Case 3: Solo-Torque Simulation FID")
    print("=" * 60)

    normalizer = MotionNormalizer()
    clip_ids = get_test_clip_ids(data_root)

    print(f"Scanning {len(clip_ids)} clips for simulation data...")
    gt_feats = []
    sim_feats = []
    found = 0

    for cid in tqdm(clip_ids, desc="GT + Sim"):
        try:
            # Check if simulation data exists
            sim_pos0, sim_pos1 = load_sim_positions(retarget_root, cid)
            if sim_pos0 is None:
                continue

            # GT positions
            gt_pos1, gt_pos2 = load_gt_positions(data_root, cid)
            if gt_pos1.shape[0] < MIN_LENGTH:
                continue

            if sim_pos0.shape[0] < MIN_LENGTH:
                continue

            # Process both pairs
            gt_f1, gt_f2, gt_L = process_position_pair(gt_pos1, gt_pos2, normalizer)
            sim_f1, sim_f2, sim_L = process_position_pair(sim_pos0, sim_pos1, normalizer)

            gt_feats.append((gt_f1, gt_f2, gt_L))
            sim_feats.append((sim_f1, sim_f2, sim_L))
            found += 1

        except Exception as e:
            continue

    if found == 0:
        print("\nERROR: No simulation position data found!")
        print(f"Expected files: {{clip_id}}_two_person_sim_positions.npy")
        print(f"in directory: {retarget_root}")
        print("\nRun simulation first with optimize_interaction.py to generate positions,")
        print("or compute torques and save simulation positions.")
        return None

    print(f"\nFound simulation data for {found} clips")
    print(f"Collecting GT embeddings ({len(gt_feats)} motions)...")
    gt_embeddings = collect_embeddings(eval_wrapper, gt_feats, batch_size, device)

    print(f"Collecting simulation embeddings ({len(sim_feats)} motions)...")
    sim_embeddings = collect_embeddings(eval_wrapper, sim_feats, batch_size, device)

    fid = compute_fid(gt_embeddings, sim_embeddings)
    print(f"\n>>> Case 3 — Solo-Torque Simulation FID: {fid:.4f}")
    print(f"    (GT: {len(gt_embeddings)} samples, Sim: {len(sim_embeddings)} samples)")
    return fid


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def build_evaluator(device):
    """Load InterCLIP evaluator model (identical to InterMask eval.py)."""
    from models.evaluator.evaluator import EvaluatorModelWrapper
    evalmodel_cfg = get_opt(
        os.path.join(PROJECT_ROOT, "checkpoints", "eval_model", "eval_model.yaml"),
        device, complete=False,
    )
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)
    print("InterCLIP evaluator loaded")
    return eval_wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="FID evaluation for motion retargeting/simulation quality"
    )
    parser.add_argument(
        "--case", nargs="+", default=["all"],
        help="Which case(s) to evaluate: 1, 2, 3, or 'all'"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU ID (-1 for CPU)"
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "InterHuman"),
        help="InterHuman data root"
    )
    parser.add_argument(
        "--retarget_root", type=str,
        default=os.path.join(PROJECT_ROOT, "data", "retargeted_v2", "interhuman"),
        help="Retargeted data root"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96,
        help="Batch size for evaluation (default: 96, same as InterMask)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse cases
    if "all" in args.case:
        cases = [1, 2, 3]
    else:
        cases = [int(c) for c in args.case]

    # Setup
    device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build evaluator
    eval_wrapper = build_evaluator(device)

    # Run cases
    results = {}

    if 1 in cases:
        fid = run_case_1(eval_wrapper, args.data_root, args.batch_size, device)
        results["Case 1 (GT Self-FID)"] = fid

    if 2 in cases:
        fid = run_case_2(
            eval_wrapper, args.data_root, args.retarget_root,
            args.batch_size, device,
        )
        results["Case 2 (Retarget FID)"] = fid

    if 3 in cases:
        fid = run_case_3(
            eval_wrapper, args.data_root, args.retarget_root,
            args.batch_size, device,
        )
        results["Case 3 (Solo-Torque FID)"] = fid

    # Summary
    print("\n" + "=" * 60)
    print("FID EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Settings: emb_scale={EMB_SCALE}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"Evaluator: InterCLIP (checkpoints/eval_model/)")
    print(f"Feature: positions + velocities (rotations = population mean)")
    print("-" * 60)
    for name, val in results.items():
        if val is not None:
            print(f"  {name}: {val:.4f}")
        else:
            print(f"  {name}: N/A (no data)")
    print("=" * 60)


if __name__ == "__main__":
    main()
