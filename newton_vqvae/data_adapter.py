"""
data_adapter.py — Load InterHuman data for Python ≥3.10.

Bypasses torch_geometric (SAGEConv) dependency by implementing a
lightweight mean-aggregation GCN that works with plain PyTorch.

Provides:
  - InterHumanPhysicsDataset: yields (motion, betas_p1, betas_p2, clip_id)
  - LightSAGEConv: drop-in replacement for SAGEConv (mean + linear)
"""
from __future__ import annotations

import os
import sys
import pickle
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# ── Add InterMask root to path so we can import data.quaternion etc. ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.quaternion import qmul_np, qinv_np, qrot_np, qbetween_np
from data.utils import rigid_transform

# ═══════════════════════════════════════════════════════════════
# Edge topology (same as utils/paramUtil.py but self-contained)
# ═══════════════════════════════════════════════════════════════

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

def _build_edge_indices() -> torch.LongTensor:
    edges = []
    for chain in T2M_KINEMATIC_CHAIN:
        for i in range(len(chain) - 1):
            edges.append([chain[i], chain[i + 1]])
            edges.append([chain[i + 1], chain[i]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

EDGE_INDICES = _build_edge_indices()  # (2, E)


# ═══════════════════════════════════════════════════════════════
# Lightweight SAGEConv replacement
# ═══════════════════════════════════════════════════════════════

class LightSAGEConv(nn.Module):
    """
    Drop-in replacement for torch_geometric.nn.SAGEConv(project=True).

    Matches the exact parameter layout of PyG SAGEConv so checkpoints load:
        lin   : Linear(in, in)   — project neighbor features
        lin_l : Linear(in, out)  — transform self features
        lin_r : Linear(in, out, bias=False) — transform aggregated features
        output = lin_l(x) + lin_r(mean_agg(lin(x_neighbors)))

    Works with plain PyTorch — no torch_geometric needed.
    """

    def __init__(self, in_channels: int, out_channels: int, project: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Neighbor projection (in → in)
        if project:
            self.lin = nn.Linear(in_channels, in_channels, bias=True)
        else:
            self.lin = None

        # Self transform (in → out)
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        # Aggregated neighbor transform (in → out, no bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in) node features (N = total nodes in batch, flattened)
            edge_index: (2, E) directed edges [src, dst]

        Returns:
            (N, C_out) updated features
        """
        N, C = x.shape
        src, dst = edge_index  # both (E,)

        # Project neighbour features
        neigh = x[src]
        if self.lin is not None:
            neigh = self.lin(neigh)

        # Mean-aggregate per destination
        agg = torch.zeros(N, C, device=x.device, dtype=x.dtype)
        count = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, C), neigh)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst, dtype=x.dtype).unsqueeze(1))
        count = count.clamp(min=1.0)
        agg = agg / count

        out = self.lin_l(x) + self.lin_r(agg)
        return out


# ═══════════════════════════════════════════════════════════════
# Motion processing (copied from data/utils.py to avoid imports
# that depend on hard-coded paths)
# ═══════════════════════════════════════════════════════════════

class MotionNormalizer:
    """Mean/std normalizer for InterHuman motions."""

    def __init__(self, stats_dir: Optional[str] = None, device: Optional[str] = None):
        if stats_dir is None:
            stats_dir = os.path.join(_PROJECT_ROOT, "data", "stats")
        self.mean_np = np.load(os.path.join(stats_dir, "global_mean.npy"))
        self.std_np = np.load(os.path.join(stats_dir, "global_std.npy"))
        # Torch versions (lazy-initialized per device)
        self._mean_t = None
        self._std_t = None
        self._device = device

    def _ensure_torch(self, device):
        if self._mean_t is None or self._mean_t.device != device:
            self._mean_t = torch.from_numpy(self.mean_np).float().to(device)
            self._std_t = torch.from_numpy(self.std_np).float().to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return (x - self.mean_np) / self.std_np
        # Torch tensor
        self._ensure_torch(x.device)
        return (x - self._mean_t) / self._std_t

    def backward(self, x):
        if isinstance(x, np.ndarray):
            return x * self.std_np + self.mean_np
        # Torch tensor
        self._ensure_torch(x.device)
        return x * self._std_t + self._mean_t


def _process_motion_np(motion: np.ndarray, feet_thre: float = 0.001,
                       prev_start: int = 0, n_joints: int = 22):
    """
    Process raw motion: coord transform, floor align, XZ center, face Z+,
    compute velocities and foot contacts.

    Matches InterMask's process_motion_np from data/utils.py exactly.

    Input:  (T, 192) — positions(66) + rotations(126)
    Output: (T-1, 262) — positions(66) + velocities(66) + rotations(126) + feet(4)

    Returns:
        data: processed motion (T-1, 262)
        root_quat_init: initial root quaternion (1, 4)
        root_pos_init_xz: initial root XZ position (1, 3)
    """
    # Coordinate transform (Y-up to Z-up)
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0]], dtype=np.float32)

    positions = motion[:, :n_joints * 3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints * 3:]

    # Apply coordinate transform
    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    # Floor alignment
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # XZ centering (root joint only)
    root_pos_init = positions[prev_start]  # (n_joints, 3)
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])  # (3,)
    positions = positions - root_pose_init_xz

    # Face Z+ direction
    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip = face_joint_indx[0], face_joint_indx[1]
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init_for_all, positions)

    # Foot contact detection
    fid_l = [7, 10]
    fid_r = [8, 11]

    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        heightfactor = np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Construct output: positions[:-1] + velocities + rotations[:-1] + feet
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rotations[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]


def _load_motion(path: str, min_length: int):
    """Load a .npy motion file, extract positions+rotations matching InterMask format.

    Raw files are (T, 492) with 62-joint positions and hand rotations.
    We extract 22-joint positions (66) and 21-joint cont6d rotations (126) → (T, 192).
    This matches InterMask's load_motion in data/utils.py.
    """
    try:
        motion = np.load(path).astype(np.float32)
    except Exception:
        return None
    # Extract positions (22 joints * 3) and rotations (21 joints * 6)
    positions = motion[:, :22 * 3]           # (T, 66)
    rotations = motion[:, 62 * 3:62 * 3 + 21 * 6]  # (T, 126)
    motion = np.concatenate([positions, rotations], axis=1)  # (T, 192)
    if motion.shape[0] < min_length:
        return None
    return motion


# ═══════════════════════════════════════════════════════════════
# Dataset: InterHuman + betas for Newton skeleton generation
# ═══════════════════════════════════════════════════════════════

class InterHumanPhysicsDataset(Dataset):
    """
    InterHuman dataset that also loads per-clip SMPL-X betas.

    Each sample yields:
        motion:   (window_size, 262)  normalised motion snippet
        betas:    (10,)               SMPL-X shape for this person
        clip_id:  str                 e.g. "000234"
        person:   int                 0 or 1
    """

    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        window_size: int = 64,
        window_stride: int = 10,
        normalize: bool = True,
        stats_dir: Optional[str] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.normalizer = MotionNormalizer(stats_dir)

        # ── Load split list ──
        split_file = os.path.join(data_root, "split", f"{mode}.txt")
        data_list = set()
        if os.path.isfile(split_file):
            data_list = {l.strip() for l in open(split_file)}

        ignore_file = os.path.join(data_root, "split", "ignore_list.txt")
        ignore_set = set()
        if os.path.isfile(ignore_file):
            ignore_set = {l.strip() for l in open(ignore_file)}

        # ── Collect motions ──
        # Stores: (motion_array, betas, clip_id, person_idx)
        self._samples: List[Tuple[np.ndarray, np.ndarray, str, int]] = []

        motions_dir = os.path.join(data_root, "motions_processed")
        pkls_dir = os.path.join(data_root, "motions")

        for root, dirs, files in os.walk(motions_dir):
            for file in tqdm(files, desc=f"Loading {mode}"):
                if not file.endswith(".npy") or "person1" not in root:
                    continue
                clip_id = file.split(".")[0]
                if clip_id in ignore_set or clip_id not in data_list:
                    continue

                p1_path = os.path.join(root, file)
                p2_path = p1_path.replace("person1", "person2")

                m1 = _load_motion(p1_path, window_size)
                m2 = _load_motion(p2_path, window_size)
                if m1 is None or m2 is None:
                    continue

                # Load betas from the raw pkl
                betas_p1, betas_p2 = self._load_betas(pkls_dir, clip_id)

                # Process (floor align, XZ center, face Z+)
                m1, rq1, rp1 = _process_motion_np(m1, n_joints=22)
                m2, rq2, rp2 = _process_motion_np(m2, n_joints=22)

                # Align person2 relative to person1
                # rq1/rq2: (1,4), rp1/rp2: (1,3) — root XZ position
                r_rel = qmul_np(rq2, qinv_np(rq1))
                angle = np.arctan2(r_rel[:, 2:3], r_rel[:, 0:1])
                xz = qrot_np(rq1, rp2 - rp1)[:, [0, 2]]
                relative = np.concatenate([angle, xz], axis=-1)[0]
                m2 = rigid_transform(relative, m2)

                self._samples.append((m1, betas_p1, clip_id, 0))
                self._samples.append((m2, betas_p2, clip_id, 1))

        # ── Build window indices ──
        self._indices: List[Tuple[int, int]] = []
        for sidx, (motion, _, _, _) in enumerate(self._samples):
            T = motion.shape[0]
            for t in range(0, T - self.window_size + 1, self.window_stride):
                self._indices.append((sidx, t))

        print(f"[InterHumanPhysics] {mode}: {len(self._samples)} motions, "
              f"{len(self._indices)} windows")

    @staticmethod
    def _load_betas(pkls_dir: str, clip_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load per-person betas from the raw InterHuman pkl."""
        pkl_path = os.path.join(pkls_dir, f"{clip_id}.pkl")
        default_betas = np.zeros(10, dtype=np.float64)
        if not os.path.isfile(pkl_path):
            return default_betas, default_betas
        try:
            with open(pkl_path, "rb") as f:
                raw = pickle.load(f)
            b1 = raw.get("person1", {}).get("betas", default_betas).astype(np.float64)
            b2 = raw.get("person2", {}).get("betas", default_betas).astype(np.float64)
            # Handle shape: betas may be (1,10) or (10,)
            if b1.ndim > 1:
                b1 = b1[0]
            if b2.ndim > 1:
                b2 = b2[0]
            return b1, b2
        except Exception:
            return default_betas, default_betas

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        sidx, t = self._indices[idx]
        motion, betas, clip_id, person_idx = self._samples[sidx]

        window = motion[t:t + self.window_size].copy()
        if self.normalize:
            window = self.normalizer.forward(window)

        return {
            "motion": torch.from_numpy(window).float(),        # (W, 262)
            "betas": torch.from_numpy(betas).float(),           # (10,)
            "clip_id": clip_id,
            "person_idx": person_idx,
        }


# ═══════════════════════════════════════════════════════════════
# Phase 2 dataset: pairs of motions for interaction coupling
# ═══════════════════════════════════════════════════════════════

class InterHumanPairDataset(Dataset):
    """
    Yields pairs (motion_p1, motion_p2, betas_p1, betas_p2) for Phase 2.
    """

    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        window_size: int = 64,
        window_stride: int = 10,
        normalize: bool = True,
        stats_dir: Optional[str] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.normalizer = MotionNormalizer(stats_dir)

        split_file = os.path.join(data_root, "split", f"{mode}.txt")
        data_list = set()
        if os.path.isfile(split_file):
            data_list = {l.strip() for l in open(split_file)}

        ignore_file = os.path.join(data_root, "split", "ignore_list.txt")
        ignore_set = set()
        if os.path.isfile(ignore_file):
            ignore_set = {l.strip() for l in open(ignore_file)}

        # (m1, m2, betas1, betas2, clip_id)
        self._pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []

        motions_dir = os.path.join(data_root, "motions_processed")
        pkls_dir = os.path.join(data_root, "motions")

        for root, dirs, files in os.walk(motions_dir):
            for file in tqdm(files, desc=f"Loading pairs {mode}"):
                if not file.endswith(".npy") or "person1" not in root:
                    continue
                clip_id = file.split(".")[0]
                if clip_id in ignore_set or clip_id not in data_list:
                    continue

                p1_path = os.path.join(root, file)
                p2_path = p1_path.replace("person1", "person2")

                m1 = _load_motion(p1_path, window_size)
                m2 = _load_motion(p2_path, window_size)
                if m1 is None or m2 is None:
                    continue

                betas_p1, betas_p2 = InterHumanPhysicsDataset._load_betas(
                    pkls_dir, clip_id
                )

                m1, rq1, rp1 = _process_motion_np(m1, n_joints=22)
                m2, rq2, rp2 = _process_motion_np(m2, n_joints=22)
                r_rel = qmul_np(rq2, qinv_np(rq1))
                angle = np.arctan2(r_rel[:, 2:3], r_rel[:, 0:1])
                xz = qrot_np(rq1, rp2 - rp1)[:, [0, 2]]
                relative = np.concatenate([angle, xz], axis=-1)[0]
                m2 = rigid_transform(relative, m2)

                self._pairs.append((m1, m2, betas_p1, betas_p2, clip_id))

        self._indices: List[Tuple[int, int]] = []
        for pidx, (m1, m2, _, _, _) in enumerate(self._pairs):
            T = min(m1.shape[0], m2.shape[0])
            for t in range(0, T - self.window_size + 1, self.window_stride):
                self._indices.append((pidx, t))

        print(f"[InterHumanPair] {mode}: {len(self._pairs)} pairs, "
              f"{len(self._indices)} windows")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        pidx, t = self._indices[idx]
        m1, m2, b1, b2, clip_id = self._pairs[pidx]

        w1 = m1[t:t + self.window_size].copy()
        w2 = m2[t:t + self.window_size].copy()
        if self.normalize:
            w1 = self.normalizer.forward(w1)
            w2 = self.normalizer.forward(w2)

        return {
            "motion_p1": torch.from_numpy(w1).float(),
            "motion_p2": torch.from_numpy(w2).float(),
            "betas_p1": torch.from_numpy(b1).float(),
            "betas_p2": torch.from_numpy(b2).float(),
            "clip_id": clip_id,
        }
