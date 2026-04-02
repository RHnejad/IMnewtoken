#!/usr/bin/env python3
"""SMPL-X mesh contact analysis for two-person interaction datasets.

This script reconstructs full-body meshes for person1/person2 from InterHuman or
InterX clips, then computes per-frame mesh contact status:

- penetrating
- touching
- barely_touching
- not_touching

It works in two modes:
1) Preferred mode (if trimesh is installed): signed-distance style checks can be
   added externally. This repository currently uses a robust no-extra-deps
   fallback that relies on:
   - nearest-vertex distances via scipy.spatial.cKDTree
   - ray-casting inside tests for candidate near-contact vertices

The fallback is designed to run in the current environment (torch/scipy/h5py),
without requiring trimesh/rtree.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class ContactConfig:
    touching_threshold_m: float = 0.005
    barely_threshold_m: float = 0.020
    penetration_probe_distance_m: float = 0.010
    penetration_min_depth_m: float = 0.002
    self_penetration_mode: str = "off"  # off | heuristic
    self_penetration_threshold_m: float = 0.004
    self_penetration_k: int = 12
    self_penetration_normal_dot_max: float = -0.2
    max_inside_queries_per_mesh: int = 64
    ray_eps: float = 1e-8


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _normalize_betas(raw_betas: np.ndarray, t_len: int, n_betas: int = 10) -> np.ndarray:
    """Normalize betas into either (n_betas,) or (T, n_betas)."""
    b = np.asarray(raw_betas, dtype=np.float32)
    if b.ndim == 1:
        if b.shape[0] < n_betas:
            out = np.zeros((n_betas,), dtype=np.float32)
            out[: b.shape[0]] = b
            return out
        return b[:n_betas]

    if b.ndim == 2:
        if b.shape[0] == t_len:
            if b.shape[1] >= n_betas:
                return b[:, :n_betas]
            out = np.zeros((t_len, n_betas), dtype=np.float32)
            out[:, : b.shape[1]] = b
            return out
        # Some files may store shape as (1, n_betas)
        row = b[0]
        return _normalize_betas(row, t_len=t_len, n_betas=n_betas)

    raise ValueError(f"Unsupported betas shape: {b.shape}")


def _extract_pose_hand(person_dict: Dict[str, np.ndarray], t_len: int) -> np.ndarray:
    """Extract 90D hand pose if present; otherwise return zeros."""
    if "pose_hand" in person_dict:
        ph = np.asarray(person_dict["pose_hand"], dtype=np.float32)
        if ph.shape[0] != t_len:
            raise ValueError(f"pose_hand length mismatch: {ph.shape} vs T={t_len}")
        if ph.shape[1] != 90:
            if ph.shape[1] < 90:
                out = np.zeros((t_len, 90), dtype=np.float32)
                out[:, : ph.shape[1]] = ph
                return out
            return ph[:, :90]
        return ph

    # Common alternate naming: left/right hand pose 45D each.
    left_keys = ["left_hand_pose", "pose_lhand", "lhand_pose"]
    right_keys = ["right_hand_pose", "pose_rhand", "rhand_pose"]
    lk = next((k for k in left_keys if k in person_dict), None)
    rk = next((k for k in right_keys if k in person_dict), None)
    if lk is not None and rk is not None:
        left = np.asarray(person_dict[lk], dtype=np.float32)
        right = np.asarray(person_dict[rk], dtype=np.float32)
        if left.shape[0] != t_len or right.shape[0] != t_len:
            raise ValueError("left/right hand pose length mismatch")
        left = left[:, :45] if left.shape[1] >= 45 else np.pad(left, ((0, 0), (0, 45 - left.shape[1])))
        right = right[:, :45] if right.shape[1] >= 45 else np.pad(right, ((0, 0), (0, 45 - right.shape[1])))
        return np.concatenate([left, right], axis=1)

    return np.zeros((t_len, 90), dtype=np.float32)


def load_interhuman_clip(data_root: str, clip_id: str) -> List[Dict[str, np.ndarray]]:
    """Load InterHuman clip from pkl and parse per-person SMPL-X parameters."""
    candidates = [
        os.path.join(data_root, "motions", f"{clip_id}.pkl"),
        os.path.join(data_root, f"{clip_id}.pkl"),
    ]
    pkl_path = next((p for p in candidates if os.path.isfile(p)), None)
    if pkl_path is None:
        tried = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(f"InterHuman clip '{clip_id}' not found. Tried:\n{tried}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    persons = []
    for key in ("person1", "person2"):
        if key not in raw:
            continue
        p = raw[key]
        trans = _as_float32(p["trans"])
        root_orient = _as_float32(p["root_orient"])
        pose_body = _as_float32(p["pose_body"])
        t_len = trans.shape[0]

        if root_orient.shape != (t_len, 3):
            raise ValueError(f"{key} root_orient shape mismatch: {root_orient.shape}")
        if pose_body.shape != (t_len, 63):
            raise ValueError(f"{key} pose_body shape mismatch: {pose_body.shape}")

        betas = _normalize_betas(p["betas"], t_len=t_len, n_betas=10)
        pose_hand = _extract_pose_hand(p, t_len=t_len)

        persons.append(
            {
                "name": key,
                "trans": trans,
                "root_orient": root_orient,
                "pose_body": pose_body,
                "pose_hand": pose_hand,
                "betas": betas,
            }
        )

    if len(persons) < 2:
        raise ValueError(
            f"InterHuman clip '{clip_id}' has fewer than 2 persons. Found: {[p['name'] for p in persons]}"
        )
    return persons


def _interx_h5_candidates(data_root: str, explicit_h5: Optional[str]) -> List[str]:
    if explicit_h5 is not None:
        if os.path.isfile(explicit_h5):
            return [explicit_h5]
        raise FileNotFoundError(f"Explicit --h5-file does not exist: {explicit_h5}")

    candidates = [
        os.path.join(data_root, "processed", "motions", "inter-x.h5"),
        os.path.join(data_root, "processed", "inter-x.h5"),
        os.path.join(data_root, "processed", "motions", "train.h5"),
        os.path.join(data_root, "processed", "motions", "val.h5"),
        os.path.join(data_root, "processed", "motions", "test.h5"),
    ]
    existing = [p for p in candidates if os.path.isfile(p)]
    if not existing:
        tried = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(f"InterX H5 not found. Tried:\n{tried}")
    return existing


def load_interx_clip(
    data_root: str,
    clip_id: str,
    h5_file: Optional[str] = None,
    convert_to_zup: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """Load InterX clip from H5 and parse per-person SMPL-X parameters."""
    import h5py

    h5_candidates = _interx_h5_candidates(data_root, h5_file)
    motion = None
    preview_keys: List[str] = []
    for h5_path in h5_candidates:
        with h5py.File(h5_path, "r") as hf:
            if not preview_keys:
                preview_keys = list(hf.keys())[:5]
            if clip_id in hf:
                motion = hf[clip_id][:].astype(np.float32)
                break
    if motion is None:
        raise KeyError(
            f"InterX clip '{clip_id}' not found in available H5 files: {h5_candidates}. "
            f"Example keys: {preview_keys}"
        )

    if motion.ndim != 3 or motion.shape[1:] != (56, 6):
        raise ValueError(f"Unexpected InterX clip shape: {motion.shape}, expected (T,56,6)")

    t_len = motion.shape[0]
    persons = []

    if convert_to_zup:
        # InterX Y-up -> Z-up via +90deg around X.
        r_mat = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        r_conv = Rotation.from_matrix(r_mat)
    else:
        r_mat = None
        r_conv = None

    for pidx in (0, 1):
        p = motion[:, :, pidx * 3 : (pidx + 1) * 3]

        root_orient = p[:, 0, :].astype(np.float32)
        pose_body = p[:, 1:22, :].reshape(t_len, 63).astype(np.float32)
        pose_hand = p[:, 25:55, :].reshape(t_len, 90).astype(np.float32)
        trans = p[:, 55, :].astype(np.float32)

        if convert_to_zup:
            trans = (r_mat @ trans.T).T.astype(np.float32)
            root_rot = Rotation.from_rotvec(root_orient.astype(np.float64))
            root_orient = (r_conv * root_rot).as_rotvec().astype(np.float32)

        persons.append(
            {
                "name": f"person{pidx + 1}",
                "trans": trans,
                "root_orient": root_orient,
                "pose_body": pose_body,
                "pose_hand": pose_hand,
                "betas": np.zeros((10,), dtype=np.float32),
            }
        )

    return persons


def reconstruct_smplx_mesh_sequence(
    persons: Sequence[Dict[str, np.ndarray]],
    body_model_path: str,
    device: str = "cuda",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct meshes for two persons.

    Returns:
        vertices_p1: (T, V, 3)
        vertices_p2: (T, V, 3)
        faces: (F, 3)
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for SMPL-X mesh reconstruction") from exc

    from data.body_model.body_model import BodyModel

    if len(persons) < 2:
        raise ValueError("Need two persons for contact analysis")
    if not os.path.isfile(body_model_path):
        raise FileNotFoundError(f"SMPL-X model not found: {body_model_path}")

    torch_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    bm = BodyModel(
        bm_fname=body_model_path,
        num_betas=10,
        num_expressions=10,
        dtype=torch.float32,
    ).to(torch_device)
    bm.eval()

    faces = bm.f.detach().cpu().numpy().astype(np.int32)

    out_vertices: List[np.ndarray] = []
    for person in persons[:2]:
        trans_np = _as_float32(person["trans"])
        root_np = _as_float32(person["root_orient"])
        pose_body_np = _as_float32(person["pose_body"])
        pose_hand_np = _as_float32(person.get("pose_hand", np.zeros((trans_np.shape[0], 90), dtype=np.float32)))
        betas_np = np.asarray(person["betas"], dtype=np.float32)

        t_len = trans_np.shape[0]
        if root_np.shape != (t_len, 3) or pose_body_np.shape != (t_len, 63):
            raise ValueError(
                f"{person['name']} tensor shapes mismatch: "
                f"trans={trans_np.shape}, root={root_np.shape}, body={pose_body_np.shape}"
            )

        trans = torch.from_numpy(trans_np).to(torch_device)
        root_orient = torch.from_numpy(root_np).to(torch_device)
        pose_body = torch.from_numpy(pose_body_np).to(torch_device)
        pose_hand = torch.from_numpy(pose_hand_np).to(torch_device)

        if betas_np.ndim == 1:
            betas = torch.from_numpy(betas_np[None, :]).to(torch_device)
            per_frame_betas = False
        elif betas_np.ndim == 2 and betas_np.shape[0] == t_len:
            betas = torch.from_numpy(betas_np).to(torch_device)
            per_frame_betas = True
        else:
            betas = torch.from_numpy(betas_np.reshape(1, -1)).to(torch_device)
            per_frame_betas = False

        v_batches = []
        for s in range(0, t_len, batch_size):
            e = min(s + batch_size, t_len)
            bs = e - s
            if per_frame_betas:
                betas_batch = betas[s:e]
            else:
                betas_batch = betas.expand(bs, -1)
            with torch.no_grad():
                out = bm(
                    root_orient=root_orient[s:e],
                    pose_body=pose_body[s:e],
                    pose_hand=pose_hand[s:e],
                    betas=betas_batch,
                    trans=trans[s:e],
                )
            v_batches.append(out.v.detach().cpu().numpy().astype(np.float32))

        out_vertices.append(np.concatenate(v_batches, axis=0))

    v1, v2 = out_vertices
    t_common = min(v1.shape[0], v2.shape[0])
    return v1[:t_common], v2[:t_common], faces


class MeshContactAnalyzer:
    """Frame-wise mesh contact analyzer."""

    def __init__(self, faces: np.ndarray, config: ContactConfig):
        self.faces = np.asarray(faces, dtype=np.int32)
        self.cfg = config
        self.vertex_neighbors = self._build_vertex_neighbors(self.faces)

        ray_dir = np.array([1.0, 0.372, 0.845], dtype=np.float64)
        ray_dir /= np.linalg.norm(ray_dir)
        self.ray_dir = ray_dir

    @staticmethod
    def _build_vertex_neighbors(faces: np.ndarray) -> Dict[int, np.ndarray]:
        neighbors: Dict[int, set] = {}
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            neighbors.setdefault(a, set()).update((b, c))
            neighbors.setdefault(b, set()).update((a, c))
            neighbors.setdefault(c, set()).update((a, b))
        return {k: np.array(sorted(v), dtype=np.int32) for k, v in neighbors.items()}

    def _compute_vertex_normals(self, vertices: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(vertices, dtype=np.float32)
        tri = vertices[self.faces]  # (F,3,3)
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # (F,3)
        for corner in range(3):
            np.add.at(normals, self.faces[:, corner], fn)
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-8)
        return normals / nrm

    def _estimate_self_penetration(self, vertices: np.ndarray) -> Tuple[np.ndarray, float]:
        """Heuristic self-penetration detector based on close non-neighbor surface pairs."""
        cfg = self.cfg
        if vertices.shape[0] == 0 or cfg.self_penetration_k <= 0:
            return np.zeros((0,), dtype=np.int32), 0.0

        k = max(2, int(cfg.self_penetration_k) + 1)  # +1 includes self in cKDTree query
        tree = cKDTree(vertices)
        dists, nn_idx = tree.query(vertices, k=k, workers=-1)
        dists = np.asarray(dists, dtype=np.float32)
        nn_idx = np.asarray(nn_idx, dtype=np.int32)
        normals = self._compute_vertex_normals(vertices).astype(np.float32)

        hits: List[int] = []
        min_non_neighbor = np.full((vertices.shape[0],), np.inf, dtype=np.float32)
        for vid in range(vertices.shape[0]):
            forbidden = self.vertex_neighbors.get(vid, np.zeros((0,), dtype=np.int32))
            for j in range(1, nn_idx.shape[1]):  # skip self at rank 0
                nid = int(nn_idx[vid, j])
                if nid == vid:
                    continue
                if forbidden.size > 0 and np.any(forbidden == nid):
                    continue
                dist = float(dists[vid, j])
                min_non_neighbor[vid] = dist
                normal_dot = float(np.dot(normals[vid], normals[nid]))
                if dist <= cfg.self_penetration_threshold_m and normal_dot <= cfg.self_penetration_normal_dot_max:
                    hits.append(vid)
                break

        hit_idx = np.unique(np.asarray(hits, dtype=np.int32))
        if hit_idx.size == 0:
            return hit_idx, 0.0
        depth_est = float(np.max(cfg.self_penetration_threshold_m - min_non_neighbor[hit_idx]))
        return hit_idx, max(0.0, depth_est)

    def analyze(
        self,
        vertices_p1: np.ndarray,
        vertices_p2: np.ndarray,
        frame_indices: Optional[np.ndarray] = None,
        verbose_every: int = 50,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, np.ndarray]]]:
        if vertices_p1.shape != vertices_p2.shape:
            raise ValueError(f"Vertex sequence shape mismatch: {vertices_p1.shape} vs {vertices_p2.shape}")

        t_len = vertices_p1.shape[0]
        if frame_indices is None:
            frame_indices = np.arange(t_len, dtype=np.int32)

        summaries: List[Dict[str, object]] = []
        details: List[Dict[str, np.ndarray]] = []

        start = time.time()
        for i, t in enumerate(frame_indices):
            frame_summary, frame_detail = self._analyze_frame(vertices_p1[t], vertices_p2[t])
            frame_summary["frame"] = int(t)
            summaries.append(frame_summary)
            details.append(frame_detail)

            if verbose_every > 0 and (i + 1) % verbose_every == 0:
                elapsed = time.time() - start
                print(
                    f"  processed {i + 1}/{len(frame_indices)} frames "
                    f"({elapsed:.1f}s, latest={frame_summary['status']}, "
                    f"dmin={frame_summary['min_distance_m']:.4f}m)",
                    flush=True,
                )

        return summaries, details

    def _analyze_frame(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        cfg = self.cfg

        tree1 = cKDTree(v1)
        tree2 = cKDTree(v2)

        d12, idx12 = tree2.query(v1, k=1, workers=-1)
        d21, idx21 = tree1.query(v2, k=1, workers=-1)

        d12 = d12.astype(np.float32)
        d21 = d21.astype(np.float32)
        idx12 = idx12.astype(np.int32)
        idx21 = idx21.astype(np.int32)

        min_d12_idx = int(np.argmin(d12))
        min_d21_idx = int(np.argmin(d21))
        min_d12 = float(d12[min_d12_idx])
        min_d21 = float(d21[min_d21_idx])

        if min_d12 <= min_d21:
            min_dist = min_d12
            closest_p1 = v1[min_d12_idx]
            closest_p2 = v2[idx12[min_d12_idx]]
            closest_pair_source = "p1_to_p2"
        else:
            min_dist = min_d21
            closest_p2 = v2[min_d21_idx]
            closest_p1 = v1[idx21[min_d21_idx]]
            closest_pair_source = "p2_to_p1"

        contact_idx1 = np.where(d12 <= cfg.touching_threshold_m)[0].astype(np.int32)
        contact_idx2 = np.where(d21 <= cfg.touching_threshold_m)[0].astype(np.int32)

        barely_idx1 = np.where((d12 > cfg.touching_threshold_m) & (d12 <= cfg.barely_threshold_m))[0].astype(np.int32)
        barely_idx2 = np.where((d21 > cfg.touching_threshold_m) & (d21 <= cfg.barely_threshold_m))[0].astype(np.int32)

        penetrating_idx1 = np.zeros((0,), dtype=np.int32)
        penetrating_idx2 = np.zeros((0,), dtype=np.int32)
        penetration_depth_est = 0.0
        self_pen_idx1 = np.zeros((0,), dtype=np.int32)
        self_pen_idx2 = np.zeros((0,), dtype=np.int32)
        self_pen_depth_est_1 = 0.0
        self_pen_depth_est_2 = 0.0

        tri2 = self._precompute_triangles(v2)
        tri1 = self._precompute_triangles(v1)

        cand1 = self._candidate_indices_for_inside(d12)
        cand2 = self._candidate_indices_for_inside(d21)

        inside1 = self._points_inside_mesh(v1[cand1], tri2)
        inside2 = self._points_inside_mesh(v2[cand2], tri1)

        # Avoid treating exact surface points as penetration.
        valid_inside1 = inside1 & (d12[cand1] > cfg.penetration_min_depth_m)
        valid_inside2 = inside2 & (d21[cand2] > cfg.penetration_min_depth_m)

        penetrating_idx1 = cand1[valid_inside1]
        penetrating_idx2 = cand2[valid_inside2]

        pen_depth_1 = float(d12[penetrating_idx1].max()) if penetrating_idx1.size > 0 else 0.0
        pen_depth_2 = float(d21[penetrating_idx2].max()) if penetrating_idx2.size > 0 else 0.0
        penetration_depth_est = max(pen_depth_1, pen_depth_2)

        self_mode = str(cfg.self_penetration_mode).lower()
        self_penetration_computed = self_mode != "off"
        if self_mode == "heuristic":
            # Heuristic self-penetration estimate (same-person mesh self-collision proxy).
            self_pen_idx1, self_pen_depth_est_1 = self._estimate_self_penetration(v1)
            self_pen_idx2, self_pen_depth_est_2 = self._estimate_self_penetration(v2)

        has_inter_penetration = penetrating_idx1.size > 0 or penetrating_idx2.size > 0
        has_self_penetration = self_pen_idx1.size > 0 or self_pen_idx2.size > 0
        if has_inter_penetration:
            status = "penetrating"
        elif min_dist <= cfg.touching_threshold_m:
            status = "touching"
        elif min_dist <= cfg.barely_threshold_m:
            status = "barely_touching"
        else:
            status = "not_touching"

        if has_inter_penetration and has_self_penetration:
            penetration_source = "both"
        elif has_inter_penetration:
            penetration_source = "other_person_only"
        elif has_self_penetration:
            penetration_source = "self_only"
        else:
            penetration_source = "none"

        summary = {
            "status": status,
            "penetration_source": penetration_source,
            "has_inter_person_penetration": bool(has_inter_penetration),
            "has_self_penetration": bool(has_self_penetration),
            "self_penetration_computed": bool(self_penetration_computed),
            "self_penetration_mode": self_mode,
            "min_distance_m": float(min_dist),
            "closest_pair_source": closest_pair_source,
            "closest_point_p1": closest_p1.astype(np.float32).tolist(),
            "closest_point_p2": closest_p2.astype(np.float32).tolist(),
            "contact_vertex_count_p1": int(contact_idx1.size),
            "contact_vertex_count_p2": int(contact_idx2.size),
            "barely_vertex_count_p1": int(barely_idx1.size),
            "barely_vertex_count_p2": int(barely_idx2.size),
            "inter_person_penetrating_vertex_count_p1": int(penetrating_idx1.size),
            "inter_person_penetrating_vertex_count_p2": int(penetrating_idx2.size),
            "inter_person_penetration_depth_est_m": float(penetration_depth_est),
            "self_penetrating_vertex_count_p1": int(self_pen_idx1.size),
            "self_penetrating_vertex_count_p2": int(self_pen_idx2.size),
            "self_penetration_depth_est_m_p1": float(self_pen_depth_est_1),
            "self_penetration_depth_est_m_p2": float(self_pen_depth_est_2),
            "self_penetration_threshold_m": float(cfg.self_penetration_threshold_m),
            # Backward-compatible aggregate field name.
            "penetration_depth_est_m": float(max(penetration_depth_est, self_pen_depth_est_1, self_pen_depth_est_2)),
        }

        detail = {
            "contact_vidx_p1": contact_idx1,
            "contact_vidx_p2": contact_idx2,
            "barely_vidx_p1": barely_idx1,
            "barely_vidx_p2": barely_idx2,
            "inter_person_penetrating_vidx_p1": penetrating_idx1.astype(np.int32),
            "inter_person_penetrating_vidx_p2": penetrating_idx2.astype(np.int32),
            "self_penetrating_vidx_p1": self_pen_idx1.astype(np.int32),
            "self_penetrating_vidx_p2": self_pen_idx2.astype(np.int32),
            # Backward-compatible detail keys.
            "penetrating_vidx_p1": penetrating_idx1.astype(np.int32),
            "penetrating_vidx_p2": penetrating_idx2.astype(np.int32),
            "nearest_vidx12": idx12,
            "nearest_vidx21": idx21,
            "nearest_dist12": d12,
            "nearest_dist21": d21,
        }
        return summary, detail

    def _candidate_indices_for_inside(self, nearest_dist: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        n = nearest_dist.shape[0]
        k = min(cfg.max_inside_queries_per_mesh, n)
        if k <= 0:
            return np.zeros((0,), dtype=np.int32)

        # Prioritize nearest-surface vertices (likely contact/penetration).
        near_k = max(1, k // 2)
        near_idx = np.argsort(nearest_dist)[:near_k].astype(np.int32)

        # Add uniform coverage vertices so deep interpenetration is not missed
        # when nearest-vertex distance is large.
        if k > near_idx.size:
            uniform_count = k - near_idx.size
            uniform_idx = np.linspace(0, n - 1, num=uniform_count, dtype=np.int32)
            idx = np.concatenate([near_idx, uniform_idx], axis=0)
        else:
            idx = near_idx

        idx = np.unique(idx)
        if idx.size > k:
            # Keep nearest vertices first when trimming.
            order = np.argsort(nearest_dist[idx])
            idx = idx[order[:k]]
        return idx.astype(np.int32)

    def _precompute_triangles(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tri = vertices[self.faces]  # (F, 3, 3)
        v0 = tri[:, 0, :].astype(np.float64)
        e1 = (tri[:, 1, :] - tri[:, 0, :]).astype(np.float64)
        e2 = (tri[:, 2, :] - tri[:, 0, :]).astype(np.float64)
        return v0, e1, e2

    def _points_inside_mesh(
        self,
        points: np.ndarray,
        tri_precomp: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        v0, e1, e2 = tri_precomp
        inside = np.zeros((points.shape[0],), dtype=bool)
        for i in range(points.shape[0]):
            inside[i] = self._point_inside_mesh(points[i].astype(np.float64), v0, e1, e2)
        return inside

    def _point_inside_mesh(self, point: np.ndarray, v0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> bool:
        """Odd-even ray casting test with Moller-Trumbore intersections."""
        d = self.ray_dir
        eps = self.cfg.ray_eps

        pvec = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pvec)

        valid = np.abs(det) > eps
        if not np.any(valid):
            return False

        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]

        tvec = point[None, :] - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)
        if not np.any(valid):
            return False

        qvec = np.cross(tvec, e1)
        v = (qvec @ d) * inv_det
        valid &= (v >= 0.0) & ((u + v) <= 1.0)
        if not np.any(valid):
            return False

        t = np.einsum("ij,ij->i", e2, qvec) * inv_det
        valid &= t > eps

        hits = int(np.count_nonzero(valid))
        return (hits % 2) == 1


def write_colored_ply(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Write ASCII PLY with per-vertex RGB colors."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    c = np.asarray(colors, dtype=np.uint8)

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property uchar red\n")
        fp.write("property uchar green\n")
        fp.write("property uchar blue\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for i in range(v.shape[0]):
            fp.write(f"{v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f} {int(c[i,0])} {int(c[i,1])} {int(c[i,2])}\n")
        for i in range(f.shape[0]):
            fp.write(f"3 {int(f[i,0])} {int(f[i,1])} {int(f[i,2])}\n")


def export_contact_frame_ply(
    output_dir: str,
    frame_index: int,
    vertices_p1: np.ndarray,
    vertices_p2: np.ndarray,
    faces: np.ndarray,
    detail: Dict[str, np.ndarray],
) -> Dict[str, str]:
    """Export one frame with contact labels as colored PLY files."""
    os.makedirs(output_dir, exist_ok=True)

    def make_colors(
        n_verts: int,
        contact: np.ndarray,
        barely: np.ndarray,
        inter_pen: np.ndarray,
        self_pen: np.ndarray,
    ) -> np.ndarray:
        col = np.full((n_verts, 3), 170, dtype=np.uint8)
        if barely.size > 0:
            col[barely] = np.array([255, 190, 80], dtype=np.uint8)
        if contact.size > 0:
            col[contact] = np.array([230, 60, 60], dtype=np.uint8)
        if inter_pen.size > 0:
            col[inter_pen] = np.array([170, 40, 230], dtype=np.uint8)
        if self_pen.size > 0:
            col[self_pen] = np.array([40, 220, 120], dtype=np.uint8)
        return col

    c1 = make_colors(
        vertices_p1.shape[0],
        detail["contact_vidx_p1"],
        detail["barely_vidx_p1"],
        detail.get("inter_person_penetrating_vidx_p1", detail["penetrating_vidx_p1"]),
        detail.get("self_penetrating_vidx_p1", np.zeros((0,), dtype=np.int32)),
    )
    c2 = make_colors(
        vertices_p2.shape[0],
        detail["contact_vidx_p2"],
        detail["barely_vidx_p2"],
        detail.get("inter_person_penetrating_vidx_p2", detail["penetrating_vidx_p2"]),
        detail.get("self_penetrating_vidx_p2", np.zeros((0,), dtype=np.int32)),
    )

    p1_path = os.path.join(output_dir, f"frame_{frame_index:05d}_person1_contact.ply")
    p2_path = os.path.join(output_dir, f"frame_{frame_index:05d}_person2_contact.ply")
    write_colored_ply(p1_path, vertices_p1, faces, c1)
    write_colored_ply(p2_path, vertices_p2, faces, c2)
    return {"person1": p1_path, "person2": p2_path}


def _build_status_summary(frame_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    statuses = [str(x["status"]) for x in frame_summaries]
    uniq = ["penetrating", "touching", "barely_touching", "not_touching"]
    counts = {k: int(sum(1 for s in statuses if s == k)) for k in uniq}

    dmin = np.array([float(x["min_distance_m"]) for x in frame_summaries], dtype=np.float32)
    pen = np.array([float(x["penetration_depth_est_m"]) for x in frame_summaries], dtype=np.float32)
    pen_src = [str(x.get("penetration_source", "none")) for x in frame_summaries]
    src_keys = ["both", "other_person_only", "self_only", "none"]
    src_counts = {k: int(sum(1 for s in pen_src if s == k)) for k in src_keys}

    return {
        "status_counts": counts,
        "status_fractions": {k: (counts[k] / max(1, len(frame_summaries))) for k in uniq},
        "penetration_source_counts": src_counts,
        "penetration_source_fractions": {k: (src_counts[k] / max(1, len(frame_summaries))) for k in src_keys},
        "min_distance_m": {
            "min": float(np.min(dmin)) if dmin.size > 0 else float("nan"),
            "mean": float(np.mean(dmin)) if dmin.size > 0 else float("nan"),
            "max": float(np.max(dmin)) if dmin.size > 0 else float("nan"),
        },
        "penetration_depth_est_m": {
            "max": float(np.max(pen)) if pen.size > 0 else 0.0,
            "mean": float(np.mean(pen)) if pen.size > 0 else 0.0,
        },
    }


def run_self_test() -> None:
    """Quick synthetic sanity checks for status classification."""

    def cube_mesh(center: Tuple[float, float, float], size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        cx, cy, cz = center
        s = size * 0.5
        verts = np.array(
            [
                [cx - s, cy - s, cz - s],
                [cx + s, cy - s, cz - s],
                [cx + s, cy + s, cz - s],
                [cx - s, cy + s, cz - s],
                [cx - s, cy - s, cz + s],
                [cx + s, cy - s, cz + s],
                [cx + s, cy + s, cz + s],
                [cx - s, cy + s, cz + s],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.int32,
        )
        return verts, faces

    cfg = ContactConfig(
        touching_threshold_m=0.01,
        barely_threshold_m=0.03,
        penetration_probe_distance_m=0.03,
        penetration_min_depth_m=0.01,
        max_inside_queries_per_mesh=64,
    )

    v_base, faces = cube_mesh((0.0, 0.0, 0.0), size=1.0)
    analyzer = MeshContactAnalyzer(faces, cfg)

    cases = [
        ("not_touching", (2.0, 0.0, 0.0)),
        ("barely_touching", (1.02, 0.0, 0.0)),
        ("touching", (1.005, 0.0, 0.0)),
        ("penetrating", (0.7, 0.0, 0.0)),
    ]

    print("Running self-test...")
    for expected, center in cases:
        v_other, _ = cube_mesh(center, size=1.0)
        summary, _ = analyzer._analyze_frame(v_base, v_other)
        got = summary["status"]
        ok = got == expected
        print(f"  expected={expected:14s} got={got:14s} min_d={summary['min_distance_m']:.4f} ok={ok}")
        if not ok:
            raise RuntimeError(f"Self-test failed for case {expected}: got {got}")

    print("Self-test passed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMPL-X mesh contact analysis for InterHuman / InterX")

    p.add_argument("--dataset", choices=["interhuman", "interx"], default="interhuman")
    p.add_argument("--clip", type=str, default=None, help="Clip ID (required unless --self-test)")
    p.add_argument("--data-root", type=str, default=None, help="Dataset root path")
    p.add_argument("--h5-file", type=str, default=None, help="InterX optional explicit H5 path")
    p.add_argument(
        "--body-model-path",
        type=str,
        default=os.path.join("data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
        help="SMPL-X neutral model .npz",
    )

    p.add_argument("--device", type=str, default="cuda", help="Torch device (cuda/cpu/cuda:0)")
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=None, help="Exclusive end frame")
    p.add_argument("--frame-step", type=int, default=1)

    p.add_argument("--touching-threshold-m", type=float, default=0.005)
    p.add_argument("--barely-threshold-m", type=float, default=0.020)
    p.add_argument("--penetration-probe-distance-m", type=float, default=0.010)
    p.add_argument("--penetration-min-depth-m", type=float, default=0.002)
    p.add_argument(
        "--self-penetration-mode",
        choices=["off", "heuristic"],
        default="off",
        help="Self-penetration detector mode. 'off' is recommended for large-scale runs.",
    )
    p.add_argument("--self-penetration-threshold-m", type=float, default=0.004)
    p.add_argument("--self-penetration-k", type=int, default=12)
    p.add_argument("--self-penetration-normal-dot-max", type=float, default=-0.2)
    p.add_argument("--max-inside-queries", type=int, default=64)

    p.add_argument("--convert-interx-to-zup", action="store_true", default=True)
    p.add_argument("--no-convert-interx-to-zup", dest="convert_interx_to_zup", action="store_false")

    p.add_argument("--output-json", type=str, default=None, help="Output summary JSON path")
    p.add_argument("--output-details", type=str, default=None, help="Optional pickle path for per-frame contact arrays")

    p.add_argument(
        "--export-ply-frame",
        type=int,
        default=None,
        help="Frame index in original clip to export colored contact PLY",
    )
    p.add_argument("--export-ply-dir", type=str, default=None, help="Directory for exported PLY files")

    p.add_argument("--self-test", action="store_true", help="Run synthetic contact checks and exit")
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.clip is None:
        raise ValueError("--clip is required unless --self-test is used")

    if args.data_root is None:
        args.data_root = (
            os.path.join(PROJECT_ROOT, "data", "InterHuman")
            if args.dataset == "interhuman"
            else os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset")
        )

    if not args.quiet:
        print(f"Dataset: {args.dataset}")
        print(f"Clip: {args.clip}")
        print(f"Data root: {args.data_root}")

    if args.dataset == "interhuman":
        persons = load_interhuman_clip(args.data_root, args.clip)
    else:
        persons = load_interx_clip(
            args.data_root,
            args.clip,
            h5_file=args.h5_file,
            convert_to_zup=args.convert_interx_to_zup,
        )

    v1, v2, faces = reconstruct_smplx_mesh_sequence(
        persons,
        body_model_path=args.body_model_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    t_len = min(v1.shape[0], v2.shape[0])
    frame_start = max(0, args.frame_start)
    frame_end = t_len if args.frame_end is None else min(t_len, args.frame_end)
    if frame_start >= frame_end:
        raise ValueError(f"Invalid frame range: [{frame_start}, {frame_end})")

    frame_indices = np.arange(frame_start, frame_end, max(1, args.frame_step), dtype=np.int32)

    cfg = ContactConfig(
        touching_threshold_m=args.touching_threshold_m,
        barely_threshold_m=args.barely_threshold_m,
        penetration_probe_distance_m=args.penetration_probe_distance_m,
        penetration_min_depth_m=args.penetration_min_depth_m,
        self_penetration_mode=args.self_penetration_mode,
        self_penetration_threshold_m=args.self_penetration_threshold_m,
        self_penetration_k=args.self_penetration_k,
        self_penetration_normal_dot_max=args.self_penetration_normal_dot_max,
        max_inside_queries_per_mesh=args.max_inside_queries,
    )

    if not args.quiet:
        print(
            "Thresholds: "
            f"touch={cfg.touching_threshold_m:.4f}m, "
            f"barely={cfg.barely_threshold_m:.4f}m, "
            f"pen_probe={cfg.penetration_probe_distance_m:.4f}m, "
            f"pen_min={cfg.penetration_min_depth_m:.4f}m, "
            f"self_mode={cfg.self_penetration_mode}, "
            f"self_pen={cfg.self_penetration_threshold_m:.4f}m, "
            f"self_dot<={cfg.self_penetration_normal_dot_max:.2f}"
        )
        print(f"Analyzing {len(frame_indices)} frames...")

    analyzer = MeshContactAnalyzer(faces=faces, config=cfg)
    frame_summaries, frame_details = analyzer.analyze(
        vertices_p1=v1,
        vertices_p2=v2,
        frame_indices=frame_indices,
        verbose_every=0 if args.quiet else 50,
    )

    output_base = os.path.join(PROJECT_ROOT, "output", "mesh_contact")
    os.makedirs(output_base, exist_ok=True)

    if args.output_json is None:
        args.output_json = os.path.join(output_base, f"{args.dataset}_{args.clip}_mesh_contact.json")

    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "clip": args.clip,
        "data_root": args.data_root,
        "body_model_path": args.body_model_path,
        "device": args.device,
        "frame_range": {
            "start": int(frame_start),
            "end_exclusive": int(frame_end),
            "step": int(args.frame_step),
            "num_frames": int(len(frame_indices)),
        },
        "thresholds_m": {
            "touching": cfg.touching_threshold_m,
            "barely": cfg.barely_threshold_m,
            "penetration_probe": cfg.penetration_probe_distance_m,
            "penetration_min_depth": cfg.penetration_min_depth_m,
            "self_penetration_mode": cfg.self_penetration_mode,
            "self_penetration_threshold": cfg.self_penetration_threshold_m,
            "self_penetration_k": int(cfg.self_penetration_k),
            "self_penetration_normal_dot_max": float(cfg.self_penetration_normal_dot_max),
        },
        "summary": _build_status_summary(frame_summaries),
        "frames": frame_summaries,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    if args.output_details is not None:
        os.makedirs(os.path.dirname(args.output_details), exist_ok=True)
        detail_payload = {
            "dataset": args.dataset,
            "clip": args.clip,
            "frame_indices": frame_indices,
            "faces": faces,
            "frame_details": frame_details,
        }
        with open(args.output_details, "wb") as f:
            pickle.dump(detail_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    exported = None
    if args.export_ply_frame is not None:
        frame_value = int(args.export_ply_frame)
        idx_in_eval = np.where(frame_indices == frame_value)[0]
        if idx_in_eval.size == 0:
            raise ValueError(
                f"--export-ply-frame={frame_value} is outside evaluated frame set "
                f"[{frame_start}, {frame_end}) step={args.frame_step}"
            )
        local_idx = int(idx_in_eval[0])
        if args.export_ply_dir is None:
            args.export_ply_dir = os.path.join(output_base, f"{args.dataset}_{args.clip}_ply")
        exported = export_contact_frame_ply(
            output_dir=args.export_ply_dir,
            frame_index=frame_value,
            vertices_p1=v1[frame_value],
            vertices_p2=v2[frame_value],
            faces=faces,
            detail=frame_details[local_idx],
        )

    if not args.quiet:
        print(f"Saved summary JSON: {args.output_json}")
        if args.output_details:
            print(f"Saved detailed contact arrays: {args.output_details}")
        if exported is not None:
            print(f"Exported PLY person1: {exported['person1']}")
            print(f"Exported PLY person2: {exported['person2']}")


if __name__ == "__main__":
    main()
