#!/usr/bin/env python3
"""Newton GUI visualization for mesh-contact points on InterHuman/InterX clips.

Overlays per-frame:
- contact points (penetrating / touching / barely touching)
- closest-point line
- SMPL 22-joint skeleton (joints + bones)
- clip text annotation
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import warp as wp

import newton
import newton.examples

try:
    import torch
except Exception as exc:
    raise RuntimeError("PyTorch is required for SMPL-X mesh reconstruction") from exc


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.body_model.body_model import BodyModel  # noqa: E402
from prepare_mesh_contact.mesh_contact_pipeline import (  # noqa: E402
    ContactConfig,
    MeshContactAnalyzer,
    load_interhuman_clip,
    load_interx_clip,
)

# SMPL-X 22-joint chain
SMPL_BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]


def _sample_indices(idx: np.ndarray, max_n: int) -> np.ndarray:
    if idx.size <= max_n:
        return idx
    sel = np.linspace(0, idx.size - 1, num=max_n, dtype=np.int32)
    return idx[sel]


def _load_clip_text(dataset: str, data_root: str, clip: str) -> List[str]:
    candidates = []
    if dataset == "interhuman":
        candidates += [
            os.path.join(data_root, "annots", f"{clip}.txt"),
            os.path.join(data_root, f"{clip}.txt"),
        ]
    else:
        candidates += [
            os.path.join(data_root, f"{clip}.txt"),
            os.path.join(data_root, "texts", f"{clip}.txt"),
            os.path.join(data_root, "texts_processed", f"{clip}.txt"),
        ]

    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                return lines
    return ["[No annotation text found]"]


def _reconstruct_mesh_and_joints(
    persons: Sequence[Dict[str, np.ndarray]],
    body_model_path: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return v1, v2, j1, j2, faces for first two persons."""
    if len(persons) < 2:
        raise ValueError("Need two persons")
    if not os.path.isfile(body_model_path):
        raise FileNotFoundError(f"SMPL-X model not found: {body_model_path}")

    torch_device = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    bm = BodyModel(
        bm_fname=body_model_path,
        num_betas=10,
        num_expressions=10,
        dtype=torch.float32,
    ).to(torch_device)
    bm.eval()

    faces = bm.f.detach().cpu().numpy().astype(np.int32)

    verts_all: List[np.ndarray] = []
    joints_all: List[np.ndarray] = []

    for p in persons[:2]:
        trans = np.asarray(p["trans"], dtype=np.float32)
        root = np.asarray(p["root_orient"], dtype=np.float32)
        body = np.asarray(p["pose_body"], dtype=np.float32)
        hand = np.asarray(p.get("pose_hand", np.zeros((trans.shape[0], 90), dtype=np.float32)), dtype=np.float32)
        betas_np = np.asarray(p["betas"], dtype=np.float32)

        t_len = trans.shape[0]
        if root.shape != (t_len, 3) or body.shape != (t_len, 63):
            raise ValueError(f"Invalid shapes for {p.get('name','person')}: trans={trans.shape}, root={root.shape}, body={body.shape}")

        t_trans = torch.from_numpy(trans).to(torch_device)
        t_root = torch.from_numpy(root).to(torch_device)
        t_body = torch.from_numpy(body).to(torch_device)
        t_hand = torch.from_numpy(hand).to(torch_device)

        if betas_np.ndim == 1:
            t_betas = torch.from_numpy(betas_np[None, :]).to(torch_device)
            per_frame_betas = False
        elif betas_np.ndim == 2 and betas_np.shape[0] == t_len:
            t_betas = torch.from_numpy(betas_np).to(torch_device)
            per_frame_betas = True
        else:
            t_betas = torch.from_numpy(betas_np.reshape(1, -1)).to(torch_device)
            per_frame_betas = False

        v_batches = []
        j_batches = []
        for s in range(0, t_len, batch_size):
            e = min(s + batch_size, t_len)
            bs = e - s
            b_batch = t_betas[s:e] if per_frame_betas else t_betas.expand(bs, -1)
            with torch.no_grad():
                out = bm(
                    root_orient=t_root[s:e],
                    pose_body=t_body[s:e],
                    pose_hand=t_hand[s:e],
                    betas=b_batch,
                    trans=t_trans[s:e],
                )
            v_batches.append(out.v.detach().cpu().numpy().astype(np.float32))
            j_batches.append(out.Jtr[:, :22, :].detach().cpu().numpy().astype(np.float32))

        verts_all.append(np.concatenate(v_batches, axis=0))
        joints_all.append(np.concatenate(j_batches, axis=0))

    v1, v2 = verts_all
    j1, j2 = joints_all
    t_common = min(v1.shape[0], v2.shape[0], j1.shape[0], j2.shape[0])
    return v1[:t_common], v2[:t_common], j1[:t_common], j2[:t_common], faces


class MeshContactNewtonVis:
    """GUI visualizer for mesh contacts + skeleton + text."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.dataset = args.dataset
        self.clip = args.clip
        self._wall_start = None
        self.time = 0.0

        requested = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = wp.get_device(requested)
        self.torch_device = "cuda" if ("cuda" in str(self.device) and torch.cuda.is_available()) else "cpu"

        if args.data_root is None:
            args.data_root = (
                os.path.join(PROJECT_ROOT, "data", "InterHuman")
                if args.dataset == "interhuman"
                else os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset")
            )

        print(f"Dataset: {args.dataset}")
        print(f"Clip: {args.clip}")
        print(f"Data root: {args.data_root}")
        print(f"Warp device: {self.device}")
        print(f"Torch device: {self.torch_device}")

        self.text_lines = _load_clip_text(args.dataset, args.data_root, args.clip)
        print("Annotation:")
        for i, line in enumerate(self.text_lines[:5]):
            print(f"  [{i+1}] {line}")
        if len(self.text_lines) > 5:
            print(f"  ... (+{len(self.text_lines)-5} more lines)")

        if args.dataset == "interhuman":
            persons = load_interhuman_clip(args.data_root, args.clip)
        else:
            persons = load_interx_clip(
                args.data_root,
                args.clip,
                h5_file=args.h5_file,
                convert_to_zup=args.convert_interx_to_zup,
            )

        self.vertices_p1, self.vertices_p2, self.joints_p1, self.joints_p2, self.faces = _reconstruct_mesh_and_joints(
            persons,
            body_model_path=args.body_model_path,
            device=self.torch_device,
            batch_size=args.batch_size,
        )

        # Optional grounding for readability
        if args.auto_ground:
            min_z = min(
                float(self.vertices_p1[:, :, 2].min()),
                float(self.vertices_p2[:, :, 2].min()),
                float(self.joints_p1[:, :, 2].min()),
                float(self.joints_p2[:, :, 2].min()),
            )
            self.ground_offset = -min_z + 0.005
            self.vertices_p1[:, :, 2] += self.ground_offset
            self.vertices_p2[:, :, 2] += self.ground_offset
            self.joints_p1[:, :, 2] += self.ground_offset
            self.joints_p2[:, :, 2] += self.ground_offset
        else:
            self.ground_offset = 0.0

        t_len = min(self.vertices_p1.shape[0], self.vertices_p2.shape[0])
        frame_start = max(0, args.frame_start)
        frame_end = t_len if args.frame_end is None else min(t_len, args.frame_end)
        if frame_start >= frame_end:
            raise ValueError(f"Invalid frame range [{frame_start}, {frame_end})")
        self.frames = np.arange(frame_start, frame_end, max(1, args.frame_step), dtype=np.int32)
        self.num_play_frames = int(self.frames.shape[0])
        self.play_fps = float(args.play_fps if args.play_fps is not None else args.fps)

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
        self.analyzer = MeshContactAnalyzer(self.faces, cfg)

        print(
            "Computing contact points for all playback frames...\n"
            f"  touch={cfg.touching_threshold_m:.4f}m "
            f"barely={cfg.barely_threshold_m:.4f}m "
            f"pen_probe={cfg.penetration_probe_distance_m:.4f}m "
            f"pen_min={cfg.penetration_min_depth_m:.4f}m "
            f"self_mode={cfg.self_penetration_mode} "
            f"self_pen={cfg.self_penetration_threshold_m:.4f}m "
            f"self_dot<={cfg.self_penetration_normal_dot_max:.2f}"
        )

        self.frame_summaries: List[Dict[str, object]] = []
        self.frame_points: List[Dict[str, np.ndarray]] = []
        for i, frame_idx in enumerate(self.frames):
            summary, detail = self.analyzer._analyze_frame(
                self.vertices_p1[frame_idx],
                self.vertices_p2[frame_idx],
            )
            summary["frame"] = int(frame_idx)
            self.frame_summaries.append(summary)
            self.frame_points.append(
                self._build_frame_points(
                    self.vertices_p1[frame_idx],
                    self.vertices_p2[frame_idx],
                    summary,
                    detail,
                    max_points_per_set=args.max_points_per_set,
                )
            )
            if (i + 1) % 50 == 0:
                print(f"  analyzed {i + 1}/{self.num_play_frames} frames")

        counts = {
            "penetrating": sum(1 for s in self.frame_summaries if s["status"] == "penetrating"),
            "touching": sum(1 for s in self.frame_summaries if s["status"] == "touching"),
            "barely_touching": sum(1 for s in self.frame_summaries if s["status"] == "barely_touching"),
            "not_touching": sum(1 for s in self.frame_summaries if s["status"] == "not_touching"),
        }
        print(f"Contact status counts: {counts}")

        if args.dry_run:
            print("Dry-run requested; skipping GUI startup.")
            self._dry_run = True
            return

        self._dry_run = False
        self._setup_scene()
        self.current_play_idx = 0
        self.current_frame = int(self.frames[0])
        self._set_play_frame(0)
        self._setup_camera()

        print(
            f"Ready: {self.num_play_frames} frames @ {self.play_fps:.1f} fps "
            f"(frame range {self.frames[0]}..{self.frames[-1]})"
        )
        print("Close viewer window to continue.")

    @staticmethod
    def _build_frame_points(
        verts1: np.ndarray,
        verts2: np.ndarray,
        summary: Dict[str, object],
        detail: Dict[str, np.ndarray],
        max_points_per_set: int,
    ) -> Dict[str, np.ndarray]:
        inter_pen1 = detail.get("inter_person_penetrating_vidx_p1", detail["penetrating_vidx_p1"])
        inter_pen2 = detail.get("inter_person_penetrating_vidx_p2", detail["penetrating_vidx_p2"])
        self_pen1 = detail.get("self_penetrating_vidx_p1", np.zeros((0,), dtype=np.int32))
        self_pen2 = detail.get("self_penetrating_vidx_p2", np.zeros((0,), dtype=np.int32))

        touch1 = np.setdiff1d(detail["contact_vidx_p1"], inter_pen1, assume_unique=False)
        touch2 = np.setdiff1d(detail["contact_vidx_p2"], inter_pen2, assume_unique=False)
        barely1 = detail["barely_vidx_p1"]
        barely2 = detail["barely_vidx_p2"]

        inter_pen1 = _sample_indices(inter_pen1.astype(np.int32), max_points_per_set)
        inter_pen2 = _sample_indices(inter_pen2.astype(np.int32), max_points_per_set)
        self_pen1 = _sample_indices(self_pen1.astype(np.int32), max_points_per_set)
        self_pen2 = _sample_indices(self_pen2.astype(np.int32), max_points_per_set)
        touch1 = _sample_indices(touch1.astype(np.int32), max_points_per_set)
        touch2 = _sample_indices(touch2.astype(np.int32), max_points_per_set)
        barely1 = _sample_indices(barely1.astype(np.int32), max_points_per_set)
        barely2 = _sample_indices(barely2.astype(np.int32), max_points_per_set)

        closest_line = np.array(
            [
                np.asarray(summary["closest_point_p1"], dtype=np.float32),
                np.asarray(summary["closest_point_p2"], dtype=np.float32),
            ],
            dtype=np.float32,
        )

        return {
            "p1_inter_pen": verts1[inter_pen1] if inter_pen1.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p2_inter_pen": verts2[inter_pen2] if inter_pen2.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p1_self_pen": verts1[self_pen1] if self_pen1.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p2_self_pen": verts2[self_pen2] if self_pen2.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p1_touch": verts1[touch1] if touch1.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p2_touch": verts2[touch2] if touch2.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p1_barely": verts1[barely1] if barely1.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "p2_barely": verts2[barely2] if barely2.size > 0 else np.zeros((0, 3), dtype=np.float32),
            "closest_line": closest_line,
        }

    def _setup_scene(self) -> None:
        builder = newton.ModelBuilder()
        builder.gravity = 0.0
        ground = builder.add_body()
        builder.add_shape_plane(
            body=ground,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            width=10.0,
            length=10.0,
        )
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.viewer.set_model(self.model)

        self.faces_wp = wp.from_numpy(self.faces.astype(np.int32).reshape(-1), dtype=int, device=self.device)
        self.vertices_p1_wp = wp.from_numpy(self.vertices_p1.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.vertices_p2_wp = wp.from_numpy(self.vertices_p2.astype(np.float32), dtype=wp.vec3, device=self.device)

        self.identity_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self.ones_scale = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=self.device)
        self.mat_default = wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self.device)
        self.col_p1 = wp.array([wp.vec3(1.0, 0.2, 0.2)], dtype=wp.vec3, device=self.device)
        self.col_p2 = wp.array([wp.vec3(0.2, 0.6, 1.0)], dtype=wp.vec3, device=self.device)
        self.ground_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)

        first = int(self.frames[0])
        self.viewer.log_mesh("/meshes/person_1", self.vertices_p1_wp[first], self.faces_wp, hidden=True)
        self.viewer.log_mesh("/meshes/person_2", self.vertices_p2_wp[first], self.faces_wp, hidden=True)
        self.viewer.log_instances("/person_1", "/meshes/person_1", self.identity_xform, self.ones_scale, self.col_p1, self.mat_default)
        self.viewer.log_instances("/person_2", "/meshes/person_2", self.identity_xform, self.ones_scale, self.col_p2, self.mat_default)

    def _setup_camera(self) -> None:
        center = (self.vertices_p1[self.frames[0]].mean(axis=0) + self.vertices_p2[self.frames[0]].mean(axis=0)) * 0.5
        cam_pos = wp.vec3(float(center[0]) + 3.2, float(center[1]), float(center[2]) + 1.2)
        self.viewer.set_camera(cam_pos, -12.0, 180.0)

    def _log_points(self, name: str, points: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
        if points is None or points.shape[0] == 0:
            self.viewer.log_points(name, None, None, None, hidden=True)
            return
        pts_wp = wp.array(points.astype(np.float32), dtype=wp.vec3, device=self.device)
        radii_wp = wp.array(np.full(points.shape[0], radius, dtype=np.float32), dtype=wp.float32, device=self.device)
        colors_wp = wp.full(points.shape[0], wp.vec3(*color), dtype=wp.vec3, device=self.device)
        self.viewer.log_points(name, pts_wp, radii_wp, colors_wp, hidden=False)

    def _log_skeleton(self, prefix: str, joints: np.ndarray, joint_color: tuple[float, float, float], line_color: tuple[float, float, float]) -> None:
        self._log_points(f"{prefix}/joints", joints, self.args.skeleton_joint_radius, joint_color)
        starts = np.array([joints[i] for i, j in SMPL_BONES if i < joints.shape[0] and j < joints.shape[0]], dtype=np.float32)
        ends = np.array([joints[j] for i, j in SMPL_BONES if i < joints.shape[0] and j < joints.shape[0]], dtype=np.float32)
        if starts.shape[0] == 0:
            self.viewer.log_lines(f"{prefix}/bones", None, None, None, hidden=True)
            return
        self.viewer.log_lines(
            f"{prefix}/bones",
            wp.array(starts, dtype=wp.vec3, device=self.device),
            wp.array(ends, dtype=wp.vec3, device=self.device),
            line_color,
            width=self.args.skeleton_line_width,
            hidden=False,
        )

    def _set_play_frame(self, play_idx: int) -> None:
        frame_idx = int(self.frames[play_idx])
        self.current_play_idx = play_idx
        self.current_frame = frame_idx

        self.viewer.log_mesh("/meshes/person_1", self.vertices_p1_wp[frame_idx], self.faces_wp, hidden=True)
        self.viewer.log_mesh("/meshes/person_2", self.vertices_p2_wp[frame_idx], self.faces_wp, hidden=True)

        if self.args.show_skeleton:
            self._log_skeleton("/skeleton/p1", self.joints_p1[frame_idx], (0.95, 0.4, 0.4), (1.0, 0.7, 0.7))
            self._log_skeleton("/skeleton/p2", self.joints_p2[frame_idx], (0.4, 0.7, 0.95), (0.7, 0.85, 1.0))
        else:
            self.viewer.log_points("/skeleton/p1/joints", None, None, None, hidden=True)
            self.viewer.log_points("/skeleton/p2/joints", None, None, None, hidden=True)
            self.viewer.log_lines("/skeleton/p1/bones", None, None, None, hidden=True)
            self.viewer.log_lines("/skeleton/p2/bones", None, None, None, hidden=True)

        fp = self.frame_points[play_idx]
        self._log_points("/contact/p1_inter_pen", fp["p1_inter_pen"], self.args.radius_pen, (0.75, 0.2, 1.0))
        self._log_points("/contact/p2_inter_pen", fp["p2_inter_pen"], self.args.radius_pen, (0.75, 0.2, 1.0))
        self._log_points("/contact/p1_self_pen", fp["p1_self_pen"], self.args.radius_self_pen, (0.2, 0.9, 0.45))
        self._log_points("/contact/p2_self_pen", fp["p2_self_pen"], self.args.radius_self_pen, (0.2, 0.9, 0.45))
        self._log_points("/contact/p1_touch", fp["p1_touch"], self.args.radius_touch, (1.0, 0.1, 0.1))
        self._log_points("/contact/p2_touch", fp["p2_touch"], self.args.radius_touch, (1.0, 0.1, 0.1))
        self._log_points("/contact/p1_barely", fp["p1_barely"], self.args.radius_barely, (1.0, 0.7, 0.2))
        self._log_points("/contact/p2_barely", fp["p2_barely"], self.args.radius_barely, (1.0, 0.7, 0.2))

        closest = fp["closest_line"]
        if closest.shape[0] == 2:
            line_color = {
                "penetrating": (0.75, 0.2, 1.0),
                "touching": (1.0, 0.1, 0.1),
                "barely_touching": (1.0, 0.7, 0.2),
                "not_touching": (0.8, 0.8, 0.8),
            }[self.frame_summaries[play_idx]["status"]]
            starts = wp.array(closest[0:1], dtype=wp.vec3, device=self.device)
            ends = wp.array(closest[1:2], dtype=wp.vec3, device=self.device)
            self.viewer.log_lines("/contact/closest_line", starts, ends, line_color, width=self.args.closest_line_width, hidden=False)
        else:
            self.viewer.log_lines("/contact/closest_line", None, None, None, hidden=True)

    def step(self) -> None:
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        self.time = time.perf_counter() - self._wall_start
        play_idx = int(self.time * self.play_fps) % max(self.num_play_frames, 1)
        if play_idx != self.current_play_idx:
            self._set_play_frame(play_idx)

    def render(self) -> None:
        self.viewer.begin_frame(self.time)
        self.viewer.log_shapes(
            "/ground",
            newton.GeoType.PLANE,
            (10.0, 10.0),
            self.ground_xform,
            wp.array([wp.vec3(0.45, 0.45, 0.45)], dtype=wp.vec3, device=self.device),
            self.mat_default,
        )
        self.viewer.end_frame()

    def gui(self, imgui) -> None:
        s = self.frame_summaries[self.current_play_idx]
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(0.45, 0.95, 0.95, 1.0), "[ MESH CONTACT VIEW ]")
        imgui.separator()
        imgui.text(f"Dataset: {self.dataset}")
        imgui.text(f"Clip:    {self.clip}")
        imgui.text(f"Frame:   {self.current_frame} ({self.current_play_idx + 1}/{self.num_play_frames})")
        imgui.text(f"Status:  {s['status']}")
        imgui.text(f"Pen source: {s.get('penetration_source', 'n/a')}")
        imgui.text(f"Min dist: {100.0 * float(s['min_distance_m']):.2f} cm")
        imgui.text(f"Inter-pen depth: {100.0 * float(s.get('inter_person_penetration_depth_est_m', 0.0)):.2f} cm")
        imgui.text(f"Self-pen depth p1/p2: {100.0 * float(s.get('self_penetration_depth_est_m_p1', 0.0)):.2f}/{100.0 * float(s.get('self_penetration_depth_est_m_p2', 0.0)):.2f} cm")

        imgui.separator()
        imgui.text_colored(imgui.ImVec4(0.95, 0.9, 0.5, 1.0), "Text annotation:")
        for ln in self.text_lines[:4]:
            imgui.text_wrapped(ln)
        if len(self.text_lines) > 4:
            imgui.text(f"... (+{len(self.text_lines)-4} more)")


def parse_args():
    parser = newton.examples.create_parser()
    parser.add_argument("--dataset", choices=["interhuman", "interx"], default="interhuman")
    parser.add_argument("--clip", type=str, required=True, help="Clip id")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--h5-file", type=str, default=None, help="InterX optional explicit H5 file")
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback playback fps")
    parser.add_argument("--play-fps", type=float, default=None, help="Playback fps (overrides --fps)")

    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None, help="Exclusive frame end")
    parser.add_argument("--frame-step", type=int, default=1)

    parser.add_argument("--touching-threshold-m", type=float, default=0.005)
    parser.add_argument("--barely-threshold-m", type=float, default=0.020)
    parser.add_argument("--penetration-probe-distance-m", type=float, default=0.010)
    parser.add_argument("--penetration-min-depth-m", type=float, default=0.002)
    parser.add_argument("--self-penetration-mode", choices=["off", "heuristic"], default="off")
    parser.add_argument("--self-penetration-threshold-m", type=float, default=0.004)
    parser.add_argument("--self-penetration-k", type=int, default=12)
    parser.add_argument("--self-penetration-normal-dot-max", type=float, default=-0.2)
    parser.add_argument("--max-inside-queries", type=int, default=64)
    parser.add_argument("--max-points-per-set", type=int, default=300)

    parser.add_argument("--radius-pen", type=float, default=0.020)
    parser.add_argument("--radius-self-pen", type=float, default=0.018)
    parser.add_argument("--radius-touch", type=float, default=0.016)
    parser.add_argument("--radius-barely", type=float, default=0.012)
    parser.add_argument("--closest-line-width", type=float, default=0.012)

    parser.add_argument("--show-skeleton", action="store_true", default=True)
    parser.add_argument("--no-show-skeleton", dest="show_skeleton", action="store_false")
    parser.add_argument("--skeleton-joint-radius", type=float, default=0.010)
    parser.add_argument("--skeleton-line-width", type=float, default=0.008)

    parser.add_argument("--auto-ground", action="store_true", default=True)
    parser.add_argument("--no-auto-ground", dest="auto_ground", action="store_false")

    parser.add_argument("--convert-interx-to-zup", action="store_true", default=True)
    parser.add_argument("--no-convert-interx-to-zup", dest="convert_interx_to_zup", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Compute everything and exit without GUI")
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args_pre = parser.parse_known_args()[0]
    if args_pre.headless:
        import pyglet

        pyglet.options["headless"] = True

    try:
        viewer, args = newton.examples.init(parser)
    except Exception as exc:
        msg = str(exc)
        if "NoSuchDisplayException" in type(exc).__name__ or "Cannot connect to" in msg:
            print(
                "ERROR: Newton GL viewer could not connect to a display. "
                "Run inside a desktop/X11 session (with DISPLAY set), "
                "or use a non-GL backend such as --viewer viser for remote inspection.",
                file=sys.stderr,
            )
            sys.exit(2)
        raise

    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    example = MeshContactNewtonVis(viewer, args)
    if getattr(example, "_dry_run", False):
        try:
            viewer.close()
        except Exception:
            pass
        sys.exit(0)

    newton.examples.run(example, args)
