"""
Newton-native skyhook visualizer.

This viewer replays the same trajectory family used for metric computation
(joint_q on the per-subject Newton model) and overlays per-frame metric values.

No matplotlib is used.

Examples:
  # Full playback on clip 1000/person0
  python prepare2/visualize_skyhook_newton.py \
      --dataset interhuman --clip 1000 --person 0 \
      --metrics-dir data/test/skyhook

  # Compare low-force vs high-force states (alternating in Newton viewer)
  python prepare2/visualize_skyhook_newton.py \
      --dataset interhuman --clip 1000 --person 0 \
      --metrics-dir data/test/skyhook \
      --mode compare --compare-dwell 45
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare2.retarget import (  # noqa: E402
    get_or_create_xml,
    load_interhuman_clip,
    load_interx_clip,
    smplx_to_joint_q,
)


def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _default_raw_dir(dataset: str) -> Path:
    if dataset == "interhuman":
        return PROJECT_ROOT / "data" / "InterHuman"
    return PROJECT_ROOT / "data" / "Inter-X_Dataset"


def _default_retarget_dir(dataset: str) -> Path:
    return PROJECT_ROOT / "data" / "retargeted_v2" / dataset


def _load_metrics_npz(metrics_dir: Path, clip: str, person: int):
    npz_path = metrics_dir / f"{clip}_person{person}_skyhook_metrics.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {npz_path}")
    with np.load(npz_path) as d:
        frame = np.asarray(d["frame"]).astype(np.int32)
        root_force_l2 = np.asarray(d["root_force_l2"]).astype(np.float32)
        if "root_force_xyz_N" in d:
            root_force_xyz = np.asarray(d["root_force_xyz_N"]).astype(np.float32)
        elif "root_force_xyz" in d:
            root_force_xyz = np.asarray(d["root_force_xyz"]).astype(np.float32)
        else:
            raise KeyError(
                f"Missing root_force_xyz_N in {npz_path}. "
                "Recompute skyhook metrics to store 3D residual force vectors."
            )
        mpjpe = np.asarray(d["mpjpe_per_frame_m"]).astype(np.float32)
        joint_q_used = (
            np.asarray(d["joint_q_used"]).astype(np.float32)
            if "joint_q_used" in d
            else None
        )
        betas = (
            np.asarray(d["betas"]).astype(np.float64)
            if "betas" in d
            else None
        )
    return frame, root_force_l2, mpjpe, joint_q_used, betas, root_force_xyz


def _align_to_metric_frames(arr: np.ndarray, frame_idx: np.ndarray, target_len: int) -> np.ndarray:
    src_len = int(arr.shape[0])
    fi = np.asarray(frame_idx, dtype=np.int64)

    if fi.size >= target_len:
        fi = fi[:target_len]
    if fi.size == target_len and np.all(fi >= 0) and np.all(fi < src_len) and np.all(np.diff(fi) >= 0):
        return arr[fi]

    if src_len == target_len:
        return arr
    if src_len > target_len and src_len % target_len == 0:
        stride = src_len // target_len
        return arr[::stride][:target_len]

    idx = np.linspace(0, src_len - 1, target_len).astype(np.int64)
    return arr[idx]


def _load_joint_q_and_betas(
    *,
    dataset: str,
    clip: str,
    person: int,
    retarget_dir: Path,
    raw_data_dir: Optional[Path],
):
    # Preferred: exact retargeted outputs used in pipeline.
    jq_path = retarget_dir / f"{clip}_person{person}_joint_q.npy"
    beta_path = retarget_dir / f"{clip}_person{person}_betas.npy"
    if jq_path.exists() and beta_path.exists():
        joint_q = np.load(jq_path).astype(np.float32)
        betas = np.load(beta_path).astype(np.float64)
        return joint_q, betas, "retargeted_joint_q"

    # Fallback: convert from raw dataset SMPL-X params.
    if raw_data_dir is None:
        raise FileNotFoundError(
            f"Missing retargeted joint_q/betas for clip={clip}, person={person}, and no raw_data_dir provided."
        )

    if dataset == "interhuman":
        raw = load_interhuman_clip(str(raw_data_dir), clip)
    else:
        raw = load_interx_clip(str(raw_data_dir), clip)
    if raw is None or person >= len(raw):
        raise FileNotFoundError(f"Raw clip not found for dataset={dataset}, clip={clip}")

    pdata = raw[person]
    joint_q = smplx_to_joint_q(
        pdata["root_orient"], pdata["pose_body"], pdata["trans"], pdata["betas"]
    ).astype(np.float32)
    betas = np.asarray(pdata["betas"], dtype=np.float64)
    return joint_q, betas, "dataset_to_joint_q"


def _build_model_from_betas(betas: np.ndarray, device: str):
    return _build_model_from_betas_list([betas], device=device)


def _build_model_from_betas_list(betas_list: List[np.ndarray], device: str):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    for betas in betas_list:
        xml_path = get_or_create_xml(
            betas, cache_dir=str(PROJECT_ROOT / "prepare2" / "xml_cache")
        )
        if hasattr(builder, "add_mjcf"):
            builder.add_mjcf(str(xml_path), enable_self_collisions=False)
        else:
            newton.utils.parse_mjcf(
                str(xml_path),
                builder,
                enable_self_collisions=False,
                up_axis=newton.Axis.Z,
                parse_visuals_as_colliders=False,
            )
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _load_person_visual_inputs(
    *,
    dataset: str,
    clip: str,
    person: int,
    metrics_dir: Path,
    retarget_dir: Path,
    raw_data_dir: Optional[Path],
) -> Dict[str, Any]:
    frame_idx, root_force_l2, mpjpe, joint_q_used, betas_from_metrics, root_force_xyz = _load_metrics_npz(
        metrics_dir, clip, person
    )

    T = min(len(frame_idx), len(root_force_l2), len(root_force_xyz), len(mpjpe))
    frame_idx = frame_idx[:T]
    root_force_l2 = root_force_l2[:T]
    root_force_xyz = root_force_xyz[:T]
    mpjpe = mpjpe[:T]

    if joint_q_used is not None and betas_from_metrics is not None:
        T = min(T, int(joint_q_used.shape[0]))
        frame_idx = frame_idx[:T]
        root_force_l2 = root_force_l2[:T]
        root_force_xyz = root_force_xyz[:T]
        mpjpe = mpjpe[:T]
        joint_q = joint_q_used[:T]
        betas = betas_from_metrics
        joint_source = "metrics_joint_q_used"
    else:
        joint_q_raw, betas, joint_source = _load_joint_q_and_betas(
            dataset=dataset,
            clip=clip,
            person=person,
            retarget_dir=retarget_dir,
            raw_data_dir=raw_data_dir,
        )
        joint_q = _align_to_metric_frames(joint_q_raw, frame_idx, T)

    return {
        "person": int(person),
        "frame_idx": frame_idx.astype(np.int32),
        "root_force_l2": root_force_l2.astype(np.float32),
        "root_force_xyz": root_force_xyz.astype(np.float32),
        "mpjpe": mpjpe.astype(np.float32),
        "joint_q": joint_q.astype(np.float32),
        "betas": betas.astype(np.float64),
        "joint_source": joint_source,
    }


class SkyhookNewtonVisualizer:
    """Newton GL/USD viewer for skyhook diagnostics on one or both persons."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device if args.device else "cuda:0"
        self.mode = args.mode
        self.compare_dwell = max(1, int(args.compare_dwell))
        self.force_scale = float(args.force_scale)
        self._wall_start = None
        self.sim_time = 0.0

        metrics_dir = _resolve(args.metrics_dir)
        retarget_dir = (
            _resolve(args.retarget_dir)
            if args.retarget_dir
            else _default_retarget_dir(args.dataset)
        )
        raw_data_dir = (
            _resolve(args.raw_data_dir)
            if args.raw_data_dir
            else _default_raw_dir(args.dataset)
        )
        if not raw_data_dir.exists():
            raw_data_dir = None

        self.clip = args.clip
        self.person_arg = str(args.person)
        self.person_ids = [0, 1] if self.person_arg == "both" else [int(self.person_arg)]
        self.n_persons = len(self.person_ids)

        per_person = []
        for p in self.person_ids:
            data = _load_person_visual_inputs(
                dataset=args.dataset,
                clip=self.clip,
                person=p,
                metrics_dir=metrics_dir,
                retarget_dir=retarget_dir,
                raw_data_dir=raw_data_dir,
            )
            per_person.append(data)

        self.T = int(min(d["joint_q"].shape[0] for d in per_person))
        self.n_per = int(per_person[0]["joint_q"].shape[1])
        self.metric_frame = np.stack([d["frame_idx"][: self.T] for d in per_person], axis=0)
        self.root_force_l2 = np.stack([d["root_force_l2"][: self.T] for d in per_person], axis=0)
        self.root_force_xyz = np.stack([d["root_force_xyz"][: self.T] for d in per_person], axis=0)
        self.mpjpe = np.stack([d["mpjpe"][: self.T] for d in per_person], axis=0)
        self.joint_q = np.stack([d["joint_q"][: self.T] for d in per_person], axis=0)
        self.betas = [d["betas"] for d in per_person]
        self.joint_sources = [d["joint_source"] for d in per_person]

        base_colors = np.array([[1.0, 0.1, 0.1], [1.0, 0.55, 0.1]], dtype=np.float32)
        self.force_line_colors = wp.array(
            base_colors[: self.n_persons], dtype=wp.vec3, device=self.device
        )

        self.low_idx = np.argmin(self.root_force_l2, axis=1).astype(np.int32)
        self.high_idx = np.argmax(self.root_force_l2, axis=1).astype(np.int32)
        self.root_force_l2_mean = np.mean(self.root_force_l2, axis=0)
        self.low_idx_agg = int(np.argmin(self.root_force_l2_mean))
        self.high_idx_agg = int(np.argmax(self.root_force_l2_mean))
        self.fixed_frame = int(np.clip(args.frame, 0, self.T - 1))

        self.model = _build_model_from_betas_list(self.betas, self.device)
        self.state = self.model.state()
        self.jqd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.device)
        self.viewer.set_model(self.model)

        self.frame = 0
        self._set_frame(0)
        self._setup_camera()

        who = "both" if self.n_persons > 1 else f"person={self.person_ids[0]}"
        print(f"Loaded clip={self.clip}, {who}, frames={self.T}")
        for i, p in enumerate(self.person_ids):
            print(
                f"  p{p}: source={self.joint_sources[i]}, "
                f"low={int(self.low_idx[i])} ({float(self.root_force_l2[i, self.low_idx[i]]):.3e} N), "
                f"high={int(self.high_idx[i])} ({float(self.root_force_l2[i, self.high_idx[i]]):.3e} N)"
            )
        print(
            f"  aggregate low/high: {self.low_idx_agg}/{self.high_idx_agg} "
            f"({float(self.root_force_l2_mean[self.low_idx_agg]):.3e} / "
            f"{float(self.root_force_l2_mean[self.high_idx_agg]):.3e} N)"
        )
        print(f"Mode: {self.mode}")

    def _setup_camera(self):
        root_center = np.mean(self.joint_q[:, 0, :3], axis=0)
        cam_dist = 5.0
        cam_pos = wp.vec3(float(root_center[0]), float(root_center[1]) - cam_dist, 2.0)
        self.viewer.set_camera(cam_pos, pitch=-15.0, yaw=90.0)

    def _set_frame(self, frame_idx: int):
        f = int(np.clip(frame_idx, 0, self.T - 1))
        self.frame = f
        combined_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        for i in range(self.n_persons):
            s = i * self.n_per
            combined_q[s:s + self.n_per] = self.joint_q[i, f]
        self.state.joint_q = wp.array(combined_q, dtype=wp.float32, device=self.device)
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def _select_frame(self):
        if self.mode == "high":
            return self.high_idx_agg
        if self.mode == "low":
            return self.low_idx_agg
        if self.mode == "frame":
            return self.fixed_frame
        if self.mode == "compare":
            tick = int(self.sim_time * 30.0)
            block = (tick // self.compare_dwell) % 2
            return self.low_idx_agg if block == 0 else self.high_idx_agg
        return int(self.sim_time * 30.0) % self.T

    def step(self):
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        self.sim_time = now - self._wall_start
        self._set_frame(self._select_frame())

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        starts_np = self.joint_q[:, self.frame, 0:3].astype(np.float32, copy=False)
        force_np = self.root_force_xyz[:, self.frame, :].astype(np.float32, copy=False)
        ends_np = starts_np + force_np * self.force_scale
        self.viewer.log_lines(
            "skyhook_force_vectors",
            wp.array(starts_np, dtype=wp.vec3, device=self.device),
            wp.array(ends_np, dtype=wp.vec3, device=self.device),
            self.force_line_colors,
            width=0.012,
            hidden=False,
        )
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(0.5, 0.9, 1.0, 1.0), "[ SKYHOOK NEWTON VIEW ]")
        imgui.separator()
        imgui.text(f"Clip:       {self.clip}")
        imgui.text(f"Person(s):  {self.person_arg}")
        imgui.text(f"Mode:       {self.mode}")
        imgui.text(f"Frame:      {self.frame}/{self.T - 1}")

        imgui.separator()
        for i, p in enumerate(self.person_ids):
            force = float(self.root_force_l2[i, self.frame])
            force_xyz = self.root_force_xyz[i, self.frame]
            mp = float(self.mpjpe[i, self.frame]) if np.isfinite(self.mpjpe[i, self.frame]) else float("nan")
            tag = ""
            if self.frame == int(self.high_idx[i]):
                tag = " [HIGH]"
            elif self.frame == int(self.low_idx[i]):
                tag = " [LOW]"
            imgui.text(f"p{p} |F_root|: {force:.3e} N{tag}")
            imgui.text(
                f"p{p} F_root xyz: [{force_xyz[0]:.2e}, {force_xyz[1]:.2e}, {force_xyz[2]:.2e}] N"
            )
            if np.isfinite(mp):
                imgui.text(f"p{p} MPJPE:   {mp * 100.0:.3f} cm")
            else:
                imgui.text(f"p{p} MPJPE:   n/a")

        imgui.separator()
        imgui.text(
            f"Aggregate low/high frame: {self.low_idx_agg}/{self.high_idx_agg}"
        )
        imgui.text(
            f"Aggregate |F_root| mean:  "
            f"{float(self.root_force_l2_mean[self.frame]):.3e} N"
        )
        if self.mode == "compare":
            imgui.text(f"Compare dwell: {self.compare_dwell} ticks")
        imgui.text(f"Force scale: {self.force_scale:.3e} m/N")


def build_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--dataset", required=True, choices=["interhuman", "interx"])
    parser.add_argument("--clip", default="1000")
    parser.add_argument(
        "--person",
        type=str,
        default="0",
        choices=["0", "1", "both"],
        help="Visualize one person or both in the same scene.",
    )
    parser.add_argument("--metrics-dir", default="data/test/skyhook")
    parser.add_argument("--retarget-dir", default=None)
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument(
        "--mode",
        default="playback",
        choices=["playback", "compare", "high", "low", "frame"],
        help="Visualization mode: full playback, low/high compare, or fixed frame.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Used when --mode frame")
    parser.add_argument(
        "--compare-dwell",
        type=int,
        default=45,
        help="Ticks to stay on each pose in compare mode.",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=0.001,
        help="Visual scale for root_force_xyz_N arrows (meters per Newton).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    viewer, args = newton.examples.init(parser)
    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda:0"
    example = SkyhookNewtonVisualizer(viewer, args)
    # Newton API compatibility:
    # - some versions expose run(example)
    # - others expose run(example, args)
    try:
        newton.examples.run(example, args)
    except TypeError:
        newton.examples.run(example)
