#!/usr/bin/env python3
"""Render static mesh-contact diagnostics without starting the Newton GUI."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.body_model.body_model import BodyModel  # noqa: E402
from prepare_mesh_contact.mesh_contact_pipeline import (  # noqa: E402
    _override_interhuman_betas_from_root,
    ContactConfig,
    MeshContactAnalyzer,
    load_interhuman_clip,
    load_interx_clip,
)

DEFAULT_BATCH_CLIPS = {
    "interhuman": ["7605", "1000"],
    "interx": ["G039T007A025R000", "G035T000A002R013"],
}

STATUS_ORDER = {
    "penetrating": 0,
    "touching": 1,
    "barely_touching": 2,
    "not_touching": 3,
}

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

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                return lines
    return ["[No annotation text found]"]


def _reconstruct_mesh_and_joints(
    persons: Sequence[Dict[str, np.ndarray]],
    body_model_path: str,
    device: str = "cpu",
    batch_size: int = 64,
):
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

    for person in persons[:2]:
        trans = np.asarray(person["trans"], dtype=np.float32)
        root = np.asarray(person["root_orient"], dtype=np.float32)
        body = np.asarray(person["pose_body"], dtype=np.float32)
        hand = np.asarray(person.get("pose_hand", np.zeros((trans.shape[0], 90), dtype=np.float32)), dtype=np.float32)
        betas_np = np.asarray(person["betas"], dtype=np.float32)

        t_len = trans.shape[0]
        if root.shape != (t_len, 3) or body.shape != (t_len, 63):
            raise ValueError(
                f"Invalid shapes for {person.get('name', 'person')}: "
                f"trans={trans.shape}, root={root.shape}, body={body.shape}"
            )

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
        for start in range(0, t_len, batch_size):
            end = min(start + batch_size, t_len)
            bs = end - start
            betas_batch = t_betas[start:end] if per_frame_betas else t_betas.expand(bs, -1)
            with torch.no_grad():
                out = bm(
                    root_orient=t_root[start:end],
                    pose_body=t_body[start:end],
                    pose_hand=t_hand[start:end],
                    betas=betas_batch,
                    trans=t_trans[start:end],
                )
            v_batches.append(out.v.detach().cpu().numpy().astype(np.float32))
            j_batches.append(out.Jtr[:, :22, :].detach().cpu().numpy().astype(np.float32))

        verts_all.append(np.concatenate(v_batches, axis=0))
        joints_all.append(np.concatenate(j_batches, axis=0))

    v1, v2 = verts_all
    j1, j2 = joints_all
    t_common = min(v1.shape[0], v2.shape[0], j1.shape[0], j2.shape[0])
    return v1[:t_common], v2[:t_common], j1[:t_common], j2[:t_common], faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render headless mesh-contact diagnostics")
    parser.add_argument("--dataset", choices=["interhuman", "interx"], default=None)
    parser.add_argument("--clip", type=str, default=None, help="Single clip to render")
    parser.add_argument(
        "--batch",
        choices=sorted(DEFAULT_BATCH_CLIPS.keys()),
        default=None,
        help="Use a small built-in sample set instead of passing --clip",
    )
    parser.add_argument("--clips-file", type=str, default=None, help="Optional newline-delimited clip id list")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional limit when using --batch or --clips-file")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--h5-file", type=str, default=None)
    parser.add_argument(
        "--betas-from-interhuman-root",
        type=str,
        default=None,
        help="InterHuman only: replace clip betas with matching GT InterHuman betas from this root",
    )
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument(
        "--frame-policy",
        choices=["first", "middle", "representative"],
        default="representative",
        help="How to choose which frames to render from a clip",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--show-caption", action="store_true", help="Overlay annotation text in the info panel")
    parser.add_argument("--touching-threshold-m", type=float, default=0.005)
    parser.add_argument("--barely-threshold-m", type=float, default=0.020)
    parser.add_argument("--penetration-probe-distance-m", type=float, default=0.010)
    parser.add_argument("--penetration-min-depth-m", type=float, default=0.002)
    parser.add_argument("--self-penetration-mode", choices=["off", "heuristic"], default="off")
    parser.add_argument("--self-penetration-threshold-m", type=float, default=0.004)
    parser.add_argument("--self-penetration-k", type=int, default=12)
    parser.add_argument("--self-penetration-normal-dot-max", type=float, default=-0.2)
    parser.add_argument("--max-inside-queries", type=int, default=256)
    parser.add_argument("--mesh-sample-verts", type=int, default=3000)
    parser.add_argument("--point-size", type=float, default=0.5)
    parser.add_argument("--contact-point-size", type=float, default=18.0)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--convert-interx-to-zup", action="store_true", default=True)
    parser.add_argument("--no-convert-interx-to-zup", dest="convert_interx_to_zup", action="store_false")
    return parser.parse_args()


def build_contact_config(args: argparse.Namespace) -> ContactConfig:
    return ContactConfig(
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


def resolve_jobs(args: argparse.Namespace) -> List[str]:
    if args.clip:
        return [args.clip]
    if args.clips_file:
        with open(args.clips_file, "r", encoding="utf-8") as f:
            clip_ids = [ln.strip() for ln in f.readlines() if ln.strip()]
    elif args.batch:
        clip_ids = list(DEFAULT_BATCH_CLIPS[args.batch])
    else:
        raise ValueError("Pass --clip, --clips-file, or --batch")

    if args.max_clips is not None and args.max_clips > 0:
        return clip_ids[: args.max_clips]
    return clip_ids


def sample_vertices(vertices: np.ndarray, max_verts: int) -> np.ndarray:
    if max_verts <= 0 or vertices.shape[0] <= max_verts:
        return vertices
    idx = np.linspace(0, vertices.shape[0] - 1, num=max_verts, dtype=np.int32)
    return vertices[idx]


def select_frame_positions(frame_summaries: Sequence[Dict[str, object]], num_frames: int, policy: str) -> List[int]:
    total = len(frame_summaries)
    if total == 0:
        return []
    if num_frames <= 0 or num_frames >= total:
        return list(range(total))

    if policy == "first":
        return list(range(num_frames))

    if policy == "middle":
        start = max(0, (total - num_frames) // 2)
        return list(range(start, min(total, start + num_frames)))

    ranked = sorted(
        range(total),
        key=lambda idx: (
            STATUS_ORDER.get(str(frame_summaries[idx]["status"]), 99),
            float(frame_summaries[idx]["min_distance_m"]),
            -int(frame_summaries[idx].get("contact_vertex_count_p1", 0))
            - int(frame_summaries[idx].get("contact_vertex_count_p2", 0)),
            idx,
        ),
    )
    selected = sorted(ranked[:num_frames])
    return selected


def _set_axes_equal(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.6)
    radius = max(radius, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_clip_bundle(
    *,
    dataset: str,
    clip: str,
    data_root: Optional[str],
    h5_file: Optional[str],
    betas_from_interhuman_root: Optional[str],
    body_model_path: str,
    device: str,
    batch_size: int,
    cfg: ContactConfig,
    convert_interx_to_zup: bool,
    caption_root: Optional[str] = None,
) -> Dict[str, object]:
    if data_root is None:
        data_root = (
            os.path.join(PROJECT_ROOT, "data", "InterHuman")
            if dataset == "interhuman"
            else os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset")
        )

    if dataset == "interhuman":
        persons = load_interhuman_clip(data_root, clip)
        if betas_from_interhuman_root is not None:
            persons, _ = _override_interhuman_betas_from_root(persons, clip, betas_from_interhuman_root)
    else:
        if betas_from_interhuman_root is not None:
            raise ValueError("--betas-from-interhuman-root is only supported for InterHuman clips")
        persons = load_interx_clip(
            data_root,
            clip,
            h5_file=h5_file,
            convert_to_zup=convert_interx_to_zup,
        )

    v1, v2, j1, j2, faces = _reconstruct_mesh_and_joints(
        persons,
        body_model_path=body_model_path,
        device=device,
        batch_size=batch_size,
    )
    analyzer = MeshContactAnalyzer(faces, cfg)
    frame_summaries, frame_details = analyzer.analyze(
        vertices_p1=v1,
        vertices_p2=v2,
        frame_indices=np.arange(v1.shape[0], dtype=np.int32),
        verbose_every=0,
    )
    for idx, summary in enumerate(frame_summaries):
        summary["frame"] = idx

    text_root = caption_root or data_root
    return {
        "dataset": dataset,
        "clip": clip,
        "data_root": data_root,
        "text_lines": _load_clip_text(dataset, text_root, clip),
        "vertices_p1": v1,
        "vertices_p2": v2,
        "joints_p1": j1,
        "joints_p2": j2,
        "faces": faces,
        "frame_summaries": frame_summaries,
        "frame_details": frame_details,
    }


def add_scene_axis(
    ax,
    bundle: Dict[str, object],
    local_idx: int,
    *,
    mesh_sample_verts: int,
    point_size: float,
    contact_point_size: float,
    elev: float,
    azim: float,
    title: str,
) -> None:
    v1 = np.asarray(bundle["vertices_p1"])[local_idx]
    v2 = np.asarray(bundle["vertices_p2"])[local_idx]
    j1 = np.asarray(bundle["joints_p1"])[local_idx]
    j2 = np.asarray(bundle["joints_p2"])[local_idx]
    detail = bundle["frame_details"][local_idx]
    summary = bundle["frame_summaries"][local_idx]

    p1_cloud = sample_vertices(v1, mesh_sample_verts)
    p2_cloud = sample_vertices(v2, mesh_sample_verts)
    ax.scatter(p1_cloud[:, 0], p1_cloud[:, 1], p1_cloud[:, 2], s=point_size, c="#f4a3a3", alpha=0.12)
    ax.scatter(p2_cloud[:, 0], p2_cloud[:, 1], p2_cloud[:, 2], s=point_size, c="#8bbcf4", alpha=0.12)

    def plot_idx_points(vertices: np.ndarray, indices: np.ndarray, color: str, label: str, size: float) -> None:
        if indices.size == 0:
            return
        pts = vertices[indices]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, alpha=0.95, label=label)

    plot_idx_points(v1, detail.get("inter_person_penetrating_vidx_p1", detail["penetrating_vidx_p1"]), "#a020f0", "P1 inter-pen", contact_point_size)
    plot_idx_points(v2, detail.get("inter_person_penetrating_vidx_p2", detail["penetrating_vidx_p2"]), "#a020f0", "P2 inter-pen", contact_point_size)
    plot_idx_points(v1, detail.get("self_penetrating_vidx_p1", np.zeros((0,), dtype=np.int32)), "#22aa66", "P1 self-pen", contact_point_size)
    plot_idx_points(v2, detail.get("self_penetrating_vidx_p2", np.zeros((0,), dtype=np.int32)), "#22aa66", "P2 self-pen", contact_point_size)
    plot_idx_points(v1, detail["contact_vidx_p1"], "#d62728", "P1 touch", contact_point_size * 0.8)
    plot_idx_points(v2, detail["contact_vidx_p2"], "#d62728", "P2 touch", contact_point_size * 0.8)
    plot_idx_points(v1, detail["barely_vidx_p1"], "#ffb347", "P1 barely", contact_point_size * 0.65)
    plot_idx_points(v2, detail["barely_vidx_p2"], "#ffb347", "P2 barely", contact_point_size * 0.65)

    for start, end in SMPL_BONES:
        if start < j1.shape[0] and end < j1.shape[0]:
            ax.plot([j1[start, 0], j1[end, 0]], [j1[start, 1], j1[end, 1]], [j1[start, 2], j1[end, 2]], c="#b22222", lw=1.0, alpha=0.9)
        if start < j2.shape[0] and end < j2.shape[0]:
            ax.plot([j2[start, 0], j2[end, 0]], [j2[start, 1], j2[end, 1]], [j2[start, 2], j2[end, 2]], c="#1f77b4", lw=1.0, alpha=0.9)

    closest_p1 = np.asarray(summary["closest_point_p1"], dtype=np.float32)
    closest_p2 = np.asarray(summary["closest_point_p2"], dtype=np.float32)
    ax.plot(
        [closest_p1[0], closest_p2[0]],
        [closest_p1[1], closest_p2[1]],
        [closest_p1[2], closest_p2[2]],
        c="#333333",
        lw=1.5,
        alpha=0.9,
    )

    all_points = np.concatenate([p1_cloud, p2_cloud, j1, j2, closest_p1[None], closest_p2[None]], axis=0)
    _set_axes_equal(ax, all_points)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def add_info_axis(ax, bundle: Dict[str, object], local_idx: int, show_caption: bool) -> None:
    summary = bundle["frame_summaries"][local_idx]
    frame_value = int(summary.get("frame", local_idx))
    lines = [
        f"dataset: {bundle['dataset']}",
        f"clip: {bundle['clip']}",
        f"frame: {frame_value}",
        f"status: {summary['status']}",
        f"penetration_source: {summary.get('penetration_source', 'none')}",
        f"min_distance: {100.0 * float(summary['min_distance_m']):.2f} cm",
        f"contact vertices p1/p2: {summary['contact_vertex_count_p1']} / {summary['contact_vertex_count_p2']}",
        f"barely vertices p1/p2: {summary['barely_vertex_count_p1']} / {summary['barely_vertex_count_p2']}",
        (
            "inter-penetrating vertices p1/p2: "
            f"{summary.get('inter_person_penetrating_vertex_count_p1', 0)} / "
            f"{summary.get('inter_person_penetrating_vertex_count_p2', 0)}"
        ),
    ]
    if summary.get("self_penetration_computed", False):
        lines.append(
            "self-penetrating vertices p1/p2: "
            f"{summary.get('self_penetrating_vertex_count_p1', 0)} / "
            f"{summary.get('self_penetrating_vertex_count_p2', 0)}"
        )

    if show_caption:
        lines.append("")
        lines.append("caption:")
        lines.extend(str(line) for line in bundle["text_lines"][:6])
        if len(bundle["text_lines"]) > 6:
            lines.append(f"... (+{len(bundle['text_lines']) - 6} more lines)")

    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10)


def render_bundle_frame(
    bundle: Dict[str, object],
    local_idx: int,
    out_path: str,
    *,
    show_caption: bool,
    mesh_sample_verts: int,
    point_size: float,
    contact_point_size: float,
    elev: float,
    azim: float,
    title_prefix: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0])
    ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
    ax_info = fig.add_subplot(gs[0, 1])

    summary = bundle["frame_summaries"][local_idx]
    title = f"{bundle['dataset']} {bundle['clip']} frame {int(summary.get('frame', local_idx))}"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    add_scene_axis(
        ax_scene,
        bundle,
        local_idx,
        mesh_sample_verts=mesh_sample_verts,
        point_size=point_size,
        contact_point_size=contact_point_size,
        elev=elev,
        azim=azim,
        title=title,
    )
    add_info_axis(ax_info, bundle, local_idx, show_caption)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dataset is None:
        args.dataset = args.batch or "interhuman"

    if args.data_root is None:
        args.data_root = (
            os.path.join(PROJECT_ROOT, "data", "InterHuman")
            if args.dataset == "interhuman"
            else os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset")
        )

    cfg = build_contact_config(args)
    clips = resolve_jobs(args)
    for clip in clips:
        bundle = load_clip_bundle(
            dataset=args.dataset,
            clip=clip,
            data_root=args.data_root,
            h5_file=args.h5_file,
            betas_from_interhuman_root=args.betas_from_interhuman_root,
            body_model_path=args.body_model_path,
            device=args.device,
            batch_size=args.batch_size,
            cfg=cfg,
            convert_interx_to_zup=args.convert_interx_to_zup,
        )
        local_indices = select_frame_positions(bundle["frame_summaries"], args.frames_per_clip, args.frame_policy)
        for local_idx in local_indices:
            frame_value = int(bundle["frame_summaries"][local_idx].get("frame", local_idx))
            out_path = os.path.join(args.out_dir, f"{args.dataset}_{clip}_frame_{frame_value:05d}.png")
            render_bundle_frame(
                bundle,
                local_idx,
                out_path,
                show_caption=args.show_caption,
                mesh_sample_verts=args.mesh_sample_verts,
                point_size=args.point_size,
                contact_point_size=args.contact_point_size,
                elev=args.elev,
                azim=args.azim,
            )
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
