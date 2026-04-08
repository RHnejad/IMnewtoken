#!/usr/bin/env python3
"""Render side-by-side InterHuman GT vs generated contact comparisons."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_MPLCONFIGDIR = os.path.join("/tmp", "render_interhuman_generated_vs_gt_mpl")
os.makedirs(_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prepare_mesh_contact.interhuman_generated_vs_gt_utils import (
    DEFAULT_COMPARISON_DIR,
    DEFAULT_GENERATED_INTERHUMAN_ROOT,
    DEFAULT_GENERATED_JSON_DIR,
    DEFAULT_GT_JSON_DIR,
    DEFAULT_INTERHUMAN_DATA_ROOT,
    comparison_png_path,
    ensure_threshold_compatibility,
    frame_count,
    list_json_paths,
    load_caption_lines,
    load_run_info,
    read_split_file,
    select_generated_penetrating_frame,
    shared_prefix_length,
)
from prepare_mesh_contact.mesh_contact_pipeline import (
    ContactConfig,
    MeshContactAnalyzer,
    _override_interhuman_betas_from_root,
    load_interhuman_clip,
    reconstruct_smplx_mesh_sequence,
)
from prepare_mesh_contact.render_contact_headless import (
    _face_colors,
    _set_3d_bounds,
    _vertex_colors,
)


def _slice_people_to_frame(persons: Sequence[Dict[str, np.ndarray]], frame_idx: int) -> List[Dict[str, np.ndarray]]:
    sliced: List[Dict[str, np.ndarray]] = []
    for person in persons[:2]:
        item: Dict[str, np.ndarray] = {"name": person["name"]}
        for key, value in person.items():
            if key == "name":
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > frame_idx and key != "betas":
                item[key] = arr[frame_idx : frame_idx + 1].copy()
            elif key == "betas" and arr.ndim == 2 and arr.shape[0] > frame_idx:
                item[key] = arr[frame_idx : frame_idx + 1].copy()
            else:
                item[key] = arr.copy()
        sliced.append(item)
    return sliced


def _threshold_config_from_run(run_info: Dict[str, object]) -> ContactConfig:
    thresholds = run_info.get("thresholds_m", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    return ContactConfig(
        touching_threshold_m=float(thresholds.get("touching", 0.005)),
        barely_threshold_m=float(thresholds.get("barely", 0.020)),
        penetration_probe_distance_m=float(thresholds.get("penetration_probe", 0.010)),
        penetration_min_depth_m=float(thresholds.get("penetration_min_depth", 0.002)),
        self_penetration_mode=str(thresholds.get("self_penetration_mode", "off")),
        self_penetration_threshold_m=float(thresholds.get("self_penetration_threshold", 0.004)),
        self_penetration_k=int(thresholds.get("self_penetration_k", 12)),
        self_penetration_normal_dot_max=float(thresholds.get("self_penetration_normal_dot_max", -0.2)),
    )


def _render_mesh_panel(ax, vertices_p1: np.ndarray, vertices_p2: np.ndarray, faces: np.ndarray, detail: Dict[str, np.ndarray]) -> None:
    vc1 = _vertex_colors(
        vertices_p1.shape[0],
        detail.get("contact_vidx_p1", np.zeros(0, dtype=np.int32)),
        detail.get("barely_vidx_p1", np.zeros(0, dtype=np.int32)),
        detail.get("inter_person_penetrating_vidx_p1", np.zeros(0, dtype=np.int32)),
        detail.get("self_penetrating_vidx_p1", np.zeros(0, dtype=np.int32)),
    )
    vc2 = _vertex_colors(
        vertices_p2.shape[0],
        detail.get("contact_vidx_p2", np.zeros(0, dtype=np.int32)),
        detail.get("barely_vidx_p2", np.zeros(0, dtype=np.int32)),
        detail.get("inter_person_penetrating_vidx_p2", np.zeros(0, dtype=np.int32)),
        detail.get("self_penetrating_vidx_p2", np.zeros(0, dtype=np.int32)),
    )
    fc1 = _face_colors(vc1, faces)
    fc2 = _face_colors(vc2, faces)
    for verts, face_colors in ((vertices_p1, fc1), (vertices_p2, fc2)):
        tri_verts = verts[faces]
        mesh = Poly3DCollection(tri_verts, linewidths=0.05, edgecolors=(0.3, 0.3, 0.3, 0.1))
        mesh.set_facecolor(face_colors)
        ax.add_collection3d(mesh)


def _info_lines(
    clip_id: str,
    frame_idx: int,
    gt_length: int,
    generated_length: int,
    caption_lines: Sequence[str],
    gt_summary: Dict[str, object],
    generated_summary: Dict[str, object],
) -> List[Tuple[str, str, str]]:
    def status_color(status: str) -> str:
        return {
            "penetrating": "purple",
            "touching": "red",
            "barely_touching": "orange",
            "not_touching": "green",
        }.get(status, "black")

    def pen_depth_cm(summary: Dict[str, object]) -> float:
        return 100.0 * float(summary.get("inter_person_penetration_depth_est_m", 0.0))

    def min_distance_cm(summary: Dict[str, object]) -> float:
        return 100.0 * float(summary.get("min_distance_m", 0.0))

    lines: List[Tuple[str, str, str]] = [
        (f"clip={clip_id}", "bold", "black"),
        (f"shared frame={frame_idx}", "bold", "black"),
        (f"GT frames={gt_length} | generated frames={generated_length}", "normal", "black"),
        ("", "normal", "black"),
    ]
    if caption_lines:
        lines.append(("Caption:", "bold", "black"))
        for line in caption_lines:
            lines.append((f"  {line}", "normal", "black"))
        lines.append(("", "normal", "black"))

    gt_status = str(gt_summary.get("status", "?"))
    gen_status = str(generated_summary.get("status", "?"))
    lines.extend(
        [
            ("GT:", "bold", "black"),
            (f"  status={gt_status}", "normal", status_color(gt_status)),
            (f"  min_distance={min_distance_cm(gt_summary):.2f} cm", "normal", "black"),
            (f"  penetration_depth={pen_depth_cm(gt_summary):.2f} cm", "normal", "black"),
            ("", "normal", "black"),
            ("Generated:", "bold", "black"),
            (f"  status={gen_status}", "normal", status_color(gen_status)),
            (f"  min_distance={min_distance_cm(generated_summary):.2f} cm", "normal", "black"),
            (f"  penetration_depth={pen_depth_cm(generated_summary):.2f} cm", "normal", "black"),
            ("", "normal", "black"),
            ("Legend:", "bold", "black"),
            ("  Grey   = no contact", "normal", "black"),
            ("  Orange = barely touching", "normal", "black"),
            ("  Red    = touching", "normal", "black"),
            ("  Purple = inter-person penetration", "normal", "black"),
            ("  Green  = self-penetration", "normal", "black"),
        ]
    )
    return lines


def render_comparison_png(
    out_path: str,
    clip_id: str,
    frame_idx: int,
    gt_vertices: Tuple[np.ndarray, np.ndarray],
    generated_vertices: Tuple[np.ndarray, np.ndarray],
    faces: np.ndarray,
    gt_detail: Dict[str, np.ndarray],
    generated_detail: Dict[str, np.ndarray],
    gt_summary: Dict[str, object],
    generated_summary: Dict[str, object],
    gt_length: int,
    generated_length: int,
    caption_lines: Sequence[str],
    elev: float,
    azim: float,
    dpi: int,
) -> str:
    fig = plt.figure(figsize=(18, 8), dpi=dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.95])
    ax_gt = fig.add_subplot(gs[0, 0], projection="3d")
    ax_gen = fig.add_subplot(gs[0, 1], projection="3d")
    ax_info = fig.add_subplot(gs[0, 2])

    gt_v1, gt_v2 = gt_vertices
    gen_v1, gen_v2 = generated_vertices
    _render_mesh_panel(ax_gt, gt_v1, gt_v2, faces, gt_detail)
    _render_mesh_panel(ax_gen, gen_v1, gen_v2, faces, generated_detail)

    shared_points = np.concatenate([gt_v1, gt_v2, gen_v1, gen_v2], axis=0)
    for ax, title in ((ax_gt, "GT mesh"), (ax_gen, "Generated mesh")):
        _set_3d_bounds(ax, shared_points, span_scale=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=10)

    ax_info.axis("off")
    y = 0.97
    for text, fontweight, color in _info_lines(
        clip_id,
        frame_idx,
        gt_length,
        generated_length,
        caption_lines,
        gt_summary,
        generated_summary,
    ):
        if text == "":
            y -= 0.03
            continue
        ax_info.text(
            0.05,
            y,
            text,
            transform=ax_info.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            fontweight=fontweight,
            color=color,
        )
        y -= 0.048

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render InterHuman GT-vs-generated contact comparisons")
    parser.add_argument("--data-root", type=str, default=DEFAULT_INTERHUMAN_DATA_ROOT)
    parser.add_argument("--generated-data-root", type=str, default=DEFAULT_GENERATED_INTERHUMAN_ROOT)
    parser.add_argument("--gt-json-dir", type=str, default=DEFAULT_GT_JSON_DIR)
    parser.add_argument("--generated-json-dir", type=str, default=DEFAULT_GENERATED_JSON_DIR)
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--clips", type=str, nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_split = set(read_split_file(os.path.join(args.data_root, "split", "test.txt")))
    gt_json_paths = list_json_paths(args.gt_json_dir)
    generated_json_paths = list_json_paths(args.generated_json_dir)
    common_clips = sorted(test_split & set(gt_json_paths) & set(generated_json_paths))
    if args.clips:
        requested = set(args.clips)
        common_clips = [clip_id for clip_id in common_clips if clip_id in requested]
    if args.max_clips > 0:
        common_clips = common_clips[: args.max_clips]

    if not common_clips:
        print("No common InterHuman GT/generated test clips available for comparison renders.")
        return

    threshold_pairs: List[Tuple[str, Dict[str, object]]] = []
    selected: List[Tuple[str, int, Dict[str, object], Dict[str, object]]] = []
    skipped = 0
    for clip_id in common_clips:
        gt_run = load_run_info(gt_json_paths[clip_id])
        gen_run = load_run_info(generated_json_paths[clip_id])
        threshold_pairs.extend([(f"gt:{clip_id}", gt_run), (f"generated:{clip_id}", gen_run)])
        shared_len = shared_prefix_length(gt_run, gen_run)
        frame_idx = select_generated_penetrating_frame(gen_run, shared_len)
        if frame_idx is None:
            skipped += 1
            continue
        selected.append((clip_id, frame_idx, gt_run, gen_run))

    thresholds = ensure_threshold_compatibility(threshold_pairs)
    print(f"Verified shared thresholds: {thresholds}")
    print(f"Common GT/generated test clips: {len(common_clips)}")
    print(f"Selected penetrating-frame renders: {len(selected)}")
    print(f"Skipped (no generated inter-person penetration in shared prefix): {skipped}")

    if not selected:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for idx, (clip_id, frame_idx, gt_run, gen_run) in enumerate(selected, start=1):
        out_path = comparison_png_path(args.out_dir, clip_id, frame_idx)
        if os.path.isfile(out_path):
            continue

        cfg = _threshold_config_from_run(gt_run)
        gt_people = load_interhuman_clip(args.data_root, clip_id)
        generated_people = load_interhuman_clip(args.generated_data_root, clip_id)
        generated_people, _ = _override_interhuman_betas_from_root(
            generated_people,
            clip_id=clip_id,
            gt_interhuman_root=args.data_root,
        )

        gt_one_frame = _slice_people_to_frame(gt_people, frame_idx)
        generated_one_frame = _slice_people_to_frame(generated_people, frame_idx)

        gt_v1_seq, gt_v2_seq, faces = reconstruct_smplx_mesh_sequence(
            gt_one_frame,
            body_model_path=args.body_model_path,
            device=args.device,
            batch_size=max(1, args.batch_size),
        )
        gen_v1_seq, gen_v2_seq, faces_gen = reconstruct_smplx_mesh_sequence(
            generated_one_frame,
            body_model_path=args.body_model_path,
            device=args.device,
            batch_size=max(1, args.batch_size),
        )
        if not np.array_equal(faces, faces_gen):
            raise ValueError(f"Face topology mismatch for clip {clip_id}")

        analyzer = MeshContactAnalyzer(faces=faces, config=cfg)
        gt_summary, gt_detail = analyzer._analyze_frame(gt_v1_seq[0], gt_v2_seq[0])
        generated_summary, generated_detail = analyzer._analyze_frame(gen_v1_seq[0], gen_v2_seq[0])

        render_comparison_png(
            out_path=out_path,
            clip_id=clip_id,
            frame_idx=frame_idx,
            gt_vertices=(gt_v1_seq[0], gt_v2_seq[0]),
            generated_vertices=(gen_v1_seq[0], gen_v2_seq[0]),
            faces=faces,
            gt_detail=gt_detail,
            generated_detail=generated_detail,
            gt_summary=gt_summary,
            generated_summary=generated_summary,
            gt_length=frame_count(gt_run),
            generated_length=frame_count(gen_run),
            caption_lines=load_caption_lines(args.data_root, clip_id, max_lines=3),
            elev=args.elev,
            azim=args.azim,
            dpi=args.dpi,
        )
        if idx % 20 == 0 or idx == len(selected):
            print(f"Rendered {idx}/{len(selected)} comparison PNGs")


if __name__ == "__main__":
    main()
