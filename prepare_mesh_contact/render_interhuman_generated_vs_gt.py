#!/usr/bin/env python3
"""Render side-by-side generated-vs-GT InterHuman contact diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prepare_mesh_contact.interhuman_generated_vs_gt_utils import (  # noqa: E402
    DEFAULT_GT_ROOT,
    choose_clip_ids,
    find_common_interhuman_clips,
    resolve_generated_root,
)
from prepare_mesh_contact.render_contact_headless import (  # noqa: E402
    add_info_axis,
    add_scene_axis,
    build_contact_config,
    load_clip_bundle,
    select_frame_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render generated-vs-GT InterHuman comparisons")
    parser.add_argument("--generated-root", type=str, default=None, help="Generated SMPL-X-compatible InterHuman root")
    parser.add_argument("--gt-root", type=str, default=DEFAULT_GT_ROOT, help="GT InterHuman root")
    parser.add_argument("--clip", type=str, default=None, help="Single clip id")
    parser.add_argument("--clips-file", type=str, default=None, help="Optional newline-delimited clip list")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional clip limit")
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--frame-policy", choices=["first", "middle", "representative"], default="representative")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--show-caption", action="store_true")
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
    return parser.parse_args()


def resolve_clips(args: argparse.Namespace, generated_root: str, gt_root: str) -> List[str]:
    if args.clip:
        return [args.clip]
    if args.clips_file:
        with open(args.clips_file, "r", encoding="utf-8") as f:
            clip_ids = [ln.strip() for ln in f.readlines() if ln.strip()]
    else:
        clip_ids = find_common_interhuman_clips(generated_root, gt_root)
    return choose_clip_ids(clip_ids, limit=args.max_clips)


def render_pair(generated_bundle, gt_bundle, local_idx: int, out_path: str, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(19, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9])
    ax_gen = fig.add_subplot(gs[0, 0], projection="3d")
    ax_gt = fig.add_subplot(gs[0, 1], projection="3d")
    ax_info = fig.add_subplot(gs[0, 2])

    add_scene_axis(
        ax_gen,
        generated_bundle,
        local_idx,
        mesh_sample_verts=args.mesh_sample_verts,
        point_size=args.point_size,
        contact_point_size=args.contact_point_size,
        elev=args.elev,
        azim=args.azim,
        title="Generated",
    )
    add_scene_axis(
        ax_gt,
        gt_bundle,
        local_idx,
        mesh_sample_verts=args.mesh_sample_verts,
        point_size=args.point_size,
        contact_point_size=args.contact_point_size,
        elev=args.elev,
        azim=args.azim,
        title="Ground Truth",
    )

    gen_summary = generated_bundle["frame_summaries"][local_idx]
    gt_summary = gt_bundle["frame_summaries"][local_idx]
    frame_value = int(gt_summary.get("frame", local_idx))
    lines = [
        f"clip: {gt_bundle['clip']}",
        f"frame: {frame_value}",
        "",
        f"generated status: {gen_summary['status']}",
        f"generated min dist: {100.0 * float(gen_summary['min_distance_m']):.2f} cm",
        (
            "generated contact p1/p2: "
            f"{gen_summary['contact_vertex_count_p1']} / {gen_summary['contact_vertex_count_p2']}"
        ),
        "",
        f"gt status: {gt_summary['status']}",
        f"gt min dist: {100.0 * float(gt_summary['min_distance_m']):.2f} cm",
        f"gt contact p1/p2: {gt_summary['contact_vertex_count_p1']} / {gt_summary['contact_vertex_count_p2']}",
    ]
    if args.show_caption:
        lines.append("")
        lines.append("caption:")
        lines.extend(str(line) for line in gt_bundle["text_lines"][:6])
        if len(gt_bundle["text_lines"]) > 6:
            lines.append(f"... (+{len(gt_bundle['text_lines']) - 6} more lines)")
    ax_info.axis("off")
    ax_info.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10)

    fig.suptitle(f"InterHuman generated vs GT | {gt_bundle['clip']} | frame {frame_value}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    generated_root = resolve_generated_root(args.generated_root)
    gt_root = args.gt_root
    cfg = build_contact_config(args)
    clips = resolve_clips(args, generated_root, gt_root)

    for clip in clips:
        gt_bundle = load_clip_bundle(
            dataset="interhuman",
            clip=clip,
            data_root=gt_root,
            h5_file=None,
            betas_from_interhuman_root=None,
            body_model_path=args.body_model_path,
            device=args.device,
            batch_size=args.batch_size,
            cfg=cfg,
            convert_interx_to_zup=True,
            caption_root=gt_root,
        )
        generated_bundle = load_clip_bundle(
            dataset="interhuman",
            clip=clip,
            data_root=generated_root,
            h5_file=None,
            betas_from_interhuman_root=gt_root,
            body_model_path=args.body_model_path,
            device=args.device,
            batch_size=args.batch_size,
            cfg=cfg,
            convert_interx_to_zup=True,
            caption_root=gt_root,
        )

        common_len = min(len(gt_bundle["frame_summaries"]), len(generated_bundle["frame_summaries"]))
        if common_len == 0:
            raise ValueError(f"Clip {clip} has zero common frames between generated and GT")
        gt_bundle["frame_summaries"] = gt_bundle["frame_summaries"][:common_len]
        gt_bundle["frame_details"] = gt_bundle["frame_details"][:common_len]
        gt_bundle["vertices_p1"] = gt_bundle["vertices_p1"][:common_len]
        gt_bundle["vertices_p2"] = gt_bundle["vertices_p2"][:common_len]
        gt_bundle["joints_p1"] = gt_bundle["joints_p1"][:common_len]
        gt_bundle["joints_p2"] = gt_bundle["joints_p2"][:common_len]
        generated_bundle["frame_summaries"] = generated_bundle["frame_summaries"][:common_len]
        generated_bundle["frame_details"] = generated_bundle["frame_details"][:common_len]
        generated_bundle["vertices_p1"] = generated_bundle["vertices_p1"][:common_len]
        generated_bundle["vertices_p2"] = generated_bundle["vertices_p2"][:common_len]
        generated_bundle["joints_p1"] = generated_bundle["joints_p1"][:common_len]
        generated_bundle["joints_p2"] = generated_bundle["joints_p2"][:common_len]

        local_indices = select_frame_positions(gt_bundle["frame_summaries"], args.frames_per_clip, args.frame_policy)
        for local_idx in local_indices:
            frame_value = int(gt_bundle["frame_summaries"][local_idx].get("frame", local_idx))
            out_path = os.path.join(args.out_dir, f"interhuman_{clip}_generated_vs_gt_frame_{frame_value:05d}.png")
            render_pair(generated_bundle, gt_bundle, local_idx, out_path, args)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
