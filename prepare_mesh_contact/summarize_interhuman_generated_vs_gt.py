#!/usr/bin/env python3
"""Summarize generated-vs-GT InterHuman mesh-contact JSON outputs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generated-vs-GT mesh-contact outputs")
    parser.add_argument("--generated-json-dir", type=str, required=True)
    parser.add_argument("--gt-json-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, default=None)
    return parser.parse_args()


def load_json_dir(path: str) -> Dict[str, Dict[str, object]]:
    outputs: Dict[str, Dict[str, object]] = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".json"):
            continue
        full = os.path.join(path, name)
        with open(full, "r", encoding="utf-8") as f:
            payload = json.load(f)
        clip = str(payload.get("clip") or os.path.splitext(name)[0])
        outputs[clip] = payload
    return outputs


def clip_stats(payload: Dict[str, object]) -> Dict[str, float]:
    frames = payload.get("frames", [])
    if not frames:
        return {
            "num_frames": 0.0,
            "penetrating_fraction": 0.0,
            "touching_fraction": 0.0,
            "barely_fraction": 0.0,
            "mean_min_distance_m": float("nan"),
            "mean_contact_vertices": 0.0,
        }

    num_frames = float(len(frames))
    statuses = [str(frame.get("status", "not_touching")) for frame in frames]
    min_d = np.array([float(frame.get("min_distance_m", np.nan)) for frame in frames], dtype=np.float32)
    mean_contact = np.mean(
        [
            float(frame.get("contact_vertex_count_p1", 0)) + float(frame.get("contact_vertex_count_p2", 0))
            for frame in frames
        ]
    )
    return {
        "num_frames": num_frames,
        "penetrating_fraction": sum(s == "penetrating" for s in statuses) / num_frames,
        "touching_fraction": sum(s == "touching" for s in statuses) / num_frames,
        "barely_fraction": sum(s == "barely_touching" for s in statuses) / num_frames,
        "mean_min_distance_m": float(np.nanmean(min_d)),
        "mean_contact_vertices": float(mean_contact),
    }


def build_markdown(summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# InterHuman Generated vs GT Mesh-Contact Summary")
    lines.append("")
    lines.append(f"Common clips: {summary['num_common_clips']}")
    lines.append("")
    lines.append("## Dataset Means")
    lines.append("")
    lines.append("| Metric | Generated | GT | Delta (Generated-GT) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for metric, values in summary["dataset_means"].items():
        lines.append(
            f"| {metric} | {values['generated']:.6f} | {values['gt']:.6f} | {values['delta']:.6f} |"
        )
    lines.append("")
    lines.append("## Largest Per-Clip Gaps")
    lines.append("")
    lines.append("| Clip | Delta penetrating frac | Delta mean min dist (m) |")
    lines.append("| --- | ---: | ---: |")
    for row in summary["largest_clip_gaps"][:10]:
        lines.append(
            f"| {row['clip']} | {row['penetrating_fraction_delta']:.6f} | {row['mean_min_distance_m_delta']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    generated = load_json_dir(args.generated_json_dir)
    gt = load_json_dir(args.gt_json_dir)
    common = sorted(set(generated) & set(gt))
    if not common:
        raise ValueError("No common clips found between generated and GT JSON directories")

    per_clip = []
    for clip in common:
        gen_stats = clip_stats(generated[clip])
        gt_stats = clip_stats(gt[clip])
        deltas = {f"{key}_delta": float(gen_stats[key] - gt_stats[key]) for key in gen_stats}
        per_clip.append({"clip": clip, "generated": gen_stats, "gt": gt_stats, **deltas})

    metric_names = [
        "penetrating_fraction",
        "touching_fraction",
        "barely_fraction",
        "mean_min_distance_m",
        "mean_contact_vertices",
    ]
    dataset_means = {}
    for metric in metric_names:
        gen_vals = np.array([row["generated"][metric] for row in per_clip], dtype=np.float64)
        gt_vals = np.array([row["gt"][metric] for row in per_clip], dtype=np.float64)
        dataset_means[metric] = {
            "generated": float(np.nanmean(gen_vals)),
            "gt": float(np.nanmean(gt_vals)),
            "delta": float(np.nanmean(gen_vals - gt_vals)),
        }

    largest_clip_gaps = sorted(
        per_clip,
        key=lambda row: (
            abs(float(row["penetrating_fraction_delta"])),
            abs(float(row["mean_min_distance_m_delta"])),
        ),
        reverse=True,
    )

    summary = {
        "num_common_clips": len(common),
        "generated_json_dir": args.generated_json_dir,
        "gt_json_dir": args.gt_json_dir,
        "dataset_means": dataset_means,
        "largest_clip_gaps": largest_clip_gaps,
        "per_clip": per_clip,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(build_markdown(summary))

    print(f"Saved {args.output_json}")
    if args.output_md:
        print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
