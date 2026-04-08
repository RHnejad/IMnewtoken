#!/usr/bin/env python3
"""Summarize InterHuman GT vs generated contact runs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prepare_mesh_contact.interhuman_generated_vs_gt_utils import (
    DEFAULT_COMPARISON_DIR,
    DEFAULT_GENERATED_JSON_DIR,
    DEFAULT_INTERHUMAN_DATA_ROOT,
    DEFAULT_GT_JSON_DIR,
    DEFAULT_REPORT_DIR,
    INTERX_COMPLETION_COMMAND,
    comparison_png_path,
    ensure_threshold_compatibility,
    frame_count,
    joined_caption,
    list_json_paths,
    load_run_info,
    read_split_file,
    select_generated_penetrating_frame,
    shared_prefix_length,
    summarize_frames,
)


def _mean_or_zero(total: float, count: float) -> float:
    return float(total / count) if count > 0 else 0.0


def _format_cm(meters: float) -> float:
    return 100.0 * float(meters)


def _collect_row_metrics(
    row_name: str,
    expected_clips: Sequence[str],
    json_paths: Dict[str, str],
) -> Tuple[Dict[str, object], List[str], List[Tuple[str, Dict[str, object]]]]:
    aggregate = {
        "total_frames": 0.0,
        "inter_penetration_frames": 0.0,
        "touching_frames": 0.0,
        "barely_touching_frames": 0.0,
        "not_touching_frames": 0.0,
        "min_distance_sum_m": 0.0,
        "inter_penetration_depth_sum_m": 0.0,
        "max_inter_penetration_depth_m": 0.0,
        "clip_has_any_inter_penetration": 0.0,
    }
    config_pairs: List[Tuple[str, Dict[str, object]]] = []
    covered: List[str] = []

    for clip_id in expected_clips:
        path = json_paths.get(clip_id)
        if path is None:
            continue
        run_info = load_run_info(path)
        if str(run_info.get("dataset")) != "interhuman":
            raise ValueError(f"{row_name}:{clip_id} is not an InterHuman JSON: {path}")

        config_pairs.append((f"{row_name}:{clip_id}", run_info))
        metrics = summarize_frames(run_info)
        covered.append(clip_id)
        for key, value in metrics.items():
            if key == "max_inter_penetration_depth_m":
                aggregate[key] = max(float(aggregate[key]), float(value))
            else:
                aggregate[key] += float(value)

    total_frames = float(aggregate["total_frames"])
    inter_penetration_frames = float(aggregate["inter_penetration_frames"])
    row = {
        "row_name": row_name,
        "expected_clip_count": int(len(expected_clips)),
        "json_covered_clip_count": int(len(covered)),
        "total_frame_count": int(total_frames),
        "inter_person_penetration_frame_count": int(inter_penetration_frames),
        "inter_person_penetration_frame_fraction": _mean_or_zero(inter_penetration_frames, total_frames),
        "clip_count_with_any_inter_person_penetration": int(aggregate["clip_has_any_inter_penetration"]),
        "touching_frame_count": int(aggregate["touching_frames"]),
        "barely_touching_frame_count": int(aggregate["barely_touching_frames"]),
        "not_touching_frame_count": int(aggregate["not_touching_frames"]),
        "mean_min_distance_cm": _format_cm(_mean_or_zero(aggregate["min_distance_sum_m"], total_frames)),
        "max_inter_person_penetration_depth_cm": _format_cm(aggregate["max_inter_penetration_depth_m"]),
        "mean_inter_person_penetration_depth_over_penetrating_frames_cm": _format_cm(
            _mean_or_zero(aggregate["inter_penetration_depth_sum_m"], inter_penetration_frames)
        ),
    }
    return row, covered, config_pairs


def _common_clip_rows(
    clip_ids: Sequence[str],
    gt_json_paths: Dict[str, str],
    generated_json_paths: Dict[str, str],
    data_root: str,
    comparison_dir: str,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, Dict[str, object]]]]:
    rows: List[Dict[str, object]] = []
    config_pairs: List[Tuple[str, Dict[str, object]]] = []

    for clip_id in clip_ids:
        gt_run = load_run_info(gt_json_paths[clip_id])
        gen_run = load_run_info(generated_json_paths[clip_id])
        config_pairs.extend(
            [
                (f"gt_common:{clip_id}", gt_run),
                (f"generated_common:{clip_id}", gen_run),
            ]
        )

        gt_metrics = summarize_frames(gt_run)
        gen_metrics = summarize_frames(gen_run)
        shared_len = shared_prefix_length(gt_run, gen_run)
        selected_frame = select_generated_penetrating_frame(gen_run, shared_len)
        png_path = ""
        if selected_frame is not None:
            png_path = comparison_png_path(comparison_dir, clip_id, selected_frame)

        rows.append(
            {
                "clip_id": clip_id,
                "caption": joined_caption(data_root, clip_id, max_lines=3),
                "gt_total_frames": frame_count(gt_run),
                "generated_total_frames": frame_count(gen_run),
                "shared_prefix_length": shared_len,
                "gt_inter_person_penetration_frame_count": int(gt_metrics["inter_penetration_frames"]),
                "generated_inter_person_penetration_frame_count": int(gen_metrics["inter_penetration_frames"]),
                "gt_max_inter_person_penetration_depth_cm": _format_cm(gt_metrics["max_inter_penetration_depth_m"]),
                "generated_max_inter_person_penetration_depth_cm": _format_cm(
                    gen_metrics["max_inter_penetration_depth_m"]
                ),
                "selected_comparison_frame": "" if selected_frame is None else int(selected_frame),
                "comparison_png_path": png_path,
            }
        )

    return rows, config_pairs


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    out = [header, sep]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize InterHuman GT vs generated mesh-contact JSONs")
    parser.add_argument("--data-root", type=str, default=DEFAULT_INTERHUMAN_DATA_ROOT)
    parser.add_argument("--gt-json-dir", type=str, default=DEFAULT_GT_JSON_DIR)
    parser.add_argument("--generated-json-dir", type=str, default=DEFAULT_GENERATED_JSON_DIR)
    parser.add_argument("--comparison-dir", type=str, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_split = read_split_file(os.path.join(args.data_root, "split", "train.txt"))
    test_split = read_split_file(os.path.join(args.data_root, "split", "test.txt"))
    val_split = read_split_file(os.path.join(args.data_root, "split", "val.txt"))

    gt_json_paths = list_json_paths(args.gt_json_dir)
    generated_json_paths = list_json_paths(args.generated_json_dir)

    train_set = sorted(train_split)
    test_set = sorted(test_split)
    split_union = set(train_split) | set(test_split) | set(val_split)

    gt_train_row, gt_train_covered, gt_train_configs = _collect_row_metrics(
        "gt_train", train_set, gt_json_paths
    )
    gt_test_row, gt_test_covered, gt_test_configs = _collect_row_metrics(
        "gt_test", test_set, gt_json_paths
    )
    generated_test_row, generated_test_covered, generated_test_configs = _collect_row_metrics(
        "generated_test_available", test_set, generated_json_paths
    )

    common_test_clips = sorted(set(gt_test_covered) & set(generated_test_covered))
    common_row, _, common_configs = _collect_row_metrics(
        "gt_test_common_with_generated", common_test_clips, gt_json_paths
    )
    common_clip_rows, common_clip_configs = _common_clip_rows(
        common_test_clips,
        gt_json_paths,
        generated_json_paths,
        args.data_root,
        args.comparison_dir,
    )

    thresholds = ensure_threshold_compatibility(
        gt_train_configs
        + gt_test_configs
        + generated_test_configs
        + common_configs
        + common_clip_configs
    )

    missing_gt_train = sorted(set(train_set) - set(gt_train_covered))
    missing_gt_test = sorted(set(test_set) - set(gt_test_covered))
    missing_generated_test = sorted(set(test_set) - set(generated_test_covered))
    extra_gt_outside_splits = sorted(set(gt_json_paths) - split_union)

    summary_rows = [
        gt_train_row,
        gt_test_row,
        generated_test_row,
        common_row,
    ]

    summary_by_split_csv = os.path.join(args.out_dir, "summary_by_split.csv")
    coverage_gaps_csv = os.path.join(args.out_dir, "coverage_gaps.csv")
    common_clip_metrics_csv = os.path.join(args.out_dir, "common_clip_metrics.csv")
    summary_md = os.path.join(args.out_dir, "summary.md")

    _write_csv(
        summary_by_split_csv,
        summary_rows,
        fieldnames=[
            "row_name",
            "expected_clip_count",
            "json_covered_clip_count",
            "total_frame_count",
            "inter_person_penetration_frame_count",
            "inter_person_penetration_frame_fraction",
            "clip_count_with_any_inter_person_penetration",
            "touching_frame_count",
            "barely_touching_frame_count",
            "not_touching_frame_count",
            "mean_min_distance_cm",
            "max_inter_person_penetration_depth_cm",
            "mean_inter_person_penetration_depth_over_penetrating_frames_cm",
        ],
    )

    coverage_rows: List[Dict[str, object]] = []
    for clip_id in missing_gt_train:
        coverage_rows.append({"category": "missing_gt_split_json", "split": "train", "clip_id": clip_id})
    for clip_id in missing_gt_test:
        coverage_rows.append({"category": "missing_gt_split_json", "split": "test", "clip_id": clip_id})
    for clip_id in missing_generated_test:
        coverage_rows.append({"category": "missing_generated_test_json", "split": "test", "clip_id": clip_id})

    _write_csv(
        coverage_gaps_csv,
        coverage_rows,
        fieldnames=["category", "split", "clip_id"],
    )
    _write_csv(
        common_clip_metrics_csv,
        common_clip_rows,
        fieldnames=[
            "clip_id",
            "caption",
            "gt_total_frames",
            "generated_total_frames",
            "shared_prefix_length",
            "gt_inter_person_penetration_frame_count",
            "generated_inter_person_penetration_frame_count",
            "gt_max_inter_person_penetration_depth_cm",
            "generated_max_inter_person_penetration_depth_cm",
            "selected_comparison_frame",
            "comparison_png_path",
        ],
    )

    selected_png_rows = [row for row in common_clip_rows if row["selected_comparison_frame"] != ""]
    summary_table_columns = [
        ("row_name", "Row"),
        ("expected_clip_count", "Expected Clips"),
        ("json_covered_clip_count", "JSON Covered"),
        ("total_frame_count", "Total Frames"),
        ("inter_person_penetration_frame_count", "Inter-Pen Frames"),
        ("inter_person_penetration_frame_fraction", "Inter-Pen Fraction"),
        ("clip_count_with_any_inter_person_penetration", "Clips With Any Inter-Pen"),
        ("touching_frame_count", "Touching"),
        ("barely_touching_frame_count", "Barely"),
        ("not_touching_frame_count", "Not Touching"),
        ("mean_min_distance_cm", "Mean Min Dist (cm)"),
        ("max_inter_person_penetration_depth_cm", "Max Inter-Pen Depth (cm)"),
        (
            "mean_inter_person_penetration_depth_over_penetrating_frames_cm",
            "Mean Inter-Pen Depth Over Pen Frames (cm)",
        ),
    ]

    md_lines: List[str] = []
    md_lines.append("# InterHuman Generated-vs-GT Contact Summary")
    md_lines.append("")
    md_lines.append("## Scope")
    md_lines.append("")
    md_lines.append(f"- GT JSON root: `{args.gt_json_dir}`")
    md_lines.append(f"- Generated JSON root: `{args.generated_json_dir}`")
    md_lines.append(f"- Comparison PNG root: `{args.comparison_dir}`")
    md_lines.append(f"- InterHuman data root: `{args.data_root}`")
    md_lines.append("")
    md_lines.append("## Split Summary")
    md_lines.append("")
    md_lines.extend(_markdown_table(summary_rows, summary_table_columns))
    md_lines.append("")
    md_lines.append("## Coverage Notes")
    md_lines.append("")
    md_lines.append(f"- Missing GT train JSONs: {len(missing_gt_train)}")
    md_lines.append(f"- Missing GT test JSONs: {len(missing_gt_test)}")
    md_lines.append(f"- Missing generated test JSONs: {len(missing_generated_test)}")
    md_lines.append(f"- GT JSONs outside official train/val/test splits: {len(extra_gt_outside_splits)}")
    if extra_gt_outside_splits:
        preview = ", ".join(extra_gt_outside_splits[:10])
        md_lines.append(f"- Example GT extras outside split files: `{preview}`")
    md_lines.append(f"- Common GT/generated test clips: {len(common_test_clips)}")
    md_lines.append(f"- Common clips with a selected generated-penetration frame: {len(selected_png_rows)}")
    md_lines.append("")
    md_lines.append("## Config Guardrails")
    md_lines.append("")
    md_lines.append(f"- Thresholds: `{thresholds}`")
    md_lines.append("- Summaries fail if mixed threshold configs are detected.")
    md_lines.append("- Summaries fail if `self_penetration_mode` is not `off`.")
    md_lines.append("")
    md_lines.append("## Output Files")
    md_lines.append("")
    md_lines.append(f"- `summary_by_split.csv`: `{summary_by_split_csv}`")
    md_lines.append(f"- `coverage_gaps.csv`: `{coverage_gaps_csv}`")
    md_lines.append(f"- `common_clip_metrics.csv`: `{common_clip_metrics_csv}`")
    md_lines.append("")
    md_lines.append("## InterX Completion Command")
    md_lines.append("")
    md_lines.append("```bash")
    md_lines.append(INTERX_COMPLETION_COMMAND)
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Review Checklist")
    md_lines.append("")
    md_lines.append("- Confirm `gt_train` expects 6022 clips and `gt_test` expects 1177 clips.")
    md_lines.append("- Confirm generated test coverage reaches 1098 clips once the generated batch is complete.")
    md_lines.append("- Confirm the missing generated test clip count reaches 79 for the verified InterMask export.")
    md_lines.append("- Confirm compare PNG counts match the number of common clips with non-empty `selected_comparison_frame`.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Wrote {summary_by_split_csv}")
    print(f"Wrote {coverage_gaps_csv}")
    print(f"Wrote {common_clip_metrics_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
