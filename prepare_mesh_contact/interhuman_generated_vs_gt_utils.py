#!/usr/bin/env python3
"""Shared helpers for InterHuman generated-vs-GT contact analysis."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INTERHUMAN_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "InterHuman")
DEFAULT_GT_JSON_DIR = os.path.join(PROJECT_ROOT, "output", "mesh_contact", "interhuman")
DEFAULT_GENERATED_JSON_DIR = os.path.join(PROJECT_ROOT, "output", "mesh_contact", "generated_interhuman")
DEFAULT_COMPARISON_DIR = os.path.join(PROJECT_ROOT, "output", "renders", "interhuman_gt_vs_generated_contact")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "output", "reports", "interhuman_generated_vs_gt")
DEFAULT_GENERATED_INTERHUMAN_ROOT = (
    "/mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/data/generated/interhuman"
)

INTERX_COMPLETION_COMMAND = "\n".join(
    [
        "cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken",
        "bash prepare_mesh_contact/run_interx_batch.sh --workers 4 --device cuda --batch-size 64",
    ]
)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_split_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_run_info(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_paths(json_dir: str) -> Dict[str, str]:
    if not os.path.isdir(json_dir):
        return {}
    return {
        os.path.splitext(name)[0]: os.path.join(json_dir, name)
        for name in os.listdir(json_dir)
        if name.endswith(".json")
    }


def frame_records(run_info: Dict[str, object]) -> List[Dict[str, object]]:
    frames = run_info.get("frames")
    if not isinstance(frames, list):
        return []
    out: List[Dict[str, object]] = []
    for frame in frames:
        if isinstance(frame, dict):
            out.append(frame)
    return out


def frame_count(run_info: Dict[str, object]) -> int:
    frame_range = run_info.get("frame_range")
    if isinstance(frame_range, dict):
        count = _safe_int(frame_range.get("num_frames"), default=-1)
        if count >= 0:
            return count
    return len(frame_records(run_info))


def shared_prefix_length(gt_run: Dict[str, object], gen_run: Dict[str, object]) -> int:
    return min(frame_count(gt_run), frame_count(gen_run))


def normalize_thresholds(run_info: Dict[str, object]) -> Dict[str, object]:
    raw = run_info.get("thresholds_m")
    if not isinstance(raw, dict):
        return {}
    return {
        "touching": round(_safe_float(raw.get("touching"), default=float("nan")), 8),
        "barely": round(_safe_float(raw.get("barely"), default=float("nan")), 8),
        "penetration_probe": round(_safe_float(raw.get("penetration_probe"), default=float("nan")), 8),
        "penetration_min_depth": round(_safe_float(raw.get("penetration_min_depth"), default=float("nan")), 8),
        "self_penetration_mode": str(raw.get("self_penetration_mode", "")),
        "self_penetration_threshold": round(
            _safe_float(raw.get("self_penetration_threshold"), default=float("nan")), 8
        ),
        "self_penetration_k": _safe_int(raw.get("self_penetration_k"), default=-1),
        "self_penetration_normal_dot_max": round(
            _safe_float(raw.get("self_penetration_normal_dot_max"), default=float("nan")), 8
        ),
    }


def thresholds_signature(run_info: Dict[str, object]) -> str:
    return json.dumps(normalize_thresholds(run_info), sort_keys=True)


def ensure_threshold_compatibility(
    named_run_infos: Sequence[Tuple[str, Dict[str, object]]],
    require_self_penetration_off: bool = True,
) -> Dict[str, object]:
    signatures: Dict[str, List[str]] = {}
    normalized_by_signature: Dict[str, Dict[str, object]] = {}
    for label, run_info in named_run_infos:
        normalized = normalize_thresholds(run_info)
        signature = json.dumps(normalized, sort_keys=True)
        signatures.setdefault(signature, []).append(label)
        normalized_by_signature[signature] = normalized

    if not signatures:
        return {}
    if len(signatures) != 1:
        parts = []
        for signature, labels in signatures.items():
            parts.append(f"{signature}: {labels[:5]}")
        raise ValueError(f"Mixed threshold configs detected across summaries: {' | '.join(parts)}")

    normalized = next(iter(normalized_by_signature.values()))
    if require_self_penetration_off and normalized.get("self_penetration_mode") != "off":
        raise ValueError(
            "Summary requires self_penetration_mode=off, "
            f"but found {normalized.get('self_penetration_mode')!r}"
        )
    return normalized


def select_generated_penetrating_frame(
    generated_run: Dict[str, object],
    shared_len: int,
) -> Optional[int]:
    if shared_len <= 0:
        return None

    candidates: List[Tuple[float, float, int]] = []
    for frame in frame_records(generated_run):
        frame_idx = _safe_int(frame.get("frame"), default=-1)
        if frame_idx < 0 or frame_idx >= shared_len:
            continue
        if not bool(frame.get("has_inter_person_penetration", False)):
            continue
        depth = _safe_float(frame.get("inter_person_penetration_depth_est_m"), default=0.0)
        min_distance = _safe_float(frame.get("min_distance_m"), default=math.inf)
        candidates.append((-depth, min_distance, frame_idx))

    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][2])


def load_caption_lines(data_root: str, clip_id: str, max_lines: int = 3) -> List[str]:
    annot_path = os.path.join(data_root, "annots", f"{clip_id}.txt")
    if max_lines <= 0 or not os.path.isfile(annot_path):
        return []
    with open(annot_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()][:max_lines]


def joined_caption(data_root: str, clip_id: str, max_lines: int = 3) -> str:
    return " || ".join(load_caption_lines(data_root, clip_id, max_lines=max_lines))


def comparison_png_path(output_dir: str, clip_id: str, frame_idx: int) -> str:
    return os.path.join(output_dir, f"interhuman_{clip_id}_frame_{frame_idx:05d}.png")


def summarize_frames(run_info: Dict[str, object]) -> Dict[str, float]:
    frames = frame_records(run_info)
    total = len(frames)
    inter_penetration_frames = 0
    touching_frames = 0
    barely_touching_frames = 0
    not_touching_frames = 0
    min_distance_sum_m = 0.0
    inter_penetration_depth_sum_m = 0.0
    max_inter_penetration_depth_m = 0.0

    for frame in frames:
        status = str(frame.get("status", ""))
        if status == "touching":
            touching_frames += 1
        elif status == "barely_touching":
            barely_touching_frames += 1
        elif status == "not_touching":
            not_touching_frames += 1

        min_distance_sum_m += _safe_float(frame.get("min_distance_m"), default=0.0)

        if bool(frame.get("has_inter_person_penetration", False)):
            inter_penetration_frames += 1
            depth = _safe_float(frame.get("inter_person_penetration_depth_est_m"), default=0.0)
            inter_penetration_depth_sum_m += depth
            max_inter_penetration_depth_m = max(max_inter_penetration_depth_m, depth)

    return {
        "total_frames": float(total),
        "inter_penetration_frames": float(inter_penetration_frames),
        "touching_frames": float(touching_frames),
        "barely_touching_frames": float(barely_touching_frames),
        "not_touching_frames": float(not_touching_frames),
        "min_distance_sum_m": float(min_distance_sum_m),
        "inter_penetration_depth_sum_m": float(inter_penetration_depth_sum_m),
        "max_inter_penetration_depth_m": float(max_inter_penetration_depth_m),
        "clip_has_any_inter_penetration": float(inter_penetration_frames > 0),
    }
