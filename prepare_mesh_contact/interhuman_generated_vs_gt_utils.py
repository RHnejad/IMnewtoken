#!/usr/bin/env python3
"""Utilities for comparing generated InterHuman SMPL-X clips against GT."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GT_ROOT = os.path.join(PROJECT_ROOT, "data", "InterHuman")

REQUIRED_SMPLX_KEYS = ("trans", "root_orient", "pose_body")
GENERATED_ROOT_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "data", "generated_interhuman_smplx"),
    os.path.join(PROJECT_ROOT, "data", "generated", "interhuman", "pkl"),
    os.path.join(PROJECT_ROOT, "data", "generated_intergen", "interhuman"),
)


def _candidate_pkl_paths(data_root: str) -> List[str]:
    motions_dir = os.path.join(data_root, "motions")
    candidates: List[str] = []
    if os.path.isdir(motions_dir):
        candidates.extend(os.path.join(motions_dir, fn) for fn in os.listdir(motions_dir) if fn.endswith(".pkl"))
    if os.path.isdir(data_root):
        candidates.extend(os.path.join(data_root, fn) for fn in os.listdir(data_root) if fn.endswith(".pkl"))
    return sorted(dict.fromkeys(candidates))


def clip_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def inspect_interhuman_pkl(path: str) -> Dict[str, object]:
    obj = load_pkl(path)
    info: Dict[str, object] = {
        "path": path,
        "clip": clip_id_from_path(path),
        "top_level_keys": sorted(obj.keys()) if isinstance(obj, dict) else [],
        "person1_keys": [],
        "person2_keys": [],
        "smplx_compatible": False,
        "missing_required_keys": [],
    }
    if not isinstance(obj, dict):
        return info

    person1 = obj.get("person1") if isinstance(obj.get("person1"), dict) else None
    person2 = obj.get("person2") if isinstance(obj.get("person2"), dict) else None
    info["person1_keys"] = sorted(person1.keys()) if person1 else []
    info["person2_keys"] = sorted(person2.keys()) if person2 else []

    required = set(REQUIRED_SMPLX_KEYS)
    if person1 and person2:
        missing = sorted((required - set(person1.keys())) | (required - set(person2.keys())))
        info["missing_required_keys"] = missing
        info["smplx_compatible"] = len(missing) == 0
    return info


def list_interhuman_clip_ids(data_root: str, require_smplx: bool = False) -> List[str]:
    clip_ids: List[str] = []
    for path in _candidate_pkl_paths(data_root):
        if require_smplx:
            info = inspect_interhuman_pkl(path)
            if not info["smplx_compatible"]:
                continue
        clip_ids.append(clip_id_from_path(path))
    return sorted(dict.fromkeys(clip_ids))


def validate_smplx_interhuman_root(data_root: str, sample_size: int = 3) -> Tuple[bool, Optional[str]]:
    pkl_paths = _candidate_pkl_paths(data_root)
    if not pkl_paths:
        return False, f"No .pkl clips found in {data_root}"

    inspected = [inspect_interhuman_pkl(path) for path in pkl_paths[:sample_size]]
    bad = [info for info in inspected if not info["smplx_compatible"]]
    if bad:
        example = bad[0]
        return (
            False,
            "Generated InterHuman root is not SMPL-X compatible. "
            f"Example clip {example['clip']} is missing keys: {example['missing_required_keys']}. "
            "Expected per-person keys include trans, root_orient, pose_body (betas optional).",
        )
    return True, None


def find_common_interhuman_clips(generated_root: str, gt_root: str = DEFAULT_GT_ROOT) -> List[str]:
    gen_ok, gen_msg = validate_smplx_interhuman_root(generated_root)
    if not gen_ok:
        raise ValueError(gen_msg)

    gt_ok, gt_msg = validate_smplx_interhuman_root(gt_root)
    if not gt_ok:
        raise ValueError(gt_msg)

    generated = set(list_interhuman_clip_ids(generated_root, require_smplx=True))
    gt = set(list_interhuman_clip_ids(gt_root, require_smplx=True))
    return sorted(generated & gt)


def choose_clip_ids(clip_ids: Sequence[str], limit: Optional[int] = None) -> List[str]:
    ordered = sorted(dict.fromkeys(str(cid) for cid in clip_ids))
    if limit is None or limit <= 0:
        return ordered
    return ordered[:limit]


def resolve_generated_root(explicit_root: Optional[str]) -> str:
    if explicit_root:
        return explicit_root
    for candidate in GENERATED_ROOT_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "No generated InterHuman SMPL-X root was found automatically. "
        "Pass --generated-root explicitly."
    )


def iter_clip_paths(data_root: str, clip_ids: Optional[Iterable[str]] = None) -> List[str]:
    allowed = None if clip_ids is None else {str(cid) for cid in clip_ids}
    paths = []
    for path in _candidate_pkl_paths(data_root):
        cid = clip_id_from_path(path)
        if allowed is None or cid in allowed:
            paths.append(path)
    return sorted(paths)
