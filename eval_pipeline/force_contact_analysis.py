#!/usr/bin/env python3
"""Force-Contact Analysis: overlay ImDy-predicted forces on mesh contact annotations.

Hypothesis: ImDy (trained on single-person data) should show detectable force
anomalies during two-person contact phases, since it cannot model interaction forces.

Subcommands:
    infer      - Re-run ImDy, save per-frame force arrays as .npz
    visualize  - Per-clip 6-panel force + contact overlay plots
    analyze    - Statistical comparison: contact vs non-contact frames
    aggregate  - Per-action-category force gap analysis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Imports from eval_pipeline
# ---------------------------------------------------------------------------
try:
    from eval_pipeline.imdy_preprocessor import preprocess_for_imdy
    from eval_pipeline.imdy_model_wrapper import ImDyWrapper
except ModuleNotFoundError:
    from imdy_preprocessor import preprocess_for_imdy
    from imdy_model_wrapper import ImDyWrapper

IMDY_TORQUE_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
]
IMDY_TORQUE_JOINT_NAME_SOURCE = "prepare5/ImDy/models/utils.py:JOINT_NAMES[:23]"
EXPECTED_TORQUE_SHAPE = (23, 3)
EXPECTED_GRF_SHAPE = (2, 24, 3)
EXPECTED_CONTACT_LOGIT_SHAPE = (2, 24, 1)
DEFAULT_INTERX_RETARGET_DIR = os.path.join("data", "retargeted_v2", "interx")
DEFAULT_INTERX_CONTACT_DIR = os.path.join("output", "mesh_contact", "interx")
DEFAULT_IMDY_CONFIG = os.path.join("prepare5", "ImDy", "config", "IDFD_mkr.yml")
DEFAULT_IMDY_CHECKPOINT = os.path.join(
    "prepare5", "ImDy", "downloaded_checkpoint", "imdy_pretrain.pt"
)
DEFAULT_FPS = 30.0
DEFAULT_WEIGHT_KG = 75.0
GRAVITY = 9.81
CONTACT_BINARY_STATUSES = ("touching", "penetrating")
CONTACT_POLICY = {
    "binary_contact_statuses": list(CONTACT_BINARY_STATUSES),
    "barely_touching_policy": "kept separate for visualization only; excluded from binary contact",
}


# ===========================================================================
# Utilities
# ===========================================================================

def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dir(path: str, label: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found: {path}")


def _ensure_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path}")


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _configure_headless_matplotlib():
    mpl_config_dir = os.path.join("/tmp", "matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib


def _shape_to_list(arr: np.ndarray) -> List[int]:
    return [int(x) for x in arr.shape]


def _validate_joint_name_mapping() -> None:
    if len(IMDY_TORQUE_JOINT_NAMES) != EXPECTED_TORQUE_SHAPE[0]:
        raise ValueError(
            "ImDy joint-name mapping does not match expected torque dimensionality: "
            f"{len(IMDY_TORQUE_JOINT_NAMES)} names vs {EXPECTED_TORQUE_SHAPE[0]} joints"
        )


def _validate_positions_array(positions: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (22, 3):
        raise ValueError(f"{label} must have shape (T, 22, 3), got {arr.shape}")
    return arr


def _validate_position_pair(
    clip_id: str, pos0: np.ndarray, pos1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    arr0 = _validate_positions_array(pos0, f"{clip_id} person0 positions")
    arr1 = _validate_positions_array(pos1, f"{clip_id} person1 positions")
    if arr0.shape[0] != arr1.shape[0]:
        raise ValueError(
            f"{clip_id} person sequences must have the same frame count, got "
            f"{arr0.shape[0]} and {arr1.shape[0]}"
        )
    return arr0, arr1


def _validate_clip_indices(
    clip_id: str, idx0: np.ndarray, idx1: np.ndarray
) -> np.ndarray:
    frame_idx0 = np.asarray(idx0)
    frame_idx1 = np.asarray(idx1)
    if frame_idx0.ndim != 1 or frame_idx1.ndim != 1:
        raise ValueError(
            f"{clip_id} frame indices must be 1D, got {frame_idx0.shape} and {frame_idx1.shape}"
        )
    if not np.array_equal(frame_idx0, frame_idx1):
        raise ValueError(f"{clip_id} person0/person1 ImDy frame indices differ")
    if frame_idx0.size == 0:
        raise ValueError(f"{clip_id} has no valid ImDy windows")
    if np.any(frame_idx0 < 0):
        raise ValueError(f"{clip_id} has negative frame indices")
    if np.any(np.diff(frame_idx0) <= 0):
        raise ValueError(f"{clip_id} frame indices must be strictly increasing")
    return frame_idx0.astype(np.int64, copy=False)


def _validate_array_shape(
    clip_id: str,
    label: str,
    arr: np.ndarray,
    expected_shape: Tuple[int, ...],
) -> np.ndarray:
    out = np.asarray(arr)
    if out.shape != expected_shape:
        raise ValueError(
            f"{clip_id} {label} must have shape {expected_shape}, got {out.shape}"
        )
    return out


def _validate_prediction_bundle(
    clip_id: str,
    label: str,
    pred: Dict[str, np.ndarray],
    num_windows: int,
    require_contact_logits: bool,
) -> Dict[str, np.ndarray]:
    torque = _validate_array_shape(
        clip_id,
        f"{label}.torque",
        pred.get("torque"),
        (num_windows,) + EXPECTED_TORQUE_SHAPE,
    )
    grf = _validate_array_shape(
        clip_id,
        f"{label}.grf",
        pred.get("grf"),
        (num_windows,) + EXPECTED_GRF_SHAPE,
    )

    contact_logits = pred.get("contact")
    if require_contact_logits and contact_logits is None:
        raise ValueError(f"{clip_id} {label}.contact logits are missing")
    if contact_logits is not None:
        contact_logits = _validate_array_shape(
            clip_id,
            f"{label}.contact",
            contact_logits,
            (num_windows,) + EXPECTED_CONTACT_LOGIT_SHAPE,
        )

    return {
        "torque": torque.astype(np.float32, copy=False),
        "grf": grf.astype(np.float32, copy=False),
        "contact_logits": None
        if contact_logits is None
        else contact_logits.astype(np.float32, copy=False),
    }


def _load_force_arrays(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        bundle = {
            "torque_p0": np.asarray(data["torque_p0"], dtype=np.float32),
            "torque_p1": np.asarray(data["torque_p1"], dtype=np.float32),
            "grf_p0": np.asarray(data["grf_p0"], dtype=np.float32),
            "grf_p1": np.asarray(data["grf_p1"], dtype=np.float32),
            "frame_indices": np.asarray(data["frame_indices"], dtype=np.int64),
        }
        if "contact_logits_p0" in data:
            bundle["contact_logits_p0"] = np.asarray(data["contact_logits_p0"], dtype=np.float32)
        if "contact_logits_p1" in data:
            bundle["contact_logits_p1"] = np.asarray(data["contact_logits_p1"], dtype=np.float32)

    clip_id = os.path.splitext(os.path.basename(npz_path))[0]
    frame_indices = bundle["frame_indices"]
    if frame_indices.ndim != 1:
        raise ValueError(f"{clip_id} frame_indices must be 1D, got {frame_indices.shape}")
    if frame_indices.size == 0:
        raise ValueError(f"{clip_id} frame_indices is empty")
    if np.any(frame_indices < 0):
        raise ValueError(f"{clip_id} frame_indices contains negative values")
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError(f"{clip_id} frame_indices must be strictly increasing")

    n = int(frame_indices.shape[0])
    _validate_array_shape(clip_id, "torque_p0", bundle["torque_p0"], (n,) + EXPECTED_TORQUE_SHAPE)
    _validate_array_shape(clip_id, "torque_p1", bundle["torque_p1"], (n,) + EXPECTED_TORQUE_SHAPE)
    _validate_array_shape(clip_id, "grf_p0", bundle["grf_p0"], (n,) + EXPECTED_GRF_SHAPE)
    _validate_array_shape(clip_id, "grf_p1", bundle["grf_p1"], (n,) + EXPECTED_GRF_SHAPE)

    for key in ("contact_logits_p0", "contact_logits_p1"):
        if key in bundle and bundle[key].size > 0:
            _validate_array_shape(
                clip_id,
                key,
                bundle[key],
                (n,) + EXPECTED_CONTACT_LOGIT_SHAPE,
            )

    return bundle


def _load_metadata_sidecar(npz_path: str) -> Optional[dict]:
    meta_path = os.path.splitext(npz_path)[0] + ".meta.json"
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _joint_labels_for_plot(metadata: Optional[dict]) -> List[str]:
    names = IMDY_TORQUE_JOINT_NAMES
    if metadata is not None:
        meta_names = metadata.get("torque_joint_names")
        if isinstance(meta_names, list) and len(meta_names) == len(IMDY_TORQUE_JOINT_NAMES):
            names = [str(x) for x in meta_names]
    return [name.replace("_", " ").title() for name in names]


def _align_contact_signals(
    clip_id: str, frame_indices: np.ndarray, contact_frames: List[dict]
) -> Tuple[np.ndarray, dict, int]:
    csignals = extract_mesh_contact_signals(contact_frames)
    n_contact_frames = len(contact_frames)
    valid_mask = frame_indices < n_contact_frames
    n_dropped = int((~valid_mask).sum())
    if not np.any(valid_mask):
        raise ValueError(
            f"{clip_id} has no valid aligned frames: max frame index {int(frame_indices.max())} "
            f"but contact JSON has only {n_contact_frames} frames"
        )

    aligned_idx = frame_indices[valid_mask].astype(np.int64, copy=False)
    aligned = {
        "frame_indices": aligned_idx,
        "contact_binary": csignals["contact_binary"][aligned_idx],
        "contact_ternary": csignals["contact_ternary"][aligned_idx],
        "min_distance_m": csignals["min_distance_m"][aligned_idx],
        "contact_vertex_count_p1": csignals["contact_vertex_count_p1"][aligned_idx],
        "contact_vertex_count_p2": csignals["contact_vertex_count_p2"][aligned_idx],
        "penetration_depth_m": csignals["penetration_depth_m"][aligned_idx],
        "valid_mask": valid_mask,
    }
    return valid_mask, aligned, n_dropped

def parse_action_class(clip_id: str) -> Optional[int]:
    """Extract action class integer from InterX clip ID like G001T000A003R000."""
    m = re.search(r"A(\d+)", clip_id)
    return int(m.group(1)) if m else None


def load_mesh_contact(contact_dir: str, clip_id: str) -> Optional[List[dict]]:
    """Load per-frame mesh contact JSON."""
    path = os.path.join(contact_dir, f"{clip_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", [])


def contact_status_to_binary(frames: List[dict]) -> np.ndarray:
    """Convert per-frame mesh contact status to binary array.

    Returns:
        contact_binary: (T,) bool — True if touching or penetrating.
    """
    return np.array(
        [f.get("status", "not_touching") in CONTACT_BINARY_STATUSES for f in frames],
        dtype=bool,
    )


def contact_status_to_ternary(frames: List[dict]) -> np.ndarray:
    """Convert per-frame status to 0=no_contact, 1=barely, 2=contact/penetrating."""
    result = np.zeros(len(frames), dtype=np.int8)
    for i, f in enumerate(frames):
        s = f.get("status", "not_touching")
        if s in ("touching", "penetrating"):
            result[i] = 2
        elif s == "barely_touching":
            result[i] = 1
    return result


def extract_mesh_contact_signals(frames: List[dict]) -> dict:
    """Extract key signals from mesh contact JSON frames."""
    min_dist = np.array([f.get("min_distance_m", np.nan) for f in frames], dtype=np.float32)
    cv_p1 = np.array([f.get("contact_vertex_count_p1", 0) for f in frames], dtype=np.float32)
    cv_p2 = np.array([f.get("contact_vertex_count_p2", 0) for f in frames], dtype=np.float32)
    pen_depth = np.array([f.get("penetration_depth_est_m", 0.0) for f in frames], dtype=np.float32)
    return {
        "min_distance_m": min_dist,
        "contact_vertex_count_p1": cv_p1,
        "contact_vertex_count_p2": cv_p2,
        "penetration_depth_m": pen_depth,
        "contact_binary": contact_status_to_binary(frames),
        "contact_ternary": contact_status_to_ternary(frames),
    }


def list_retargeted_clips(data_dir: str) -> List[str]:
    """List clip IDs from retargeted npy directory."""
    import glob as _glob
    pat0 = re.compile(r"^(.+)_person0\.npy$")
    pat1 = re.compile(r"^(.+)_person1\.npy$")
    clips0 = set()
    clips1 = set()
    for fp in _glob.glob(os.path.join(data_dir, "*_person0.npy")):
        name = os.path.basename(fp)
        if "_joint_q" in name or "_betas" in name or "_torques" in name:
            continue
        m = pat0.match(name)
        if m:
            clips0.add(m.group(1))
    for fp in _glob.glob(os.path.join(data_dir, "*_person1.npy")):
        name = os.path.basename(fp)
        if "_joint_q" in name or "_betas" in name or "_torques" in name:
            continue
        m = pat1.match(name)
        if m:
            clips1.add(m.group(1))
    return sorted(clips0 & clips1)


def load_retargeted_positions(data_dir: str, clip_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load person0 and person1 positions from retargeted npy files."""
    p0 = np.load(os.path.join(data_dir, f"{clip_id}_person0.npy")).astype(np.float32)
    p1 = np.load(os.path.join(data_dir, f"{clip_id}_person1.npy")).astype(np.float32)
    return p0, p1


# ===========================================================================
# Subcommand: infer
# ===========================================================================

def cmd_infer(args: argparse.Namespace) -> None:
    """Re-run ImDy inference on selected clips, save per-frame arrays."""
    _validate_joint_name_mapping()
    _ensure_dir(args.data_dir, "InterX retargeted data")
    _ensure_dir(args.contact_dir, "InterX mesh contact")
    _ensure_file(args.imdy_config, "ImDy config")
    _ensure_file(args.imdy_checkpoint, "ImDy checkpoint")
    os.makedirs(args.output_dir, exist_ok=True)

    all_clips = list_retargeted_clips(args.data_dir)
    if not all_clips:
        raise ValueError(f"No InterX retargeted clips found in {args.data_dir}")

    # Filter to clips that also have mesh contact data
    has_contact = set()
    for cid in all_clips:
        if os.path.isfile(os.path.join(args.contact_dir, f"{cid}.json")):
            has_contact.add(cid)
    all_clips = [c for c in all_clips if c in has_contact]
    print(f"[infer] {len(all_clips)} clips with both retargeted data and mesh contact")
    if not all_clips:
        raise ValueError("No overlapping InterX clips found between retargeted data and mesh contact")

    # Stratified sampling by action category
    if args.sample_per_action > 0:
        by_action: Dict[int, List[str]] = defaultdict(list)
        for cid in all_clips:
            ac = parse_action_class(cid)
            if ac is not None:
                by_action[ac].append(cid)
        selected = []
        for ac in sorted(by_action.keys()):
            candidates = sorted(by_action[ac])
            rng = np.random.default_rng(args.seed + ac)
            n = min(args.sample_per_action, len(candidates))
            selected.extend(sorted(rng.choice(candidates, n, replace=False).tolist()))
        all_clips = selected
        print(f"[infer] Stratified sample: {len(all_clips)} clips from {len(by_action)} actions")

    if args.max_clips:
        all_clips = all_clips[:args.max_clips]
    if not all_clips:
        raise ValueError("Clip selection is empty after applying sampling/max-clips filters")

    # Load model
    model = ImDyWrapper(
        config_path=args.imdy_config,
        checkpoint_path=args.imdy_checkpoint,
        device=args.device,
        use_contact_mask=True,
    )
    print(f"[infer] Model loaded on {model.device}, scoring {len(all_clips)} clips")

    past_kf = model.past_kf if args.past_kf is None else int(args.past_kf)
    fut_kf = model.fut_kf if args.fut_kf is None else int(args.fut_kf)
    preprocess_meta = {
        "past_kf": int(past_kf),
        "fut_kf": int(fut_kf),
        "treadmill": bool(args.treadmill),
        "remove_heading": bool(args.remove_heading),
        "fps": float(args.fps),
        "dt": float(1.0 / args.fps),
    }

    success, fail = 0, 0
    failures: List[dict] = []
    for i, cid in enumerate(all_clips, 1):
        out_path = os.path.join(args.output_dir, f"{cid}.npz")
        meta_path = os.path.splitext(out_path)[0] + ".meta.json"
        if not args.overwrite and os.path.isfile(out_path) and os.path.isfile(meta_path):
            success += 1
            continue
        try:
            pos0, pos1 = load_retargeted_positions(args.data_dir, cid)
            pos0, pos1 = _validate_position_pair(cid, pos0, pos1)

            mkr0, mvel0, idx0 = preprocess_for_imdy(
                pos0,
                past_kf=past_kf,
                fut_kf=fut_kf,
                treadmill=args.treadmill,
                remove_heading=args.remove_heading,
                dt=1.0 / args.fps,
            )
            mkr1, mvel1, idx1 = preprocess_for_imdy(
                pos1,
                past_kf=past_kf,
                fut_kf=fut_kf,
                treadmill=args.treadmill,
                remove_heading=args.remove_heading,
                dt=1.0 / args.fps,
            )

            frame_indices = _validate_clip_indices(cid, idx0, idx1)
            n = int(frame_indices.shape[0])

            pred0 = model.predict_clip(mkr0, mvel0, batch_size=args.batch_size)
            pred1 = model.predict_clip(mkr1, mvel1, batch_size=args.batch_size)
            pred0 = _validate_prediction_bundle(cid, "person0", pred0, n, require_contact_logits=True)
            pred1 = _validate_prediction_bundle(cid, "person1", pred1, n, require_contact_logits=True)

            np.savez_compressed(
                out_path,
                torque_p0=pred0["torque"],
                torque_p1=pred1["torque"],
                grf_p0=pred0["grf"],
                grf_p1=pred1["grf"],
                contact_logits_p0=pred0["contact_logits"],
                contact_logits_p1=pred1["contact_logits"],
                frame_indices=frame_indices,
            )
            metadata = {
                "created_at_utc": _timestamp_utc(),
                "clip_id": cid,
                "action_class": parse_action_class(cid),
                "source": {
                    "retargeted_data_dir": os.path.abspath(args.data_dir),
                    "contact_dir": os.path.abspath(args.contact_dir),
                    "contact_json_path": os.path.abspath(
                        os.path.join(args.contact_dir, f"{cid}.json")
                    ),
                },
                "imdy": {
                    "config_path": os.path.abspath(args.imdy_config),
                    "checkpoint_path": os.path.abspath(args.imdy_checkpoint),
                    "imdy_root": os.path.abspath(model.imdy_root),
                    "device_requested": str(args.device),
                    "device_used": str(model.device),
                    "batch_size": int(args.batch_size),
                },
                "sampling": {
                    "seed": int(args.seed),
                    "sample_per_action": int(args.sample_per_action),
                    "max_clips": None if args.max_clips is None else int(args.max_clips),
                },
                "preprocess": preprocess_meta,
                "torque_joint_names": IMDY_TORQUE_JOINT_NAMES,
                "torque_joint_name_source": IMDY_TORQUE_JOINT_NAME_SOURCE,
                "contact_policy": CONTACT_POLICY,
                "expected_shapes": {
                    "torque": [n] + list(EXPECTED_TORQUE_SHAPE),
                    "grf": [n] + list(EXPECTED_GRF_SHAPE),
                    "contact_logits": [n] + list(EXPECTED_CONTACT_LOGIT_SHAPE),
                    "frame_indices": [n],
                },
                "actual_shapes": {
                    "torque_p0": _shape_to_list(pred0["torque"]),
                    "torque_p1": _shape_to_list(pred1["torque"]),
                    "grf_p0": _shape_to_list(pred0["grf"]),
                    "grf_p1": _shape_to_list(pred1["grf"]),
                    "contact_logits_p0": _shape_to_list(pred0["contact_logits"]),
                    "contact_logits_p1": _shape_to_list(pred1["contact_logits"]),
                    "frame_indices": _shape_to_list(frame_indices),
                },
                "sequence": {
                    "num_retargeted_frames": int(pos0.shape[0]),
                    "num_windows_scored": n,
                    "frame_index_start": int(frame_indices[0]),
                    "frame_index_end": int(frame_indices[-1]),
                },
            }
            _write_json(meta_path, metadata)
            success += 1
        except Exception as exc:
            print(f"[infer][WARN] {cid}: {exc}")
            fail += 1
            failures.append({"clip_id": cid, "error": str(exc)})

        if i % args.log_every == 0 or i == len(all_clips):
            print(f"[infer] {i}/{len(all_clips)} (ok={success}, fail={fail})")

    summary = {
        "created_at_utc": _timestamp_utc(),
        "data_dir": os.path.abspath(args.data_dir),
        "contact_dir": os.path.abspath(args.contact_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "n_requested": len(all_clips),
        "n_succeeded": success,
        "n_failed": fail,
        "sampling": {
            "seed": int(args.seed),
            "sample_per_action": int(args.sample_per_action),
            "max_clips": None if args.max_clips is None else int(args.max_clips),
        },
        "preprocess": preprocess_meta,
        "torque_joint_name_source": IMDY_TORQUE_JOINT_NAME_SOURCE,
        "contact_policy": CONTACT_POLICY,
        "failures": failures,
    }
    _write_json(os.path.join(args.output_dir, "_infer_summary.json"), summary)
    print(f"[infer] Done: {success} saved, {fail} failed → {args.output_dir}")


# ===========================================================================
# Subcommand: visualize
# ===========================================================================

def cmd_visualize(args: argparse.Namespace) -> None:
    """Generate 6-panel force + contact overlay plots."""
    _configure_headless_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    _validate_joint_name_mapping()
    _ensure_dir(args.arrays_dir, "Force-array")
    _ensure_dir(args.contact_dir, "InterX mesh contact")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine clip list
    if args.clip_ids:
        clip_ids = [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    else:
        clip_ids = [
            os.path.splitext(os.path.basename(f))[0]
            for f in sorted(os.listdir(args.arrays_dir))
            if f.endswith(".npz")
        ]
        if args.max_clips:
            clip_ids = clip_ids[:args.max_clips]

    print(f"[viz] Generating plots for {len(clip_ids)} clips")

    for cid in clip_ids:
        npz_path = os.path.join(args.arrays_dir, f"{cid}.npz")
        if not os.path.isfile(npz_path):
            print(f"[viz][WARN] No .npz for {cid}, skipping")
            continue

        contact_frames = load_mesh_contact(args.contact_dir, cid)
        if contact_frames is None:
            print(f"[viz][WARN] No mesh contact JSON for {cid}, skipping")
            continue

        data = _load_force_arrays(npz_path)
        metadata = _load_metadata_sidecar(npz_path)
        frame_idx = data["frame_indices"]
        valid_mask, aligned, n_dropped = _align_contact_signals(cid, frame_idx, contact_frames)
        if n_dropped > 0:
            print(f"[viz][WARN] {cid}: dropped {n_dropped} out-of-range frames during alignment")

        torque_p0 = data["torque_p0"][valid_mask]
        torque_p1 = data["torque_p1"][valid_mask]
        grf_p0 = data["grf_p0"][valid_mask]
        grf_p1 = data["grf_p1"][valid_mask]
        x = aligned["frame_indices"]
        N = len(x)
        joint_labels = _joint_labels_for_plot(metadata)

        # Compute per-frame signals
        torque_mag_p0 = np.linalg.norm(torque_p0, axis=-1).mean(axis=-1)  # (N,)
        torque_mag_p1 = np.linalg.norm(torque_p1, axis=-1).mean(axis=-1)
        # Vertical GRF: sum across 24 bodies, take z-component, use timestep 0
        vgrf_p0 = grf_p0[:, 0, :, 2].sum(axis=-1)  # (N,)
        vgrf_p1 = grf_p1[:, 0, :, 2].sum(axis=-1)
        contact_ternary_aligned = aligned["contact_ternary"]
        min_dist_aligned = aligned["min_distance_m"]
        cv_total_aligned = aligned["contact_vertex_count_p1"] + aligned["contact_vertex_count_p2"]

        # Build figure
        fig = plt.figure(figsize=(16, 18))
        gs = GridSpec(6, 1, figure=fig, hspace=0.35)
        axes = [fig.add_subplot(gs[i]) for i in range(6)]

        def shade_contact(ax):
            """Add contact-phase shading."""
            in_contact = False
            start = None
            for j in range(N):
                if contact_ternary_aligned[j] == 2 and not in_contact:
                    in_contact = True
                    start = x[j]
                elif contact_ternary_aligned[j] != 2 and in_contact:
                    in_contact = False
                    ax.axvspan(start, x[j], alpha=0.2, color="red", linewidth=0)
            if in_contact:
                ax.axvspan(start, x[-1], alpha=0.2, color="red", linewidth=0)

            # Barely touching: lighter shading
            in_barely = False
            start_b = None
            for j in range(N):
                if contact_ternary_aligned[j] == 1 and not in_barely:
                    in_barely = True
                    start_b = x[j]
                elif contact_ternary_aligned[j] != 1 and in_barely:
                    in_barely = False
                    ax.axvspan(start_b, x[j], alpha=0.1, color="orange", linewidth=0)
            if in_barely:
                ax.axvspan(start_b, x[-1], alpha=0.1, color="orange", linewidth=0)

        # Panel 1: Mean torque magnitude
        ax = axes[0]
        ax.plot(x, torque_mag_p0, label="Person 0", color="tab:blue", linewidth=0.8)
        ax.plot(x, torque_mag_p1, label="Person 1", color="tab:orange", linewidth=0.8)
        shade_contact(ax)
        ax.set_ylabel("Mean |Torque| (Nm)")
        ax.set_title(f"{cid} — Force-Contact Overlay", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)

        # Panel 2: Vertical GRF
        ax = axes[1]
        ax.plot(x, vgrf_p0, label="Person 0", color="tab:blue", linewidth=0.8)
        ax.plot(x, vgrf_p1, label="Person 1", color="tab:orange", linewidth=0.8)
        shade_contact(ax)
        ax.set_ylabel("Vertical GRF (N)")
        ax.legend(loc="upper right", fontsize=8)

        # Panel 3: Inter-person min distance
        ax = axes[2]
        ax.plot(x, min_dist_aligned, color="tab:green", linewidth=0.8)
        shade_contact(ax)
        ax.set_ylabel("Min Distance (m)")
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.5)

        # Panel 4: Contact vertex count
        ax = axes[3]
        ax.fill_between(x, 0, cv_total_aligned, alpha=0.5, color="tab:red")
        ax.set_ylabel("Contact Vertices")

        # Panel 5: Torque heatmap person 0
        ax = axes[4]
        torque_hm_p0 = np.linalg.norm(torque_p0, axis=-1).T  # (23, N)
        im = ax.imshow(torque_hm_p0, aspect="auto", cmap="hot",
                        extent=[x[0], x[-1], 22.5, -0.5], interpolation="nearest")
        ax.set_ylabel("Joint (P0)")
        ax.set_yticks(range(0, 23, 4))
        ax.set_yticklabels([joint_labels[i] for i in range(0, 23, 4)], fontsize=7)
        plt.colorbar(im, ax=ax, label="Nm", shrink=0.8)

        # Panel 6: Torque heatmap person 1
        ax = axes[5]
        torque_hm_p1 = np.linalg.norm(torque_p1, axis=-1).T
        im = ax.imshow(torque_hm_p1, aspect="auto", cmap="hot",
                        extent=[x[0], x[-1], 22.5, -0.5], interpolation="nearest")
        ax.set_ylabel("Joint (P1)")
        ax.set_xlabel("Frame")
        ax.set_yticks(range(0, 23, 4))
        ax.set_yticklabels([joint_labels[i] for i in range(0, 23, 4)], fontsize=7)
        plt.colorbar(im, ax=ax, label="Nm", shrink=0.8)

        # Legend for contact shading
        legend_elements = [
            Patch(facecolor="red", alpha=0.2, label="Touching/Penetrating"),
            Patch(facecolor="orange", alpha=0.1, label="Barely touching"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9)

        out_path = os.path.join(args.output_dir, f"{cid}_force_contact.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] Saved {out_path}")

    print(f"[viz] Done: {len(clip_ids)} plots → {args.output_dir}")


# ===========================================================================
# Subcommand: analyze
# ===========================================================================

def cmd_analyze(args: argparse.Namespace) -> None:
    """Statistical comparison of force metrics during contact vs non-contact."""
    from scipy import stats

    _validate_joint_name_mapping()
    _ensure_dir(args.arrays_dir, "Force-array")
    _ensure_dir(args.contact_dir, "InterX mesh contact")

    npz_files = sorted([f for f in os.listdir(args.arrays_dir) if f.endswith(".npz")])
    print(f"[analyze] Found {len(npz_files)} .npz files")

    per_clip_results = []
    joint_labels: Optional[List[str]] = None
    skipped = {
        "missing_contact_json": 0,
        "no_valid_alignment": 0,
        "insufficient_contact_split": 0,
        "invalid_array_bundle": 0,
    }

    for fname in npz_files:
        cid = os.path.splitext(fname)[0]
        contact_frames = load_mesh_contact(args.contact_dir, cid)
        if contact_frames is None:
            skipped["missing_contact_json"] += 1
            continue

        npz_path = os.path.join(args.arrays_dir, fname)
        try:
            data = _load_force_arrays(npz_path)
        except Exception as exc:
            print(f"[analyze][WARN] {cid}: invalid array bundle ({exc})")
            skipped["invalid_array_bundle"] += 1
            continue
        metadata = _load_metadata_sidecar(npz_path)
        if joint_labels is None:
            joint_labels = _joint_labels_for_plot(metadata)
        try:
            valid_mask, aligned, n_dropped = _align_contact_signals(
                cid, data["frame_indices"], contact_frames
            )
        except ValueError as exc:
            print(f"[analyze][WARN] {cid}: {exc}")
            skipped["no_valid_alignment"] += 1
            continue

        torque_p0 = data["torque_p0"][valid_mask]
        torque_p1 = data["torque_p1"][valid_mask]
        grf_p0 = data["grf_p0"][valid_mask]
        grf_p1 = data["grf_p1"][valid_mask]
        N = int(aligned["frame_indices"].shape[0])
        contact_mask = aligned["contact_binary"]

        n_contact = int(contact_mask.sum())
        n_noncontact = N - n_contact
        if n_contact < 5 or n_noncontact < 5:
            skipped["insufficient_contact_split"] += 1
            continue  # Skip clips without enough frames in both groups

        # Compute per-frame metrics (averaged over both persons)
        torque_mag = 0.5 * (
            np.linalg.norm(torque_p0, axis=-1).mean(axis=-1)
            + np.linalg.norm(torque_p1, axis=-1).mean(axis=-1)
        )  # (N,)
        torque_bw = torque_mag / (DEFAULT_WEIGHT_KG * GRAVITY)

        vgrf = 0.5 * (
            grf_p0[:, 0, :, 2].sum(axis=-1) + grf_p1[:, 0, :, 2].sum(axis=-1)
        )
        neg_grf = (vgrf < 0).astype(float)

        # Per-joint torque magnitude (both persons combined)
        joint_torque_p0 = np.linalg.norm(torque_p0, axis=-1)  # (N, 23)
        joint_torque_p1 = np.linalg.norm(torque_p1, axis=-1)
        joint_torque_avg = 0.5 * (joint_torque_p0 + joint_torque_p1)  # (N, 23)

        # Split by contact
        contact_torque_bw = torque_bw[contact_mask].mean()
        noncontact_torque_bw = torque_bw[~contact_mask].mean()
        contact_torque_std = torque_bw[contact_mask].std()
        noncontact_torque_std = torque_bw[~contact_mask].std()
        contact_vgrf = np.abs(vgrf[contact_mask]).mean()
        noncontact_vgrf = np.abs(vgrf[~contact_mask]).mean()
        contact_neg_rate = neg_grf[contact_mask].mean()
        noncontact_neg_rate = neg_grf[~contact_mask].mean()

        # Per-joint breakdown
        contact_joint = joint_torque_avg[contact_mask].mean(axis=0)  # (23,)
        noncontact_joint = joint_torque_avg[~contact_mask].mean(axis=0)

        ac = parse_action_class(cid)

        per_clip_results.append({
            "clip_id": cid,
            "action_class": ac,
            "n_frames_total": int(data["frame_indices"].shape[0]),
            "n_frames_aligned": int(N),
            "n_frames_dropped_alignment": int(n_dropped),
            "n_contact": n_contact,
            "n_noncontact": n_noncontact,
            "contact_rate": float(n_contact / N),
            "contact_torque_bw": float(contact_torque_bw),
            "noncontact_torque_bw": float(noncontact_torque_bw),
            "torque_gap_bw": float(contact_torque_bw - noncontact_torque_bw),
            "contact_torque_std": float(contact_torque_std),
            "noncontact_torque_std": float(noncontact_torque_std),
            "contact_vgrf": float(contact_vgrf),
            "noncontact_vgrf": float(noncontact_vgrf),
            "vgrf_gap": float(contact_vgrf - noncontact_vgrf),
            "contact_neg_grf_rate": float(contact_neg_rate),
            "noncontact_neg_grf_rate": float(noncontact_neg_rate),
            "contact_joint_torque": contact_joint.tolist(),
            "noncontact_joint_torque": noncontact_joint.tolist(),
        })

    if not per_clip_results:
        print("[analyze] No clips with sufficient contact/non-contact frames")
        return

    print(f"[analyze] {len(per_clip_results)} clips analyzed")

    # Dataset-level paired tests
    contact_vals = np.array([r["contact_torque_bw"] for r in per_clip_results])
    noncontact_vals = np.array([r["noncontact_torque_bw"] for r in per_clip_results])
    vgrf_c = np.array([r["contact_vgrf"] for r in per_clip_results])
    vgrf_nc = np.array([r["noncontact_vgrf"] for r in per_clip_results])

    def paired_test(a, b, name):
        diff = a - b
        mean_diff = float(np.mean(diff))
        pooled_std = float(np.sqrt(0.5 * (np.var(a) + np.var(b))))
        cohen_d = mean_diff / pooled_std if pooled_std > 1e-12 else 0.0
        try:
            stat, p = stats.wilcoxon(a, b)
        except Exception:
            stat, p = 0.0, 1.0
        return {
            "metric": name,
            "contact_mean": float(np.mean(a)),
            "noncontact_mean": float(np.mean(b)),
            "mean_gap": mean_diff,
            "cohen_d": cohen_d,
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "n_clips": len(a),
        }

    dataset_stats = [
        paired_test(contact_vals, noncontact_vals, "torque_bw"),
        paired_test(vgrf_c, vgrf_nc, "abs_vertical_grf"),
    ]

    # Per-joint gap analysis
    joint_contact = np.array([r["contact_joint_torque"] for r in per_clip_results])  # (C, 23)
    joint_noncontact = np.array([r["noncontact_joint_torque"] for r in per_clip_results])
    joint_gap = (joint_contact - joint_noncontact).mean(axis=0)  # (23,)
    if joint_labels is None:
        joint_labels = _joint_labels_for_plot(None)
    joint_gap_ranked = sorted(
        [(joint_labels[j], float(joint_gap[j])) for j in range(23)],
        key=lambda x: -abs(x[1]),
    )

    output = {
        "created_at_utc": _timestamp_utc(),
        "n_input_npz": len(npz_files),
        "n_clips": len(per_clip_results),
        "processing_summary": skipped,
        "contact_policy": CONTACT_POLICY,
        "joint_name_source": IMDY_TORQUE_JOINT_NAME_SOURCE,
        "joint_names": joint_labels,
        "dataset_stats": dataset_stats,
        "joint_gap_ranked": joint_gap_ranked,
        "per_clip": per_clip_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[analyze] Results → {args.output}")

    # Print summary
    for s in dataset_stats:
        sig = "***" if s["p_value"] < 0.001 else "**" if s["p_value"] < 0.01 else "*" if s["p_value"] < 0.05 else "ns"
        print(f"  {s['metric']}: contact={s['contact_mean']:.4f} vs noncontact={s['noncontact_mean']:.4f} "
              f"(gap={s['mean_gap']:.4f}, d={s['cohen_d']:.3f}, p={s['p_value']:.2e}) {sig}")

    print("  Top 5 joints by torque gap (contact - noncontact):")
    for name, gap in joint_gap_ranked[:5]:
        print(f"    {name}: {gap:+.2f} Nm")


# ===========================================================================
# Subcommand: aggregate
# ===========================================================================

def cmd_aggregate(args: argparse.Namespace) -> None:
    """Per-action-category analysis with plots."""
    _configure_headless_matplotlib()
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.analysis_results, "r") as f:
        results = json.load(f)

    per_clip = results["per_clip"]
    if not per_clip:
        table_path = os.path.join(args.output_dir, "action_summary.json")
        with open(table_path, "w") as f:
            json.dump([], f, indent=2)
        print("[aggregate] No per-clip rows found in analysis results; wrote empty action_summary.json")
        return

    # Group by action class
    by_action: Dict[int, List[dict]] = defaultdict(list)
    for r in per_clip:
        ac = r.get("action_class")
        if ac is not None:
            by_action[ac].append(r)

    if not by_action:
        table_path = os.path.join(args.output_dir, "action_summary.json")
        with open(table_path, "w") as f:
            json.dump([], f, indent=2)
        print("[aggregate] No action labels found in analysis results; wrote empty action_summary.json")
        return

    action_summary = []
    for ac in sorted(by_action.keys()):
        clips = by_action[ac]
        torque_gaps = [r["torque_gap_bw"] for r in clips]
        contact_rates = [r["contact_rate"] for r in clips]
        action_summary.append({
            "action_class": ac,
            "n_clips": len(clips),
            "mean_torque_gap_bw": float(np.mean(torque_gaps)),
            "std_torque_gap_bw": float(np.std(torque_gaps)),
            "mean_contact_rate": float(np.mean(contact_rates)),
        })

    # Save table
    table_path = os.path.join(args.output_dir, "action_summary.json")
    with open(table_path, "w") as f:
        json.dump(action_summary, f, indent=2)

    # Plot 1: Bar chart — torque gap per action
    fig, ax = plt.subplots(figsize=(14, 5))
    acs = [s["action_class"] for s in action_summary]
    gaps = [s["mean_torque_gap_bw"] for s in action_summary]
    stds = [s["std_torque_gap_bw"] for s in action_summary]
    colors = ["tab:red" if g > 0 else "tab:blue" for g in gaps]
    ax.bar(range(len(acs)), gaps, yerr=stds, color=colors, alpha=0.7, capsize=2)
    ax.set_xticks(range(len(acs)))
    ax.set_xticklabels([f"A{a:03d}" for a in acs], rotation=90, fontsize=7)
    ax.set_ylabel("Torque Gap (BW-normalized)\ncontact − non-contact")
    ax.set_title("Per-Action Torque Gap: Contact vs Non-Contact Phases")
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    bar_path = os.path.join(args.output_dir, "torque_gap_by_action.png")
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] Bar chart → {bar_path}")

    # Plot 2: Scatter — contact rate vs torque gap
    fig, ax = plt.subplots(figsize=(8, 6))
    crates = [s["mean_contact_rate"] for s in action_summary]
    ax.scatter(crates, gaps, s=60, alpha=0.7, edgecolors="black", linewidths=0.5)
    for s in action_summary:
        ax.annotate(f"A{s['action_class']:03d}", (s["mean_contact_rate"], s["mean_torque_gap_bw"]),
                    fontsize=6, ha="center", va="bottom")
    ax.set_xlabel("Mean Contact Rate")
    ax.set_ylabel("Mean Torque Gap (BW-normalized)")
    ax.set_title("Contact Rate vs Torque Discrepancy by Action Category")

    # Fit trend line
    if len(crates) > 2:
        z = np.polyfit(crates, gaps, 1)
        xfit = np.linspace(min(crates), max(crates), 100)
        ax.plot(xfit, np.polyval(z, xfit), "r--", alpha=0.5, label=f"slope={z[0]:.3f}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    scatter_path = os.path.join(args.output_dir, "contact_rate_vs_torque_gap.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[aggregate] Scatter → {scatter_path}")

    # Print ranked table
    ranked = sorted(action_summary, key=lambda s: -abs(s["mean_torque_gap_bw"]))
    print(f"\n[aggregate] Actions ranked by |torque gap| (n={len(ranked)}):")
    print(f"  {'Action':>8}  {'N':>4}  {'Contact%':>9}  {'Gap(BW)':>9}")
    print(f"  {'------':>8}  {'--':>4}  {'-------':>9}  {'------':>9}")
    for s in ranked[:15]:
        print(f"  A{s['action_class']:03d}     {s['n_clips']:4d}  "
              f"{s['mean_contact_rate']*100:8.1f}%  {s['mean_torque_gap_bw']:+8.4f}")

    print(f"\n[aggregate] Done → {args.output_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Force-Contact Analysis: ImDy forces × mesh contact overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- infer ---
    sp = sub.add_parser("infer", help="Re-run ImDy, save per-frame arrays as .npz")
    sp.add_argument(
        "--data-dir",
        default=DEFAULT_INTERX_RETARGET_DIR,
        help="InterX retargeted positions directory",
    )
    sp.add_argument("--output-dir", required=True, help="Output directory for .npz files")
    sp.add_argument(
        "--contact-dir",
        default=DEFAULT_INTERX_CONTACT_DIR,
        help="InterX mesh contact JSON directory (used for filtering and alignment)",
    )
    sp.add_argument("--imdy-config", default=DEFAULT_IMDY_CONFIG)
    sp.add_argument("--imdy-checkpoint", default=DEFAULT_IMDY_CHECKPOINT)
    sp.add_argument("--device", default="cuda:0")
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--fps", type=float, default=DEFAULT_FPS)
    sp.add_argument(
        "--past-kf",
        type=int,
        default=None,
        help="Override ImDy past window size (default: from config/checkpoint)",
    )
    sp.add_argument(
        "--fut-kf",
        type=int,
        default=None,
        help="Override ImDy future window size (default: from config/checkpoint)",
    )
    sp.add_argument(
        "--remove-heading",
        action="store_true",
        help="Remove global heading during ImDy preprocessing",
    )
    sp.add_argument(
        "--treadmill",
        dest="treadmill",
        action="store_true",
        help="Use treadmill preprocessing (default)",
    )
    sp.add_argument(
        "--no-treadmill",
        dest="treadmill",
        action="store_false",
        help="Disable treadmill preprocessing",
    )
    sp.set_defaults(treadmill=True)
    sp.add_argument("--sample-per-action", type=int, default=20,
                    help="Clips per action category (0=all)")
    sp.add_argument("--max-clips", type=int, default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--log-every", type=int, default=10)

    # --- visualize ---
    sp = sub.add_parser("visualize", help="Per-clip 6-panel force + contact overlay")
    sp.add_argument("--arrays-dir", required=True, help="Directory with .npz files from infer")
    sp.add_argument("--contact-dir", default=DEFAULT_INTERX_CONTACT_DIR)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--clip-ids", default="", help="Comma-separated clip IDs (default: all)")
    sp.add_argument("--max-clips", type=int, default=None)

    # --- analyze ---
    sp = sub.add_parser("analyze", help="Statistical comparison: contact vs non-contact")
    sp.add_argument("--arrays-dir", required=True)
    sp.add_argument("--contact-dir", default=DEFAULT_INTERX_CONTACT_DIR)
    sp.add_argument("--output", required=True, help="Output JSON path")

    # --- aggregate ---
    sp = sub.add_parser("aggregate", help="Per-action-category force gap analysis")
    sp.add_argument("--analysis-results", required=True, help="JSON from analyze subcommand")
    sp.add_argument("--output-dir", required=True)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "aggregate":
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
