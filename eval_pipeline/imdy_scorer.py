#!/usr/bin/env python3
"""ImDy scoring pipeline for InterHuman clips."""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from eval_pipeline.imdy_preprocessor import preprocess_for_imdy
    from eval_pipeline.imdy_model_wrapper import ImDyWrapper
    from eval_pipeline.imdy_metrics import (
        DEFAULT_WEIGHT_KG,
        aggregate_metric_dicts,
        compare_metric_distributions,
        compute_interaction_metrics,
        compute_person_metrics,
    )
except ModuleNotFoundError:
    from imdy_preprocessor import preprocess_for_imdy
    from imdy_model_wrapper import ImDyWrapper
    from imdy_metrics import (
        DEFAULT_WEIGHT_KG,
        aggregate_metric_dicts,
        compare_metric_distributions,
        compute_interaction_metrics,
        compute_person_metrics,
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _load_pickle_compat(path: str):
    # Compatibility shim for pickles referencing numpy internals.
    import sys as _sys

    if "numpy._core" not in _sys.modules:
        _sys.modules["numpy._core"] = np.core  # type: ignore[attr-defined]
    if "numpy._core.multiarray" not in _sys.modules:
        _sys.modules["numpy._core.multiarray"] = np.core.multiarray  # type: ignore[attr-defined]

    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")
        except Exception:
            # Some files are joblib pickles or use numpy-internal module names.
            f.seek(0)
            try:
                import joblib

                return joblib.load(f)
            except Exception:
                raise


def _detect_layout(data_dir: str) -> str:
    if glob.glob(os.path.join(data_dir, "*_person0.npy")):
        return "retargeted"
    if os.path.isdir(os.path.join(data_dir, "motions_processed", "person1")):
        return "interhuman_raw"

    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if pkl_files:
        sample = _load_pickle_compat(pkl_files[0])
        if (
            isinstance(sample, dict)
            and "person1" in sample
            and isinstance(sample["person1"], dict)
            and "positions_zup" in sample["person1"]
        ):
            return "positions_pkl"

    raise ValueError(
        f"Could not detect supported data layout in {data_dir}. "
        "Expected retargeted npy, InterHuman raw, or positions pkl."
    )


def _list_clip_ids(data_dir: str, layout: str) -> List[str]:
    if layout == "retargeted":
        pat0 = re.compile(r"^(?P<cid>.+)_person0\.npy$")
        pat1 = re.compile(r"^(?P<cid>.+)_person1\.npy$")
        p0 = set()
        p1 = set()

        for fp in glob.glob(os.path.join(data_dir, "*_person0.npy")):
            name = os.path.basename(fp)
            if "_joint_q" in name or "_betas" in name or "_torques" in name:
                continue
            m = pat0.match(name)
            if m:
                p0.add(m.group("cid"))

        for fp in glob.glob(os.path.join(data_dir, "*_person1.npy")):
            name = os.path.basename(fp)
            if "_joint_q" in name or "_betas" in name or "_torques" in name:
                continue
            m = pat1.match(name)
            if m:
                p1.add(m.group("cid"))

        return sorted(p0 & p1)

    if layout == "interhuman_raw":
        p1_dir = os.path.join(data_dir, "motions_processed", "person1")
        p2_dir = os.path.join(data_dir, "motions_processed", "person2")
        c1 = {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(p1_dir, "*.npy"))
        }
        c2 = {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(p2_dir, "*.npy"))
        }
        return sorted(c1 & c2)

    if layout == "positions_pkl":
        ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(data_dir, "*.pkl"))
        ]
        return sorted(ids)

    raise ValueError(f"Unsupported layout: {layout}")


def _load_interhuman_raw_positions(data_dir: str, clip_id: str) -> Tuple[np.ndarray, np.ndarray]:
    p1_path = os.path.join(data_dir, "motions_processed", "person1", f"{clip_id}.npy")
    p2_path = os.path.join(data_dir, "motions_processed", "person2", f"{clip_id}.npy")

    arr1 = np.load(p1_path).astype(np.float32)
    arr2 = np.load(p2_path).astype(np.float32)

    pos1_yup = arr1[:, :66].reshape(-1, 22, 3)
    pos2_yup = arr2[:, :66].reshape(-1, 22, 3)

    trans_matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    pos1_zup = np.einsum("mn,tjn->tjm", trans_matrix, pos1_yup)
    pos2_zup = np.einsum("mn,tjn->tjm", trans_matrix, pos2_yup)
    return pos1_zup, pos2_zup


def _load_clip_positions(
    data_dir: str,
    layout: str,
    clip_id: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    if layout == "retargeted":
        p0 = np.load(os.path.join(data_dir, f"{clip_id}_person0.npy")).astype(np.float32)
        p1 = np.load(os.path.join(data_dir, f"{clip_id}_person1.npy")).astype(np.float32)
        return p0, p1, None

    if layout == "interhuman_raw":
        p0, p1 = _load_interhuman_raw_positions(data_dir, clip_id)
        text_path = os.path.join(data_dir, "annots", f"{clip_id}.txt")
        text = None
        if os.path.isfile(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f if x.strip()]
            text = lines[0] if lines else None
        return p0, p1, text

    if layout == "positions_pkl":
        raw = _load_pickle_compat(os.path.join(data_dir, f"{clip_id}.pkl"))
        p0 = np.asarray(raw["person1"]["positions_zup"], dtype=np.float32)
        p1 = np.asarray(raw["person2"]["positions_zup"], dtype=np.float32)
        text = raw.get("text") if isinstance(raw, dict) else None
        return p0, p1, text

    raise ValueError(f"Unsupported layout: {layout}")


def _canonical_grf_for_metrics(pred: Dict[str, np.ndarray], mask_contact: bool) -> np.ndarray:
    if mask_contact and "grf_masked" in pred:
        return pred["grf_masked"]
    return pred["grf"]


def _score_one_clip(
    clip_id: str,
    pos_person1: np.ndarray,
    pos_person2: np.ndarray,
    model: ImDyWrapper,
    args: argparse.Namespace,
    text: Optional[str] = None,
) -> dict:
    mkr1, mvel1, idx1 = preprocess_for_imdy(
        pos_person1,
        past_kf=args.past_kf,
        fut_kf=args.fut_kf,
        treadmill=True,
        remove_heading=args.remove_heading,
        dt=1.0 / args.fps,
    )
    mkr2, mvel2, idx2 = preprocess_for_imdy(
        pos_person2,
        past_kf=args.past_kf,
        fut_kf=args.fut_kf,
        treadmill=True,
        remove_heading=args.remove_heading,
        dt=1.0 / args.fps,
    )

    n = min(len(idx1), len(idx2))
    if n <= 0:
        raise ValueError("No valid ImDy windows for this clip")

    mkr1, mvel1, idx1 = mkr1[:n], mvel1[:n], idx1[:n]
    mkr2, mvel2, idx2 = mkr2[:n], mvel2[:n], idx2[:n]

    pred1 = model.predict_clip(mkr1, mvel1, batch_size=args.batch_size)
    pred2 = model.predict_clip(mkr2, mvel2, batch_size=args.batch_size)

    grf1 = _canonical_grf_for_metrics(pred1, mask_contact=args.mask_contact)
    grf2 = _canonical_grf_for_metrics(pred2, mask_contact=args.mask_contact)

    center_pos1 = pos_person1[idx1]
    center_pos2 = pos_person2[idx2]

    person1_metrics = compute_person_metrics(
        pred1["torque"],
        grf1,
        center_pos1,
        body_weight_kg=args.weight_kg,
    )
    person2_metrics = compute_person_metrics(
        pred2["torque"],
        grf2,
        center_pos2,
        body_weight_kg=args.weight_kg,
    )

    merged_metrics: Dict[str, float] = {}
    for k, v in person1_metrics.items():
        merged_metrics[f"person1_{k}"] = float(v)
    for k, v in person2_metrics.items():
        merged_metrics[f"person2_{k}"] = float(v)

    if not getattr(args, "per_person_only", False):
        interaction_metrics = compute_interaction_metrics(
            pred1["torque"],
            pred2["torque"],
            grf1,
            grf2,
            center_pos1,
            center_pos2,
            proximity_threshold_m=args.contact_threshold,
        )
        for k, v in interaction_metrics.items():
            merged_metrics[f"interaction_{k}"] = float(v)

    return {
        "clip_id": clip_id,
        "text": text,
        "num_frames_person1": int(pos_person1.shape[0]),
        "num_frames_person2": int(pos_person2.shape[0]),
        "num_windows_scored": int(n),
        "metrics": merged_metrics,
    }


def _run_scoring(args: argparse.Namespace) -> None:
    _ensure_dir(args.output_dir)

    layout = args.dataset_type
    if layout == "auto":
        layout = _detect_layout(args.data_dir)

    clip_ids = _list_clip_ids(args.data_dir, layout)
    if args.clip_ids:
        requested = {x.strip() for x in args.clip_ids.split(",") if x.strip()}
        clip_ids = [cid for cid in clip_ids if cid in requested]
    if args.max_clips is not None:
        clip_ids = clip_ids[: args.max_clips]
    if not clip_ids:
        raise RuntimeError("No clips found after filtering")

    model = ImDyWrapper(
        config_path=args.imdy_config,
        checkpoint_path=args.imdy_checkpoint,
        device=args.device,
        use_contact_mask=args.mask_contact,
    )
    print(f"[ImDy] layout={layout}, clips={len(clip_ids)}, device={model.device}")

    per_clip_metrics: List[Dict[str, float]] = []
    failures: List[dict] = []

    for idx, clip_id in enumerate(clip_ids, start=1):
        try:
            pos1, pos2, text = _load_clip_positions(args.data_dir, layout, clip_id)
            record = _score_one_clip(clip_id, pos1, pos2, model, args, text=text)
            _write_json(os.path.join(args.output_dir, f"{clip_id}_imdy.json"), record)
            per_clip_metrics.append(record["metrics"])
            if idx % max(args.log_every, 1) == 0 or idx == len(clip_ids):
                print(f"[ImDy] {idx}/{len(clip_ids)} clips scored")
        except Exception as exc:
            failures.append({"clip_id": clip_id, "error": str(exc)})
            print(f"[ImDy][WARN] clip {clip_id} failed: {exc}")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": os.path.abspath(args.data_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "dataset_layout": layout,
        "dataset_type": args.dataset_type,
        "imdy_config": os.path.abspath(args.imdy_config),
        "imdy_checkpoint": os.path.abspath(args.imdy_checkpoint),
        "device": str(model.device),
        "clips_requested": len(clip_ids),
        "clips_succeeded": len(per_clip_metrics),
        "clips_failed": len(failures),
        "failures": failures,
        "metrics": aggregate_metric_dicts(per_clip_metrics),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    _write_json(summary_path, summary)
    print(f"[ImDy] Summary written to {summary_path}")


def _load_metric_rows_from_dir(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for fp in sorted(glob.glob(os.path.join(path, "*_imdy.json"))):
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        metrics = obj.get("metrics", {})
        if isinstance(metrics, dict):
            rows.append(metrics)
    return rows


def _run_compare(args: argparse.Namespace) -> None:
    gt_rows = _load_metric_rows_from_dir(args.gt_dir)
    gen_rows = _load_metric_rows_from_dir(args.gen_dir)
    if not gt_rows:
        raise RuntimeError(f"No *_imdy.json metrics found in GT dir: {args.gt_dir}")
    if not gen_rows:
        raise RuntimeError(f"No *_imdy.json metrics found in generated dir: {args.gen_dir}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gt_dir": os.path.abspath(args.gt_dir),
        "gen_dir": os.path.abspath(args.gen_dir),
        "gt_clip_count": len(gt_rows),
        "gen_clip_count": len(gen_rows),
        "distribution_comparison": compare_metric_distributions(gt_rows, gen_rows),
    }
    _ensure_dir(os.path.dirname(args.output) or ".")
    _write_json(args.output, report)
    print(f"[ImDy] Comparison report written to {args.output}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ImDy scorer for InterHuman")
    p.add_argument("--compare", action="store_true", help="Run GT vs generated comparison mode")

    p.add_argument("--data-dir", default="", help="Input data directory (scoring mode)")
    p.add_argument("--output-dir", default="", help="Output dir for per-clip files and summary")
    p.add_argument(
        "--dataset-type",
        default="auto",
        choices=["auto", "retargeted", "interhuman_raw", "positions_pkl"],
        help="Input data layout",
    )

    p.add_argument("--imdy-config", default="prepare5/ImDy/config/IDFD_mkr.yml")
    p.add_argument("--imdy-checkpoint", default="prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt")
    p.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    p.add_argument("--batch-size", type=int, default=256)

    p.add_argument("--past-kf", type=int, default=2)
    p.add_argument("--fut-kf", type=int, default=2)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--weight-kg", type=float, default=DEFAULT_WEIGHT_KG)
    p.add_argument("--remove-heading", action="store_true", help="Apply heading-invariant preprocessing")
    p.add_argument("--mask-contact", action="store_true", help="Mask GRF by predicted contact logits")
    p.add_argument("--contact-threshold", type=float, default=1.0)
    p.add_argument("--per-person-only", action="store_true",
                    help="Skip interaction metrics, score each person independently")

    p.add_argument("--clip-ids", default="", help="Comma-separated clip ids")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--gt-dir", default="", help="GT metrics dir for --compare")
    p.add_argument("--gen-dir", default="", help="Generated metrics dir for --compare")
    p.add_argument("--output", default="", help="Comparison JSON path for --compare")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.compare:
        if not args.gt_dir or not args.gen_dir or not args.output:
            parser.error("--compare requires --gt-dir, --gen-dir, and --output")
        _run_compare(args)
        return

    if not args.data_dir or not args.output_dir:
        parser.error("Scoring mode requires --data-dir and --output-dir")
    _run_scoring(args)


if __name__ == "__main__":
    main()
