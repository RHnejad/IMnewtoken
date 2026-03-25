"""
Plot relation between skyhook root force and foot-ground contact area.

This script reads skyhook metrics NPZ files (which already include joint_q_used and
root_force_l2), reconstructs the per-subject Newton model, and computes per-frame
foot-ground contact patch area from rigid contact points.

Outputs per person:
  - <clip>_personX_force_vs_contact_area.csv
  - <clip>_personX_force_vs_contact_area.png
  - <clip>_personX_force_vs_contact_area.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import warp as wp

wp.config.verbose = False

import newton

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare2.compute_skyhook_metrics import build_single_person_model  # noqa: E402


def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_person_npz(metrics_dir: Path, clip: str, person: int) -> Dict[str, np.ndarray]:
    npz_path = metrics_dir / f"{clip}_person{person}_skyhook_metrics.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {npz_path}")

    with np.load(npz_path) as d:
        if "joint_q_used" not in d:
            raise KeyError(f"{npz_path} is missing joint_q_used")
        if "betas" not in d:
            raise KeyError(f"{npz_path} is missing betas")
        if "root_force_l2" not in d:
            raise KeyError(f"{npz_path} is missing root_force_l2")

        frame = np.asarray(d["frame"]).astype(np.int32) if "frame" in d else None
        joint_q = np.asarray(d["joint_q_used"]).astype(np.float32)
        betas = np.asarray(d["betas"]).astype(np.float64)
        force = np.asarray(d["root_force_l2"]).astype(np.float32)

    T = min(joint_q.shape[0], force.shape[0], len(frame) if frame is not None else joint_q.shape[0])
    out_frame = np.arange(T, dtype=np.int32) if frame is None else frame[:T]
    return {
        "npz_path": np.asarray([str(npz_path)]),
        "frame": out_frame,
        "joint_q": joint_q[:T],
        "betas": betas,
        "root_force_l2": force[:T],
    }


def _shape_groups(model) -> Tuple[Set[int], Set[int], Set[int]]:
    labels = list(model.shape_label)
    left_ids: Set[int] = set()
    right_ids: Set[int] = set()
    ground_ids: Set[int] = set()

    for i, lbl in enumerate(labels):
        ll = lbl.lower()
        if "ground" in ll:
            ground_ids.add(i)
        if ("l_ankle" in ll) or ("l_toe" in ll):
            left_ids.add(i)
        if ("r_ankle" in ll) or ("r_toe" in ll):
            right_ids.add(i)

    if not ground_ids:
        raise RuntimeError("Could not find ground shape(s) in model.shape_label")
    if not left_ids and not right_ids:
        raise RuntimeError("Could not find foot shape IDs (ankle/toe) in model.shape_label")
    return left_ids, right_ids, ground_ids


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_area_xy(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 0.0

    # Remove duplicates to avoid degenerate hull construction.
    pts = np.unique(np.round(points_xy.astype(np.float64), 7), axis=0)
    if pts.shape[0] < 3:
        return 0.0

    pts_list = sorted((float(p[0]), float(p[1])) for p in pts)
    pts_np = [np.array([x, y], dtype=np.float64) for x, y in pts_list]

    lower: List[np.ndarray] = []
    for p in pts_np:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts_np):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    poly = np.stack(hull, axis=0)
    x = poly[:, 0]
    y = poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area


def _extract_ground_contact_xy(
    *,
    shape0: np.ndarray,
    shape1: np.ndarray,
    point0: np.ndarray,
    point1: np.ndarray,
    left_ids: Set[int],
    right_ids: Set[int],
    ground_ids: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    left_pts: List[np.ndarray] = []
    right_pts: List[np.ndarray] = []

    for s0, s1, p0, p1 in zip(shape0, shape1, point0, point1, strict=False):
        foot_id = None
        ground_pt = None
        if int(s0) in ground_ids and (int(s1) in left_ids or int(s1) in right_ids):
            foot_id = int(s1)
            ground_pt = p0
        elif int(s1) in ground_ids and (int(s0) in left_ids or int(s0) in right_ids):
            foot_id = int(s0)
            ground_pt = p1

        if foot_id is None or ground_pt is None:
            continue

        if foot_id in left_ids:
            left_pts.append(np.asarray(ground_pt[:2], dtype=np.float64))
        elif foot_id in right_ids:
            right_pts.append(np.asarray(ground_pt[:2], dtype=np.float64))

    left_xy = np.stack(left_pts, axis=0) if left_pts else np.zeros((0, 2), dtype=np.float64)
    right_xy = np.stack(right_pts, axis=0) if right_pts else np.zeros((0, 2), dtype=np.float64)
    return left_xy, right_xy


def _compute_contact_area_series(
    *,
    model,
    joint_q: np.ndarray,
    left_ids: Set[int],
    right_ids: Set[int],
    ground_ids: Set[int],
    device: str,
) -> Dict[str, np.ndarray]:
    T = int(joint_q.shape[0])
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    area_left = np.zeros((T,), dtype=np.float32)
    area_right = np.zeros((T,), dtype=np.float32)
    n_left = np.zeros((T,), dtype=np.int32)
    n_right = np.zeros((T,), dtype=np.int32)

    for t in range(T):
        state.joint_q = wp.array(joint_q[t], dtype=wp.float32, device=device)
        jqd.zero_()
        newton.eval_fk(model, state.joint_q, jqd, state)
        contacts = model.collide(state)

        n_contacts = int(contacts.rigid_contact_count.numpy()[0])
        if n_contacts <= 0:
            continue

        s0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
        s1 = contacts.rigid_contact_shape1.numpy()[:n_contacts]
        p0 = contacts.rigid_contact_point0.numpy()[:n_contacts]
        p1 = contacts.rigid_contact_point1.numpy()[:n_contacts]

        left_xy, right_xy = _extract_ground_contact_xy(
            shape0=s0,
            shape1=s1,
            point0=p0,
            point1=p1,
            left_ids=left_ids,
            right_ids=right_ids,
            ground_ids=ground_ids,
        )

        n_left[t] = int(left_xy.shape[0])
        n_right[t] = int(right_xy.shape[0])
        area_left[t] = np.float32(_convex_hull_area_xy(left_xy))
        area_right[t] = np.float32(_convex_hull_area_xy(right_xy))

    area_total = area_left + area_right
    n_total = n_left + n_right
    return {
        "contact_area_left_m2": area_left,
        "contact_area_right_m2": area_right,
        "contact_area_total_m2": area_total.astype(np.float32),
        "contact_points_left": n_left,
        "contact_points_right": n_right,
        "contact_points_total": n_total,
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    da = float(np.linalg.norm(aa))
    db = float(np.linalg.norm(bb))
    if da <= 0.0 or db <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / (da * db))


def _save_csv(
    out_csv: Path,
    frame: np.ndarray,
    force: np.ndarray,
    area: Dict[str, np.ndarray],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "root_force_l2_N",
                "contact_area_total_m2",
                "contact_area_left_m2",
                "contact_area_right_m2",
                "contact_points_total",
                "contact_points_left",
                "contact_points_right",
            ]
        )
        for i in range(frame.shape[0]):
            w.writerow(
                [
                    int(frame[i]),
                    float(force[i]),
                    float(area["contact_area_total_m2"][i]),
                    float(area["contact_area_left_m2"][i]),
                    float(area["contact_area_right_m2"][i]),
                    int(area["contact_points_total"][i]),
                    int(area["contact_points_left"][i]),
                    int(area["contact_points_right"][i]),
                ]
            )


def _save_plot(
    out_png: Path,
    *,
    clip: str,
    person: int,
    frame: np.ndarray,
    force: np.ndarray,
    area_total_m2: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    area_cm2 = area_total_m2 * 1.0e4
    safe_force = np.maximum(force, 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"clip {clip} person {person} |F_root| vs foot-ground contact area")

    ax0 = axes[0]
    ax0.plot(frame, safe_force, color="#d62728", linewidth=1.2, label="|F_root| (N)")
    ax0.set_yscale("log")
    ax0.set_ylabel("|F_root| (N, log)")
    ax0.grid(alpha=0.25, which="both")
    ax0b = ax0.twinx()
    ax0b.plot(frame, area_cm2, color="#1f77b4", linewidth=1.2, label="contact area (cm^2)")
    ax0b.set_ylabel("Contact Area (cm^2)")
    ax0.set_xlabel("Frame")
    ax0.set_title("Per-frame force and contact area")

    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax0b.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, loc="upper right")

    ax1 = axes[1]
    sc = ax1.scatter(
        area_cm2,
        safe_force,
        c=frame,
        s=18,
        alpha=0.75,
        cmap="viridis",
        edgecolors="none",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Foot Contact Area (cm^2)")
    ax1.set_ylabel("|F_root| (N, log)")
    ax1.set_title("Force vs contact area (color = frame)")
    ax1.grid(alpha=0.25, which="both")
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("Frame")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _person_list(person_arg: str) -> List[int]:
    if person_arg == "both":
        return [0, 1]
    return [int(person_arg)]


def main():
    parser = argparse.ArgumentParser(
        description="Plot |F_root| against foot-ground contact area from skyhook metrics."
    )
    parser.add_argument("--clip", type=str, required=True)
    parser.add_argument(
        "--person",
        type=str,
        default="both",
        choices=["0", "1", "both"],
        help="Person index or both.",
    )
    parser.add_argument("--metrics-dir", type=str, default="data/test/skyhook")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plot/CSV outputs (default: metrics-dir).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Newton/Warp device, e.g. cuda:0 or cpu.",
    )
    args = parser.parse_args()

    metrics_dir = _resolve(args.metrics_dir)
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics-dir not found: {metrics_dir}")
    output_dir = _resolve(args.output_dir) if args.output_dir else metrics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    people = _person_list(args.person)
    for p in people:
        raw = _load_person_npz(metrics_dir, args.clip, p)
        frame = raw["frame"]
        joint_q = raw["joint_q"]
        force = raw["root_force_l2"]
        betas = raw["betas"]

        model = build_single_person_model(betas, device=args.device, with_ground=True)
        left_ids, right_ids, ground_ids = _shape_groups(model)
        area = _compute_contact_area_series(
            model=model,
            joint_q=joint_q,
            left_ids=left_ids,
            right_ids=right_ids,
            ground_ids=ground_ids,
            device=args.device,
        )

        base = f"{args.clip}_person{p}_force_vs_contact_area"
        out_csv = output_dir / f"{base}.csv"
        out_png = output_dir / f"{base}.png"
        out_json = output_dir / f"{base}.json"

        _save_csv(out_csv, frame, force, area)
        _save_plot(
            out_png,
            clip=args.clip,
            person=p,
            frame=frame,
            force=force,
            area_total_m2=area["contact_area_total_m2"],
        )

        valid = np.isfinite(force) & np.isfinite(area["contact_area_total_m2"])
        corr_raw = _pearson(force[valid], area["contact_area_total_m2"][valid]) if valid.any() else float("nan")
        corr_log = (
            _pearson(np.log10(np.maximum(force[valid], 1e-9)), area["contact_area_total_m2"][valid])
            if valid.any()
            else float("nan")
        )
        summary = {
            "clip": args.clip,
            "person": p,
            "n_frames": int(frame.shape[0]),
            "device": args.device,
            "metrics_npz": str(raw["npz_path"][0]),
            "contact_area_mean_m2": float(np.mean(area["contact_area_total_m2"])),
            "contact_area_max_m2": float(np.max(area["contact_area_total_m2"])),
            "force_mean_N": float(np.mean(force)),
            "force_max_N": float(np.max(force)),
            "pearson_force_vs_area": corr_raw,
            "pearson_log10force_vs_area": corr_log,
            "csv_path": str(out_csv),
            "png_path": str(out_png),
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[clip {args.clip} p{p}] wrote: {out_csv.name}, {out_png.name}, {out_json.name} "
            f"| corr(force, area)={corr_raw:.4f} corr(log10(force), area)={corr_log:.4f}"
        )


if __name__ == "__main__":
    main()
