"""
Render clip-level skyhook diagnostics to MP4 with a Newton-like 3D view.

This creates one video combining:
  - 3D skeleton playback (same world convention as prepare2/visualize_newton.py)
  - Per-frame root residual force ||F_root||_2
  - Per-frame MPJPE

Usage:
  python prepare2/visualize_skyhook_mp4.py \
      --clip 1000 \
      --dataset interhuman \
      --metrics-dir data/test/skyhook \
      --output data/test/skyhook/1000_skyhook_newton_like.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# 22-joint chain used across prepare scripts
BONES = [
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),
]

PERSON_STYLES = {
    0: {"color": "#1f77b4", "alpha": 1.0, "label": "person0"},
    1: {"color": "#d62728", "alpha": 0.8, "label": "person1"},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Ensure local project imports (prepare2.*) are resolvable regardless of cwd.
PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _default_retarget_dir(dataset: str) -> Path:
    return PROJECT_ROOT / "data" / "retargeted_v2" / dataset


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _align_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    src_len = int(arr.shape[0])
    if src_len == target_len:
        return arr
    if src_len > target_len and src_len % target_len == 0:
        stride = src_len // target_len
        return arr[::stride][:target_len]
    idx = np.linspace(0, src_len - 1, target_len).astype(np.int64)
    return arr[idx]


def _detect_persons(clip: str, metrics_dir: Path, person: int | None) -> List[int]:
    if person is not None:
        return [person]
    found: List[int] = []
    for p in (0, 1):
        npz_path = metrics_dir / f"{clip}_person{p}_skyhook_metrics.npz"
        if npz_path.exists():
            found.append(p)
    return found


def _align_positions_to_metrics(pos: np.ndarray, frame_idx: np.ndarray, target_len: int) -> np.ndarray:
    src_len = int(pos.shape[0])
    frame_int = np.asarray(frame_idx, dtype=np.int64)

    # Prefer explicit metric frame indices if they are valid for the source clip.
    if frame_int.size == target_len:
        if np.all(frame_int >= 0) and np.all(frame_int < src_len):
            if np.all(np.diff(frame_int) >= 0):
                return pos[frame_int]

    if src_len == target_len:
        return pos
    if src_len > target_len and src_len % target_len == 0:
        stride = src_len // target_len
        return pos[::stride][:target_len]

    idx = np.linspace(0, src_len - 1, target_len).astype(np.int64)
    return pos[idx]


def _load_joint_source_hint(metrics_dir: Path, clip: str, person: int) -> str:
    jpath = metrics_dir / f"{clip}_person{person}_skyhook_metrics.json"
    if not jpath.exists():
        return "unknown"
    try:
        import json

        with open(jpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("joint_source", "unknown"))
    except Exception:
        return "unknown"


def _try_load_exact_fk_positions(
    *,
    clip: str,
    person: int,
    dataset: str,
    target_len: int,
    metrics_dir: Path,
    retarget_dir: Path,
    raw_data_dir: Optional[Path],
    device: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try to reconstruct the exact trajectory used in metric computation:
      joint_q -> FK on per-subject Newton model.
    Returns (positions_or_none, error_or_none).
    """
    try:
        import warnings

        import warp as wp

        wp.config.verbose = False
        warnings.filterwarnings("ignore", message="Custom attribute")

        import newton
        from prepare2.retarget import (
            extract_positions_from_fk,
            load_interhuman_clip,
            load_interx_clip,
            smplx_to_joint_q,
        )
    except Exception as exc:
        return None, f"warp/newton unavailable for exact FK: {exc}"

    jq_path = retarget_dir / f"{clip}_person{person}_joint_q.npy"
    betas_path = retarget_dir / f"{clip}_person{person}_betas.npy"
    joint_q = None
    betas = None

    if jq_path.exists() and betas_path.exists():
        joint_q = np.load(jq_path).astype(np.float32)
        betas = np.load(betas_path).astype(np.float64)
    else:
        if raw_data_dir is None:
            return None, "No retargeted joint_q/betas and no raw_data_dir provided."
        if dataset == "interhuman":
            persons_raw = load_interhuman_clip(str(raw_data_dir), clip)
        else:
            persons_raw = load_interx_clip(str(raw_data_dir), clip)
        if persons_raw is None or person >= len(persons_raw):
            return None, f"Raw clip not found for {dataset}:{clip}."
        pdata = persons_raw[person]
        joint_q = smplx_to_joint_q(
            pdata["root_orient"], pdata["pose_body"], pdata["trans"], pdata["betas"]
        ).astype(np.float32)
        betas = np.asarray(pdata["betas"], dtype=np.float64)

    try:
        joint_q_aligned = _align_time(joint_q, target_len)
        xml_path = PROJECT_ROOT / "prepare2" / "xml_cache"
        os.makedirs(xml_path, exist_ok=True)
        # Use same model family as metric calculation: per-subject SMPL + ground.
        from prepare2.retarget import get_or_create_xml

        smpl_xml = get_or_create_xml(betas, cache_dir=str(xml_path))
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        if hasattr(builder, "add_mjcf"):
            builder.add_mjcf(smpl_xml, enable_self_collisions=False)
        else:
            # Compatibility path for Newton builds exposing parse_mjcf utility.
            if not hasattr(newton, "utils") or not hasattr(newton.utils, "parse_mjcf"):
                return None, "Neither builder.add_mjcf nor newton.utils.parse_mjcf is available."
            newton.utils.parse_mjcf(
                str(smpl_xml),
                builder,
                enable_self_collisions=False,
                up_axis=newton.Axis.Z,
                parse_visuals_as_colliders=False,
            )
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        pred = extract_positions_from_fk(model, joint_q_aligned, device=device).astype(np.float32)
    except Exception as exc:
        return None, f"Exact FK build/eval failed: {exc}"

    if pred.shape[0] != target_len:
        pred = _align_time(pred, target_len)

    return pred, None


def _load_data(
    clip: str,
    dataset: str,
    persons: List[int],
    metrics_dir: Path,
    retarget_dir: Path,
    raw_data_dir: Optional[Path],
    prefer_exact: bool,
    require_exact: bool,
    device: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for p in persons:
        npz_path = metrics_dir / f"{clip}_person{p}_skyhook_metrics.npz"
        pos_path = retarget_dir / f"{clip}_person{p}.npy"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {npz_path}")
        if not pos_path.exists():
            raise FileNotFoundError(f"Missing positions file: {pos_path}")

        with np.load(npz_path) as d:
            frame = np.asarray(d["frame"]).astype(np.int32)
            root_force_l2 = np.asarray(d["root_force_l2"]).astype(np.float32)
            mpjpe = np.asarray(d["mpjpe_per_frame_m"]).astype(np.float32)

        T = min(len(frame), len(root_force_l2), len(mpjpe))
        frame = frame[:T]
        root_force_l2 = root_force_l2[:T]
        mpjpe = mpjpe[:T]
        joint_source = _load_joint_source_hint(metrics_dir, clip, p)

        pos_aligned = None
        exact_err = None
        if prefer_exact:
            pos_aligned, exact_err = _try_load_exact_fk_positions(
                clip=clip,
                person=p,
                dataset=dataset,
                target_len=T,
                metrics_dir=metrics_dir,
                retarget_dir=retarget_dir,
                raw_data_dir=raw_data_dir,
                device=device,
            )

        if pos_aligned is None:
            if require_exact:
                raise RuntimeError(
                    f"Exact trajectory required but unavailable for person {p}: {exact_err}"
                )
            if not pos_path.exists():
                raise FileNotFoundError(
                    f"Missing fallback positions file: {pos_path}. "
                    f"Exact FK error: {exact_err}"
                )
            pos = np.load(pos_path).astype(np.float32)
            if pos.ndim != 3 or pos.shape[1:] != (22, 3):
                raise ValueError(f"Unexpected position shape in {pos_path}: {pos.shape}")
            pos_aligned = _align_positions_to_metrics(pos, frame, T)
            position_source = "retarget_positions_fallback"
        else:
            position_source = "exact_metric_fk"

        out[p] = {
            "frame": frame,
            "root_force_l2": root_force_l2,
            "mpjpe_m": mpjpe,
            "pos": pos_aligned,
            "joint_source": joint_source,
            "position_source": position_source,
        }
    return out


def _compute_bounds(data: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, float]:
    all_pos = np.concatenate([v["pos"] for v in data.values()], axis=0)  # (sumT, 22, 3)
    mins = all_pos.reshape(-1, 3).min(axis=0)
    maxs = all_pos.reshape(-1, 3).max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-6)
    max_span = float(np.max(span)) * 0.65
    return {
        "cx": float(center[0]),
        "cy": float(center[1]),
        "cz": float(center[2]),
        "half": max_span,
        "zmin": float(mins[2]),
    }


def render_video(
    clip: str,
    dataset: str,
    data: Dict[int, Dict[str, np.ndarray]],
    output_mp4: Path,
    fps: int,
    dpi: int,
    focus_person: Optional[int],
    ghost_extremes: bool,
):
    persons = sorted(data.keys())
    if not persons:
        raise ValueError("No persons available to render.")

    if focus_person is None:
        focus_person = max(
            persons,
            key=lambda p: float(np.max(data[p]["root_force_l2"])),
        )
    if focus_person not in data:
        focus_person = persons[0]

    T = min(v["pos"].shape[0] for v in data.values())
    bounds = _compute_bounds(data)
    half = bounds["half"]
    focus_force = np.asarray(data[focus_person]["root_force_l2"][:T], dtype=np.float32)
    low_idx = int(np.argmin(focus_force))
    high_idx = int(np.argmax(focus_force))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.25,
        hspace=0.28,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_force = fig.add_subplot(gs[0, 1])
    ax_mpjpe = fig.add_subplot(gs[1, 1], sharex=ax_force)

    fig.suptitle(f"Clip {clip} ({dataset}) - Skyhook + MPJPE", fontsize=14)

    # Newton-like camera: Z-up, look toward +Y with slight downward tilt.
    ax3d.view_init(elev=15.0, azim=-90.0)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_box_aspect((1.0, 1.0, 1.0))
    ax3d.grid(True, alpha=0.2)

    ax3d.set_xlim(bounds["cx"] - half, bounds["cx"] + half)
    ax3d.set_ylim(bounds["cy"] - half, bounds["cy"] + half)
    ax3d.set_zlim(max(0.0, bounds["zmin"] - 0.05), bounds["cz"] + half)

    # Draw a translucent ground patch near z=0 for spatial reference.
    gx = np.linspace(bounds["cx"] - half, bounds["cx"] + half, 2)
    gy = np.linspace(bounds["cy"] - half, bounds["cy"] + half, 2)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    ax3d.plot_surface(GX, GY, GZ, color="#bbbbbb", alpha=0.12, linewidth=0)

    # Prepare skeleton artists
    skeleton_lines: Dict[int, List] = {}
    joint_scatters: Dict[int, object] = {}
    for p in persons:
        style = PERSON_STYLES.get(p, {"color": "#333333", "alpha": 0.8, "label": f"person{p}"})
        lines = []
        for _ in BONES:
            (line,) = ax3d.plot([], [], [], color=style["color"], linewidth=2.1, alpha=style["alpha"])
            lines.append(line)
        skeleton_lines[p] = lines
        scatter = ax3d.scatter([], [], [], s=18, color=style["color"], alpha=style["alpha"], label=style["label"])
        joint_scatters[p] = scatter

    ax3d.legend(loc="upper left")
    info_text = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes, fontsize=10)

    if ghost_extremes:
        low_pose = data[focus_person]["pos"][low_idx]
        high_pose = data[focus_person]["pos"][high_idx]
        for i, j in BONES:
            ax3d.plot(
                [float(low_pose[i, 0]), float(low_pose[j, 0])],
                [float(low_pose[i, 1]), float(low_pose[j, 1])],
                [float(low_pose[i, 2]), float(low_pose[j, 2])],
                color="#2ca02c",
                linewidth=1.3,
                alpha=0.25,
            )
            ax3d.plot(
                [float(high_pose[i, 0]), float(high_pose[j, 0])],
                [float(high_pose[i, 1]), float(high_pose[j, 1])],
                [float(high_pose[i, 2]), float(high_pose[j, 2])],
                color="#ff7f0e",
                linewidth=1.3,
                alpha=0.25,
            )
        ax3d.text2D(
            0.02,
            0.90,
            (
                f"Ghost poses (focus p{focus_person}): "
                f"LOW@{low_idx} ({focus_force[low_idx]:.2e}N, green), "
                f"HIGH@{high_idx} ({focus_force[high_idx]:.2e}N, orange)"
            ),
            transform=ax3d.transAxes,
            fontsize=9,
            color="#333333",
        )

    # Metric traces
    metric_x = np.arange(T, dtype=np.int32)
    force_markers = {}
    mpjpe_markers = {}
    for p in persons:
        style = PERSON_STYLES.get(p, {"color": "#333333", "label": f"person{p}"})
        force = np.maximum(data[p]["root_force_l2"][:T], 1e-9)
        mpjpe_cm = data[p]["mpjpe_m"][:T] * 100.0
        ax_force.plot(metric_x, force, color=style["color"], linewidth=1.6, label=style["label"])
        ax_mpjpe.plot(metric_x, mpjpe_cm, color=style["color"], linewidth=1.6, label=style["label"])
        (fm,) = ax_force.plot([], [], "o", color=style["color"], markersize=5)
        (mm,) = ax_mpjpe.plot([], [], "o", color=style["color"], markersize=5)
        force_markers[p] = fm
        mpjpe_markers[p] = mm

    # Explicit high/low force markers on focus person
    ax_force.plot(
        [low_idx],
        [max(float(focus_force[low_idx]), 1e-9)],
        marker="o",
        markersize=7,
        color="#2ca02c",
        linestyle="None",
        label=f"focus low@{low_idx}",
    )
    ax_force.plot(
        [high_idx],
        [max(float(focus_force[high_idx]), 1e-9)],
        marker="o",
        markersize=7,
        color="#ff7f0e",
        linestyle="None",
        label=f"focus high@{high_idx}",
    )

    ax_force.set_yscale("log")
    ax_force.set_ylabel("||F_root|| (N, log)")
    ax_force.set_title("External Residual Force (Skyhook)")
    ax_force.grid(True, alpha=0.25, which="both")
    ax_force.legend(loc="upper right")

    ax_mpjpe.set_ylabel("MPJPE (cm)")
    ax_mpjpe.set_xlabel("Frame")
    ax_mpjpe.set_title("MPJPE")
    ax_mpjpe.grid(True, alpha=0.25)
    ax_mpjpe.legend(loc="upper right")

    force_cursor = ax_force.axvline(0, color="black", linestyle="--", alpha=0.35)
    mpjpe_cursor = ax_mpjpe.axvline(0, color="black", linestyle="--", alpha=0.35)

    def _update(frame_idx: int):
        frame_idx = int(frame_idx)
        for p in persons:
            pos = data[p]["pos"][frame_idx]
            for li, (i, j) in enumerate(BONES):
                xs = [float(pos[i, 0]), float(pos[j, 0])]
                ys = [float(pos[i, 1]), float(pos[j, 1])]
                zs = [float(pos[i, 2]), float(pos[j, 2])]
                skeleton_lines[p][li].set_data_3d(xs, ys, zs)
            joint_scatters[p]._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

            fval = float(data[p]["root_force_l2"][frame_idx])
            mval = float(data[p]["mpjpe_m"][frame_idx] * 100.0)
            force_markers[p].set_data([frame_idx], [max(fval, 1e-9)])
            if np.isfinite(mval):
                mpjpe_markers[p].set_data([frame_idx], [mval])
            else:
                mpjpe_markers[p].set_data([], [])

        force_cursor.set_xdata([frame_idx, frame_idx])
        mpjpe_cursor.set_xdata([frame_idx, frame_idx])

        # compact status line
        tag = ""
        if frame_idx == high_idx:
            tag = " [HIGH FORCE]"
        elif frame_idx == low_idx:
            tag = " [LOW FORCE]"
        parts = [f"Frame {frame_idx}/{T - 1}{tag}"]
        for p in persons:
            fval = float(data[p]["root_force_l2"][frame_idx])
            mval = float(data[p]["mpjpe_m"][frame_idx] * 100.0)
            if np.isfinite(mval):
                parts.append(f"p{p}: |F|={fval:.2e} N, MPJPE={mval:.2f} cm")
            else:
                parts.append(f"p{p}: |F|={fval:.2e} N, MPJPE=n/a")
        info_text.set_text(" | ".join(parts))
        return []

    anim = FuncAnimation(
        fig,
        _update,
        frames=T,
        interval=max(1, int(round(1000.0 / max(fps, 1)))),
        blit=False,
    )

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving MP4 -> {output_mp4}")
    anim.save(str(output_mp4), writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render skyhook diagnostics MP4.")
    parser.add_argument("--clip", type=str, default="1000")
    parser.add_argument("--dataset", type=str, default="interhuman", choices=["interhuman", "interx"])
    parser.add_argument("--person", type=int, default=None, choices=[0, 1], help="Render one person only.")
    parser.add_argument("--metrics-dir", type=str, default="data/test/skyhook")
    parser.add_argument(
        "--retarget-dir",
        type=str,
        default=None,
        help="Directory with <clip>_personX.npy positions. Defaults to data/retargeted_v2/<dataset>.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help="Raw dataset dir for exact FK fallback when retargeted joint_q is missing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Warp device for exact FK reconstruction.",
    )
    parser.add_argument(
        "--prefer-exact",
        action="store_true",
        help="Prefer exact metric trajectory (joint_q + FK).",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail if exact metric trajectory cannot be reconstructed.",
    )
    parser.add_argument(
        "--focus-person",
        type=int,
        default=None,
        choices=[0, 1],
        help="Person to use for high/low-force callouts. Default: auto (highest force spike).",
    )
    parser.add_argument(
        "--no-ghost-extremes",
        action="store_true",
        help="Disable static low/high ghost poses in the 3D view.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    metrics_dir = _resolve_path(args.metrics_dir)
    retarget_dir = _resolve_path(args.retarget_dir) if args.retarget_dir else _default_retarget_dir(args.dataset)
    persons = _detect_persons(args.clip, metrics_dir, args.person)
    if not persons:
        raise FileNotFoundError(
            f"No skyhook metrics found for clip {args.clip} in {metrics_dir}. "
            "Expected files like <clip>_person0_skyhook_metrics.npz"
        )

    output = (
        _resolve_path(args.output)
        if args.output
        else metrics_dir / f"{args.clip}_skyhook_newton_like.mp4"
    )
    default_raw = "data/InterHuman" if args.dataset == "interhuman" else "data/Inter-X_Dataset"
    raw_data_dir = _resolve_path(args.raw_data_dir) if args.raw_data_dir else _resolve_path(default_raw)
    if not raw_data_dir.exists():
        raw_data_dir = None
    data = _load_data(
        clip=args.clip,
        dataset=args.dataset,
        persons=persons,
        metrics_dir=metrics_dir,
        retarget_dir=retarget_dir,
        raw_data_dir=raw_data_dir,
        prefer_exact=args.prefer_exact or args.require_exact,
        require_exact=args.require_exact,
        device=args.device,
    )
    for p in persons:
        print(
            f"person{p}: joint_source={data[p]['joint_source']}, "
            f"position_source={data[p]['position_source']}"
        )
    render_video(
        clip=args.clip,
        dataset=args.dataset,
        data=data,
        output_mp4=output,
        fps=args.fps,
        dpi=args.dpi,
        focus_person=args.focus_person,
        ghost_extremes=not args.no_ghost_extremes,
    )
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
