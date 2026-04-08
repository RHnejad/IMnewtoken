#!/usr/bin/env python3
"""Headless renderer for mesh-contact visualizations.

Renders per-frame contact-colored mesh images (PNG) without any display.
Uses matplotlib's Agg backend — works in containers / headless RunAI jobs.

Can render from:
  (a) A single clip on-the-fly (--dataset + --clip)
  (b) A batch of completed clips (--batch)

Usage:
    # Render a single clip (all frames), output PNGs to a directory
    python render_contact_headless.py --dataset interhuman --clip 7605 --out-dir output/renders/interhuman_7605

    # Render specific frames only
    python render_contact_headless.py --dataset interhuman --clip 7605 \\
        --frames 0 10 50 100 --out-dir output/renders/

    # Batch: render one representative frame of every completed clip
    python render_contact_headless.py --batch interhuman --frames-per-clip 1 \\
        --out-dir output/renders/interhuman_batch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Force headless backend before any other matplotlib import
_MPLCONFIGDIR = os.path.join("/tmp", "render_contact_headless_mpl")
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


STATUS_PRIORITY = {
    "penetrating": 0,
    "touching": 1,
    "barely_touching": 2,
    "not_touching": 3,
}

INTERHUMAN_BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]

INTERHUMAN_PERSON_COLORS = ("#2C7FB8", "#F16913")


def _safe_float(value: object, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 10**9) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _status_frame_sort_key(summary: Dict[str, object]) -> Tuple[int, float, int]:
    return (
        STATUS_PRIORITY.get(str(summary.get("status", "")), len(STATUS_PRIORITY)),
        _safe_float(summary.get("min_distance_m"), default=float("inf")),
        _safe_int(summary.get("frame"), default=10**9),
    )


def _select_representative_frame_from_run_info(run_info: Dict[str, object]) -> Optional[int]:
    frames = run_info.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    best = min(frames, key=_status_frame_sort_key)
    frame_idx = _safe_int(best.get("frame"), default=-1)
    return frame_idx if frame_idx >= 0 else None


def _frame_window(center: int, n_frames: int, count: int) -> List[int]:
    if n_frames <= 0 or count <= 0:
        return []
    if count >= n_frames:
        return list(range(n_frames))

    start = int(center) - (count // 2)
    end = start + count
    if start < 0:
        end -= start
        start = 0
    if end > n_frames:
        start = max(0, start - (end - n_frames))
        end = n_frames
    return list(range(start, end))


def _choose_frame_indices(
    n_frames: int,
    count: int,
    frame_policy: str,
    representative_frame: Optional[int] = None,
) -> List[int]:
    if n_frames <= 0:
        return []
    if count <= 0 or count >= n_frames:
        return list(range(n_frames))

    if frame_policy == "first":
        return list(range(count))
    if frame_policy == "middle":
        return _frame_window(n_frames // 2, n_frames, count)

    center = representative_frame if representative_frame is not None else (n_frames // 2)
    return _frame_window(center, n_frames, count)


def _load_interhuman_caption_lines(data_root: str, clip_id: str, max_lines: int = 3) -> List[str]:
    annot_path = os.path.join(data_root, "annots", f"{clip_id}.txt")
    if not os.path.isfile(annot_path) or max_lines <= 0:
        return []

    with open(annot_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:max_lines]


def _load_interhuman_processed_positions(
    data_root: str,
    clip_id: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    persons: List[np.ndarray] = []
    for person_idx in (1, 2):
        npy_path = os.path.join(data_root, "motions_processed", f"person{person_idx}", f"{clip_id}.npy")
        if not os.path.isfile(npy_path):
            return None
        arr = np.load(npy_path)
        if arr.ndim != 2 or arr.shape[1] < 66:
            return None
        persons.append(arr[:, :66].reshape(-1, 22, 3).astype(np.float32))
    return persons[0], persons[1]


def _map_raw_frame_to_processed(raw_frame: int, raw_len: int, proc_len: int) -> int:
    if proc_len <= 1 or raw_len <= 1:
        return 0
    scaled = round((proc_len - 1) * float(raw_frame) / float(raw_len - 1))
    return int(np.clip(scaled, 0, proc_len - 1))


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _set_3d_bounds(ax, points: np.ndarray, span_scale: float = 0.6) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return

    mid = points.mean(axis=0)
    span = float(max(points.max(axis=0) - points.min(axis=0))) * span_scale
    if span <= 1e-6:
        span = 0.5
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)


def _plot_processed_skeleton(
    ax,
    positions_by_person: Sequence[np.ndarray],
    title: str,
    elev: float,
    azim: float,
) -> None:
    all_points = np.concatenate([np.asarray(pos, dtype=np.float32) for pos in positions_by_person], axis=0)
    for person_idx, pos in enumerate(positions_by_person):
        pos = np.asarray(pos, dtype=np.float32)
        color = INTERHUMAN_PERSON_COLORS[person_idx % len(INTERHUMAN_PERSON_COLORS)]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=12, c=color, alpha=0.85)
        for start, end in INTERHUMAN_BONES:
            xyz = pos[[start, end]]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2.0, alpha=0.9)

        hips = pos[[1, 2]]
        shoulders = pos[[16, 17]]
        ax.plot(hips[:, 0], hips[:, 1], hips[:, 2], color=color, linestyle="--", linewidth=1.4, alpha=0.7)
        ax.plot(shoulders[:, 0], shoulders[:, 1], shoulders[:, 2], color=color, linestyle=":", linewidth=1.4, alpha=0.7)

        forward = _normalize_vec(np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), hips[1] - hips[0]))
        pelvis = pos[0]
        ax.quiver(
            pelvis[0], pelvis[1], pelvis[2],
            0.25 * forward[0], 0.25 * forward[1], 0.25 * forward[2],
            color=color, linewidth=2.0, arrow_length_ratio=0.25,
        )

    _set_3d_bounds(ax, all_points, span_scale=0.7)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)


def _select_representative_frame_from_sequence(
    analyzer,
    vertices_p1: np.ndarray,
    vertices_p2: np.ndarray,
) -> int:
    best_frame = 0
    best_key: Optional[Tuple[int, float, int]] = None
    t_len = min(vertices_p1.shape[0], vertices_p2.shape[0])
    for frame_idx in range(t_len):
        summary, _ = analyzer._analyze_frame(vertices_p1[frame_idx], vertices_p2[frame_idx])
        summary["frame"] = frame_idx
        key = _status_frame_sort_key(summary)
        if best_key is None or key < best_key:
            best_key = key
            best_frame = frame_idx
    return best_frame


def _vertex_colors(
    n_verts: int,
    contact: np.ndarray,
    barely: np.ndarray,
    inter_pen: np.ndarray,
    self_pen: np.ndarray,
) -> np.ndarray:
    """Per-vertex RGBA float colors. Same scheme as export_contact_frame_ply."""
    c = np.full((n_verts, 4), [0.7, 0.7, 0.7, 0.4], dtype=np.float32)
    if barely.size > 0:
        c[barely] = [1.0, 0.75, 0.3, 0.7]
    if contact.size > 0:
        c[contact] = [0.9, 0.23, 0.23, 0.85]
    if inter_pen.size > 0:
        c[inter_pen] = [0.67, 0.16, 0.9, 0.95]
    if self_pen.size > 0:
        c[self_pen] = [0.16, 0.86, 0.47, 0.95]
    return c


def _face_colors(verts_colors: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Average vertex colors per face."""
    return verts_colors[faces].mean(axis=1)


def render_frame(
    vertices_p1: np.ndarray,
    vertices_p2: np.ndarray,
    faces: np.ndarray,
    detail: Dict[str, np.ndarray],
    summary: Dict[str, object],
    out_path: str,
    title: str = "",
    caption_lines: Optional[Sequence[str]] = None,
    compare_positions: Optional[Sequence[np.ndarray]] = None,
    compare_title: str = "",
    elev: float = 20.0,
    azim: float = -60.0,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 150,
) -> str:
    """Render one frame to PNG. Returns the output path."""
    vc1 = _vertex_colors(
        vertices_p1.shape[0],
        detail.get("contact_vidx_p1", np.zeros(0, dtype=np.int32)),
        detail.get("barely_vidx_p1", np.zeros(0, dtype=np.int32)),
        detail.get(
            "inter_person_penetrating_vidx_p1",
            detail.get("penetrating_vidx_p1", np.zeros(0, dtype=np.int32)),
        ),
        detail.get("self_penetrating_vidx_p1", np.zeros(0, dtype=np.int32)),
    )
    vc2 = _vertex_colors(
        vertices_p2.shape[0],
        detail.get("contact_vidx_p2", np.zeros(0, dtype=np.int32)),
        detail.get("barely_vidx_p2", np.zeros(0, dtype=np.int32)),
        detail.get(
            "inter_person_penetrating_vidx_p2",
            detail.get("penetrating_vidx_p2", np.zeros(0, dtype=np.int32)),
        ),
        detail.get("self_penetrating_vidx_p2", np.zeros(0, dtype=np.int32)),
    )

    fc1 = _face_colors(vc1, faces)
    fc2 = _face_colors(vc2, faces)

    if compare_positions is not None and figsize[0] <= 14:
        figsize = (18, figsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if compare_positions is not None:
        gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax_cmp = fig.add_subplot(gs[0, 1], projection="3d")
        ax_info = fig.add_subplot(gs[0, 2])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        ax_cmp = None
        ax_info = fig.add_subplot(gs[0, 1])

    for verts, fc in ((vertices_p1, fc1), (vertices_p2, fc2)):
        tri_verts = verts[faces]
        mesh = Poly3DCollection(tri_verts, linewidths=0.05, edgecolors=(0.3, 0.3, 0.3, 0.1))
        mesh.set_facecolor(fc)
        ax.add_collection3d(mesh)

    all_v = np.concatenate([vertices_p1, vertices_p2], axis=0)
    _set_3d_bounds(ax, all_v, span_scale=0.6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "Contact Mesh", fontsize=10)

    if ax_cmp is not None and compare_positions is not None:
        _plot_processed_skeleton(
            ax_cmp,
            compare_positions,
            title=compare_title or "Processed 22-joint skeleton",
            elev=elev,
            azim=azim,
        )

    ax_info.axis("off")
    status = summary.get("status", "?")
    pen_src = summary.get("penetration_source", "?")
    min_d_cm = 100.0 * float(summary.get("min_distance_m", 0))
    pen_d_cm = 100.0 * float(
        summary.get(
            "inter_person_penetration_depth_est_m",
            summary.get("penetration_depth_est_m", 0),
        )
    )
    n_contact = int(summary.get("contact_vertex_count_p1", 0)) + int(summary.get("contact_vertex_count_p2", 0))
    n_pen = int(summary.get("inter_person_penetrating_vertex_count_p1", 0)) + int(
        summary.get("inter_person_penetrating_vertex_count_p2", 0)
    )

    color_map = {
        "penetrating": "purple",
        "touching": "red",
        "barely_touching": "orange",
        "not_touching": "green",
    }
    status_color = color_map.get(str(status), "black")

    info_entries: List[Dict[str, str]] = []
    if title:
        info_entries.append({"text": title, "fontweight": "bold", "color": "black"})
    if caption_lines:
        info_entries.append({"text": "Caption:", "fontweight": "bold", "color": "black"})
        for line in caption_lines:
            info_entries.append({"text": f"  {line}", "fontweight": "normal", "color": "black"})
        info_entries.append({"text": "", "fontweight": "normal", "color": "black"})

    info_entries.extend([
        {"text": f"Status: {status}", "fontweight": "bold", "color": status_color},
        {"text": f"Penetration source: {pen_src}", "fontweight": "normal", "color": "black"},
        {"text": f"Min distance: {min_d_cm:.2f} cm", "fontweight": "normal", "color": "black"},
        {"text": f"Penetration depth: {pen_d_cm:.2f} cm", "fontweight": "normal", "color": "black"},
        {"text": f"Contact vertices: {n_contact}", "fontweight": "normal", "color": "black"},
        {"text": f"Penetrating vertices: {n_pen}", "fontweight": "normal", "color": "black"},
        {"text": "", "fontweight": "normal", "color": "black"},
        {"text": "Legend:", "fontweight": "bold", "color": "black"},
        {"text": "  Grey     = no contact", "fontweight": "normal", "color": "black"},
        {"text": "  Orange   = barely touching", "fontweight": "normal", "color": "black"},
        {"text": "  Red      = touching", "fontweight": "normal", "color": "black"},
        {"text": "  Purple   = inter-person penetration", "fontweight": "normal", "color": "black"},
        {"text": "  Green    = self-penetration", "fontweight": "normal", "color": "black"},
    ])

    y = 0.97
    line_step = 0.05 if compare_positions is not None else 0.052
    for entry in info_entries:
        line = entry["text"]
        if line == "":
            y -= line_step * 0.5
            continue
        ax_info.text(
            0.05,
            y,
            line,
            transform=ax_info.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            fontweight=entry["fontweight"],
            color=entry["color"],
        )
        y -= line_step

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def render_clip(
    dataset: str,
    clip_id: str,
    data_root: str,
    body_model_path: str,
    out_dir: str,
    frame_indices: Optional[List[int]] = None,
    max_frames: int = 0,
    device: str = "cuda",
    batch_size: int = 64,
    self_pen_mode: str = "off",
    h5_file: Optional[str] = None,
    frame_policy: str = "representative",
    show_caption: bool = False,
    caption_line_count: int = 3,
    compare_processed_skeleton: bool = False,
    elev: float = 20.0,
    azim: float = -60.0,
    dpi: int = 150,
) -> List[str]:
    """Render a clip end-to-end: load data, compute contacts, render PNGs."""
    from prepare_mesh_contact.mesh_contact_pipeline import (
        ContactConfig,
        MeshContactAnalyzer,
        load_interhuman_clip,
        load_interx_clip,
        reconstruct_smplx_mesh_sequence,
    )

    if dataset == "interhuman":
        persons = load_interhuman_clip(data_root, clip_id)
    else:
        persons = load_interx_clip(data_root, clip_id, h5_file=h5_file)

    v1, v2, faces = reconstruct_smplx_mesh_sequence(
        persons, body_model_path, device=device, batch_size=batch_size
    )

    t_len = min(v1.shape[0], v2.shape[0])
    cfg = ContactConfig(self_penetration_mode=self_pen_mode)
    analyzer = MeshContactAnalyzer(faces, cfg)

    if frame_indices:
        frames = sorted(set(f for f in frame_indices if 0 <= f < t_len))
    elif max_frames > 0 and max_frames < t_len:
        representative_frame = None
        if frame_policy == "representative":
            print(f"  selecting representative frame for clip={clip_id} from {t_len} frames")
            representative_frame = _select_representative_frame_from_sequence(analyzer, v1, v2)
        frames = _choose_frame_indices(t_len, max_frames, frame_policy, representative_frame)
    else:
        frames = list(range(t_len))

    caption: List[str] = []
    if dataset == "interhuman" and show_caption:
        caption = _load_interhuman_caption_lines(data_root, clip_id, max_lines=caption_line_count)

    processed_sequences: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if compare_processed_skeleton:
        if dataset != "interhuman":
            print(f"  compare_processed_skeleton is only supported for InterHuman; ignoring clip={clip_id}")
        else:
            processed_sequences = _load_interhuman_processed_positions(data_root, clip_id)
            if processed_sequences is None:
                print(f"  processed 22-joint skeleton not found for clip={clip_id}; continuing without debug overlay")

    os.makedirs(out_dir, exist_ok=True)
    rendered = []
    for i, fi in enumerate(frames):
        summary, detail = analyzer._analyze_frame(v1[fi], v2[fi])
        summary["frame"] = fi
        path = os.path.join(out_dir, f"frame_{fi:05d}.png")

        compare_positions = None
        compare_title = ""
        if processed_sequences is not None:
            proc_len = min(processed_sequences[0].shape[0], processed_sequences[1].shape[0])
            proc_frame = _map_raw_frame_to_processed(fi, t_len, proc_len)
            compare_positions = [
                processed_sequences[0][proc_frame],
                processed_sequences[1][proc_frame],
            ]
            compare_title = f"Processed 22-joint skeleton\n(raw={fi}, processed={proc_frame})"

        render_frame(
            v1[fi],
            v2[fi],
            faces,
            detail,
            summary,
            path,
            title=f"{dataset} clip={clip_id} frame={fi}",
            caption_lines=caption,
            compare_positions=compare_positions,
            compare_title=compare_title,
            elev=elev,
            azim=azim,
            dpi=dpi,
        )
        rendered.append(path)
        if (i + 1) % 20 == 0 or (i + 1) == len(frames):
            print(f"  rendered {i + 1}/{len(frames)} frames")

    return rendered


def render_batch(
    dataset: str,
    out_dir: str,
    frames_per_clip: int = 1,
    max_clips: int = 0,
    frame_policy: str = "representative",
    show_caption: bool = False,
    caption_line_count: int = 3,
    compare_processed_skeleton: bool = False,
    elev: float = 20.0,
    azim: float = -60.0,
    device: str = "cuda",
    dpi: int = 150,
) -> None:
    """Render sample frames from all completed clips in a batch."""
    if dataset == "interhuman":
        json_dir = os.path.join(PROJECT_ROOT, "output", "mesh_contact", "interhuman")
        data_root = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    else:
        json_dir = os.path.join(PROJECT_ROOT, "output", "mesh_contact", "interx")
        data_root = os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset")
        if compare_processed_skeleton:
            print("compare_processed_skeleton is only supported for InterHuman; disabling it for this batch")
            compare_processed_skeleton = False

    body_model = os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz")

    if not os.path.isdir(json_dir):
        print(f"No completed outputs at {json_dir}")
        return

    jsons = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    if max_clips > 0:
        indices = np.linspace(0, len(jsons) - 1, min(max_clips, len(jsons)), dtype=int)
        jsons = [jsons[i] for i in indices]

    print(f"Rendering {len(jsons)} clips from {dataset}, {frames_per_clip} frame(s) each")
    os.makedirs(out_dir, exist_ok=True)

    for ci, jf in enumerate(jsons):
        clip_id = jf.replace(".json", "")
        with open(os.path.join(json_dir, jf), "r") as f:
            run_info = json.load(f)

        n_frames = _safe_int(run_info.get("frame_range", {}).get("num_frames"), default=0)
        representative_frame = None
        if frame_policy == "representative":
            representative_frame = _select_representative_frame_from_run_info(run_info)
        frame_list = _choose_frame_indices(
            n_frames,
            frames_per_clip,
            frame_policy,
            representative_frame=representative_frame,
        )

        try:
            rendered = render_clip(
                dataset,
                clip_id,
                data_root,
                body_model,
                out_dir,
                frame_indices=frame_list,
                device=device,
                frame_policy=frame_policy,
                show_caption=show_caption,
                caption_line_count=caption_line_count,
                compare_processed_skeleton=compare_processed_skeleton,
                elev=elev,
                azim=azim,
                dpi=dpi,
            )
            for old_path in rendered:
                new_name = f"{dataset}_{clip_id}_{os.path.basename(old_path)}"
                new_path = os.path.join(out_dir, new_name)
                os.rename(old_path, new_path)
        except Exception as e:
            print(f"  ERROR clip={clip_id}: {e}")

        if (ci + 1) % 10 == 0 or (ci + 1) == len(jsons):
            print(f"  [{ci + 1}/{len(jsons)}] clips rendered")


def main():
    parser = argparse.ArgumentParser(description="Headless mesh-contact visualization renderer")

    parser.add_argument("--dataset", choices=["interhuman", "interx"], default="interhuman")
    parser.add_argument("--clip", type=str, default=None, help="Single clip ID to render")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--h5-file", type=str, default=None)
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "body_model", "smplx", "SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-pen-mode", choices=["off", "heuristic"], default="off")

    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for PNGs")
    parser.add_argument("--frames", type=int, nargs="+", default=None, help="Specific frame indices to render")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per clip (0 = all)")
    parser.add_argument(
        "--frame-policy",
        choices=["first", "middle", "representative"],
        default="representative",
        help="How to choose diagnostic frames when rendering a subset",
    )
    caption_group = parser.add_mutually_exclusive_group()
    caption_group.add_argument(
        "--show-caption",
        action="store_true",
        default=None,
        dest="show_caption",
        help="Show InterHuman motion captions in the info panel",
    )
    caption_group.add_argument(
        "--no-show-caption",
        action="store_false",
        dest="show_caption",
        help="Disable captions",
    )
    parser.add_argument("--caption-lines", type=int, default=3, help="Maximum number of caption lines to show")
    parser.add_argument(
        "--compare-processed-skeleton",
        action="store_true",
        help="InterHuman only: add a processed 22-joint skeleton debug subplot",
    )
    parser.add_argument("--elev", type=float, default=20.0, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth")
    parser.add_argument("--dpi", type=int, default=150)

    parser.add_argument("--batch", type=str, default=None, choices=["interhuman", "interx"], help="Batch-render completed clips")
    parser.add_argument("--frames-per-clip", type=int, default=1, help="Number of frames per clip in batch mode")
    parser.add_argument("--max-clips", type=int, default=0, help="Max clips in batch mode (0 = all)")

    args = parser.parse_args()

    batch_dataset = args.batch if args.batch else args.dataset
    show_caption = args.show_caption
    if show_caption is None:
        show_caption = batch_dataset == "interhuman"

    if args.batch:
        render_batch(
            args.batch,
            args.out_dir,
            frames_per_clip=args.frames_per_clip,
            max_clips=args.max_clips,
            frame_policy=args.frame_policy,
            show_caption=show_caption,
            caption_line_count=args.caption_lines,
            compare_processed_skeleton=args.compare_processed_skeleton,
            elev=args.elev,
            azim=args.azim,
            device=args.device,
            dpi=args.dpi,
        )
    elif args.clip:
        data_root = args.data_root
        if data_root is None:
            data_root = (
                os.path.join(PROJECT_ROOT, "data", "InterHuman")
                if args.dataset == "interhuman"
                else os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset")
            )
        rendered = render_clip(
            args.dataset,
            args.clip,
            data_root,
            args.body_model_path,
            args.out_dir,
            frame_indices=args.frames,
            max_frames=args.max_frames,
            device=args.device,
            batch_size=args.batch_size,
            self_pen_mode=args.self_pen_mode,
            h5_file=args.h5_file,
            frame_policy=args.frame_policy,
            show_caption=show_caption,
            caption_line_count=args.caption_lines,
            compare_processed_skeleton=args.compare_processed_skeleton,
            elev=args.elev,
            azim=args.azim,
            dpi=args.dpi,
        )
        print(f"Rendered {len(rendered)} frames to {args.out_dir}")
    else:
        parser.error("Provide either --clip or --batch")


if __name__ == "__main__":
    main()
