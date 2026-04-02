#!/usr/bin/env python3
"""Visualize ImDy per-person torque and GRF predictions for individual clips.

Generates per-clip PNG/PDF with:
  - Per-joint torque magnitude over time (heatmap)
  - Total torque magnitude over time (line)
  - Vertical GRF over time (line)
  - Contact prediction over time (binary heatmap)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_pipeline.imdy_preprocessor import preprocess_for_imdy
from eval_pipeline.imdy_model_wrapper import ImDyWrapper

# SMPL joint names for labeling (23 joints, pelvis is root so torque has 23 DoF groups)
JOINT_NAMES_23 = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "Jaw",
]


def visualize_person(
    positions: np.ndarray,
    model: ImDyWrapper,
    person_label: str,
    fps: float = 30.0,
    batch_size: int = 256,
):
    """Run ImDy and return (fig, metrics_dict) for one person."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    mkr, mvel, idx = preprocess_for_imdy(positions, dt=1.0 / fps)
    pred = model.predict_clip(mkr, mvel, batch_size=batch_size)

    torque = pred["torque"]       # (N, 23, 3)
    grf = pred["grf"]             # (N, 2, 24, 3)
    contact = pred["contact"]     # (N, 2, 24, 1)

    tor_mag = np.linalg.norm(torque, axis=-1)  # (N, 23)
    total_tor = tor_mag.mean(axis=1)           # (N,)
    vert_grf = grf[:, 0, :, 2]                # (N, 24) — Z component, current frame
    total_vert_grf = vert_grf.sum(axis=1)     # (N,)
    contact_prob = 1.0 / (1.0 + np.exp(-contact[:, 0, :, 0]))  # sigmoid, (N, 24)

    time_s = idx / fps

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"{person_label} — ImDy Predictions", fontsize=14, fontweight="bold")
    gs = GridSpec(4, 1, height_ratios=[3, 1.5, 1.5, 2], hspace=0.35)

    # 1) Torque heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(
        tor_mag.T, aspect="auto", cmap="hot", interpolation="nearest",
        extent=[time_s[0], time_s[-1], 22.5, -0.5],
    )
    ax1.set_yticks(range(23))
    ax1.set_yticklabels(JOINT_NAMES_23, fontsize=7)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("Per-Joint Torque Magnitude (Nm)")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Nm")

    # 2) Total torque line
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_s, total_tor, color="firebrick", linewidth=0.8)
    ax2.fill_between(time_s, total_tor, alpha=0.2, color="firebrick")
    ax2.set_ylabel("Mean |τ| (Nm)")
    ax2.set_title("Mean Torque Magnitude Over Time")
    ax2.set_xlim(time_s[0], time_s[-1])
    ax2.grid(True, alpha=0.3)

    # 3) Vertical GRF
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_s, total_vert_grf, color="steelblue", linewidth=0.8)
    ax3.fill_between(time_s, total_vert_grf, alpha=0.2, color="steelblue")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax3.set_ylabel("Vert GRF (N)")
    ax3.set_title("Total Vertical Ground Reaction Force")
    ax3.set_xlim(time_s[0], time_s[-1])
    ax3.grid(True, alpha=0.3)

    # 4) Contact heatmap (feet only: ankles 7,8 and feet 10,11)
    foot_indices = [7, 8, 10, 11]
    foot_labels = ["L_Ankle", "R_Ankle", "L_Foot", "R_Foot"]
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(
        contact_prob[:, foot_indices].T, aspect="auto", cmap="Blues",
        interpolation="nearest", vmin=0, vmax=1,
        extent=[time_s[0], time_s[-1], 3.5, -0.5],
    )
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(foot_labels, fontsize=9)
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Foot Contact Probability")
    plt.colorbar(im4, ax=ax4, shrink=0.8, label="P(contact)")

    metrics = {
        "mean_torque_nm": float(np.mean(tor_mag)),
        "peak_torque_nm": float(np.percentile(tor_mag, 95)),
        "mean_vert_grf_n": float(np.mean(total_vert_grf)),
        "negative_grf_frac": float(np.mean(vert_grf < 0)),
        "mean_contact_prob": float(np.mean(contact_prob[:, foot_indices])),
        "num_windows": int(len(idx)),
    }
    return fig, metrics


def visualize_clip(
    clip_id: str,
    data_dir: str,
    model: ImDyWrapper,
    output_dir: str,
    fps: float = 30.0,
):
    """Visualize both persons in a clip, save side-by-side."""
    import matplotlib.pyplot as plt

    p0 = np.load(os.path.join(data_dir, f"{clip_id}_person0.npy")).astype(np.float32)
    p1 = np.load(os.path.join(data_dir, f"{clip_id}_person1.npy")).astype(np.float32)

    fig_a, metrics_a = visualize_person(p0, model, f"Clip {clip_id} — Person A", fps=fps)
    fig_b, metrics_b = visualize_person(p1, model, f"Clip {clip_id} — Person B", fps=fps)

    os.makedirs(output_dir, exist_ok=True)

    path_a = os.path.join(output_dir, f"{clip_id}_personA.png")
    path_b = os.path.join(output_dir, f"{clip_id}_personB.png")
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    plt.close(fig_b)

    print(f"  [{clip_id}] Person A: mean_torque={metrics_a['mean_torque_nm']:.1f} Nm, "
          f"vert_grf={metrics_a['mean_vert_grf_n']:.1f} N, neg_grf={metrics_a['negative_grf_frac']:.2%}")
    print(f"  [{clip_id}] Person B: mean_torque={metrics_b['mean_torque_nm']:.1f} Nm, "
          f"vert_grf={metrics_b['mean_vert_grf_n']:.1f} N, neg_grf={metrics_b['negative_grf_frac']:.2%}")
    print(f"  Saved: {path_a}")
    print(f"  Saved: {path_b}")

    return metrics_a, metrics_b


def main():
    parser = argparse.ArgumentParser(description="Visualize ImDy per-person predictions")
    parser.add_argument("--data-dir", required=True, help="Directory with *_person0/1.npy files")
    parser.add_argument("--output-dir", default="output/imdy_viz", help="Where to save plots")
    parser.add_argument("--clip-ids", default="", help="Comma-separated clip IDs (default: first 5)")
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--imdy-config", default="prepare5/ImDy/config/IDFD_mkr.yml")
    parser.add_argument("--imdy-checkpoint", default="prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    import glob
    import re

    # Discover clips
    if args.clip_ids:
        clip_ids = [c.strip() for c in args.clip_ids.split(",") if c.strip()]
    else:
        pat = re.compile(r"^(.+)_person0\.npy$")
        clip_ids = []
        for f in sorted(glob.glob(os.path.join(args.data_dir, "*_person0.npy"))):
            name = os.path.basename(f)
            if "_joint_q" in name or "_betas" in name or "_torques" in name:
                continue
            m = pat.match(name)
            if m:
                clip_ids.append(m.group(1))
        clip_ids = clip_ids[:args.max_clips]

    if not clip_ids:
        print("No clips found!")
        return

    print(f"Loading ImDy model...")
    model = ImDyWrapper(
        config_path=args.imdy_config,
        checkpoint_path=args.imdy_checkpoint,
        device=args.device,
    )
    print(f"Model loaded on {model.device}")

    print(f"\nVisualizing {len(clip_ids)} clips: {clip_ids}")
    for cid in clip_ids:
        visualize_clip(cid, args.data_dir, model, args.output_dir, fps=args.fps)

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
