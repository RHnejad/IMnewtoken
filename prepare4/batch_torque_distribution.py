"""
batch_torque_distribution.py — Compute torque distribution over sampled clips.

Randomly samples N clips from the InterHuman dataset (or from generated
outputs), runs inverse dynamics on both persons in each clip, and aggregates
per-DOF torque statistics + histograms.

Usage:
    # Sample 200 clips from GT dataset (default)
    python prepare4/batch_torque_distribution.py

    # Custom sample size
    python prepare4/batch_torque_distribution.py --n-clips 500

    # Run on generated (InterMask) outputs
    python prepare4/batch_torque_distribution.py --source generated \
        --data-dir data/generated/interhuman --n-clips 200

    # All generated clips
    python prepare4/batch_torque_distribution.py --source generated \
        --data-dir data/generated/interhuman --n-clips 0

    # Specific output directory
    python prepare4/batch_torque_distribution.py --output-dir data/torque_stats

    # Fixed random seed for reproducibility
    python prepare4/batch_torque_distribution.py --seed 42

    # Resume from saved intermediate results
    python prepare4/batch_torque_distribution.py --resume
"""
import os
import sys
import glob
import json
import time
import argparse
import pickle
import warnings
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", message="Custom attribute")

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# DOF names for readable output
DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]
BODY_NAMES_JOINTS = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]
for _body in BODY_NAMES_JOINTS:
    DOF_NAMES.extend([f"{_body}_x", f"{_body}_y", f"{_body}_z"])

# Group DOFs by body region for summary
BODY_GROUPS = {
    "Root trans":  slice(0, 3),
    "Root rot":    slice(3, 6),
    "L_Leg":       slice(6, 18),    # L_Hip(3) + L_Knee(3) + L_Ankle(3) + L_Toe(3)
    "R_Leg":       slice(18, 30),
    "Spine/Torso": slice(30, 45),   # Torso(3) + Spine(3) + Chest(3) + Neck(3) + Head(3)
    "L_Arm":       slice(45, 60),   # L_Thorax(3) + L_Shoulder(3) + L_Elbow(3) + L_Wrist(3) + L_Hand(3)
    "R_Arm":       slice(60, 75),
}

FPS = 30  # InterMask operates at 30fps (GT PKL is 60fps, downsampled 2x)
DT = 1.0 / FPS


def load_generated_pkl(clip_path):
    """Load persons from a generated InterMask PKL file.

    Generated PKLs have person1/person2 dicts directly at top level
    (same keys as GT: root_orient, pose_body, trans, betas).

    Returns list of dicts, or None on failure.
    """
    if not os.path.isfile(clip_path):
        return None
    with open(clip_path, "rb") as f:
        raw = pickle.load(f)
    results = []
    for pkey in ["person1", "person2"]:
        if pkey not in raw:
            continue
        p = raw[pkey]
        d = {
            "root_orient": p["root_orient"].astype(np.float64),
            "pose_body": p["pose_body"].astype(np.float64),
            "trans": p["trans"].astype(np.float64),
            "betas": p["betas"].astype(np.float64),
        }
        if "positions_zup" in p:
            d["positions_zup"] = p["positions_zup"].astype(np.float64)
        results.append(d)
    return results if results else None


def process_clip(clip_path, device="cuda:0", source="gt"):
    """Compute physically-grounded PD torques for one clip (both persons).

    Runs a Newton/MuJoCo forward sim with ground contacts and PD tracking
    to extract the applied feedback torques — no skyhook artefacts.

    Args:
        clip_path: path to the clip PKL
        device: CUDA device
        source: 'gt' (InterHuman dataset) or 'generated' (InterMask output)

    Returns list of (T, 75) torque arrays, or None on failure.
    """
    from prepare4.retarget import rotation_retarget, ik_retarget, load_interhuman_pkl
    from prepare4.run_full_analysis import pd_forward_torques

    clip_id = os.path.splitext(os.path.basename(clip_path))[0]

    if source == "generated":
        persons = load_generated_pkl(clip_path)
    else:
        data_dir = os.path.dirname(os.path.dirname(clip_path))
        persons = load_interhuman_pkl(data_dir, clip_id)

    if persons is None:
        return None

    results = []
    for p in persons:
        if source == "generated":
            if "positions_zup" not in p:
                print(f"    SKIP clip {clip_id}: no positions_zup")
                results.append(None)
                continue
            joint_q, _ = ik_retarget(
                p["positions_zup"], p["betas"],
                ik_iters=50, device=device, sequential=True,
            )
        else:
            joint_q = rotation_retarget(
                p["root_orient"], p["pose_body"], p["trans"], p["betas"]
            )
            joint_q = joint_q[::2]  # 60fps → 30fps

        T = joint_q.shape[0]
        if T < 11:
            results.append(None)
            continue

        try:
            torques, _ = pd_forward_torques(
                joint_q, p["betas"],
                dt=DT, device=device, verbose=False,
            )
            results.append(torques)
        except Exception as e:
            print(f"    ERROR on clip {clip_id}: {e}")
            results.append(None)

    return results


def batch_compute(clips, device="cuda:0", save_dir=None, resume=False, source="gt"):
    """Process a batch of clips, collecting all torque frames.

    Args:
        clips: list of clip file paths
        device: CUDA device
        save_dir: directory to save intermediate results
        resume: if True, skip clips that already have saved results
        source: 'gt' or 'generated'

    Returns:
        all_torques: (N_total_frames, 75) concatenated torques
        clip_stats: list of per-clip summary dicts
    """
    all_torques_list = []
    clip_stats = []
    n_errors = 0
    t_start = time.time()

    # Load existing results if resuming
    done_clips = set()
    if resume and save_dir:
        intermediate_path = os.path.join(save_dir, "intermediate_torques.npz")
        if os.path.exists(intermediate_path):
            data = np.load(intermediate_path, allow_pickle=True)
            all_torques_list.append(data["torques"])
            done_clips = set(data["processed_clips"].tolist())
            print(f"Resumed: loaded {len(done_clips)} previously processed clips")
            print(f"  Existing frames: {data['torques'].shape[0]}")

    for i, clip_path in enumerate(clips):
        clip_id = os.path.splitext(os.path.basename(clip_path))[0]

        if clip_id in done_clips:
            continue

        elapsed = time.time() - t_start
        rate = (i + 1) / max(elapsed, 1)
        remaining = (len(clips) - i - 1) / max(rate, 0.01)
        print(f"[{i+1}/{len(clips)}] clip={clip_id}  "
              f"elapsed={elapsed/60:.1f}m  ETA={remaining/60:.1f}m")

        results = process_clip(clip_path, device=device, source=source)
        if results is None:
            n_errors += 1
            continue

        for j, torques in enumerate(results):
            if torques is None:
                continue
            all_torques_list.append(torques)
            clip_stats.append({
                "clip": clip_id,
                "person": j + 1,
                "T": torques.shape[0],
                "hinge_mean": float(np.abs(torques[:, 6:]).mean()),
                "hinge_max": float(np.abs(torques[:, 6:]).max()),
            })
        done_clips.add(clip_id)

        # Save intermediate results every 50 clips
        if save_dir and (i + 1) % 50 == 0:
            _save_intermediate(all_torques_list, done_clips, save_dir)

    if not all_torques_list:
        print("No torques computed!")
        return np.zeros((0, 75)), []

    all_torques = np.concatenate(all_torques_list, axis=0)
    print(f"\nDone: {len(clips)} clips, {n_errors} errors, "
          f"{all_torques.shape[0]} total frames, "
          f"{time.time() - t_start:.0f}s elapsed")

    return all_torques, clip_stats


def _save_intermediate(torques_list, done_clips, save_dir):
    """Save intermediate results for resume capability."""
    os.makedirs(save_dir, exist_ok=True)
    torques = np.concatenate(torques_list, axis=0)
    np.savez_compressed(
        os.path.join(save_dir, "intermediate_torques.npz"),
        torques=torques,
        processed_clips=np.array(list(done_clips)),
    )


def compute_statistics(all_torques):
    """Compute per-DOF and per-group statistics.

    Args:
        all_torques: (N, 75) all torque frames

    Returns:
        stats: dict with keys per DOF and per group
    """
    N = all_torques.shape[0]
    percentiles = [1, 5, 25, 50, 75, 95, 99]

    stats = {
        "n_frames": N,
        "dof": {},
        "groups": {},
    }

    # Per-DOF
    for d in range(75):
        vals = all_torques[:, d]
        stats["dof"][DOF_NAMES[d]] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "abs_mean": float(np.abs(vals).mean()),
            "abs_max": float(np.abs(vals).max()),
            "percentiles": {
                str(p): float(np.percentile(vals, p)) for p in percentiles
            },
        }

    # Per-group
    for name, sl in BODY_GROUPS.items():
        vals = all_torques[:, sl]
        stats["groups"][name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "abs_mean": float(np.abs(vals).mean()),
            "abs_max": float(np.abs(vals).max()),
            "percentiles": {
                str(p): float(np.percentile(np.abs(vals), p))
                for p in percentiles
            },
        }

    # All hinge DOFs (excluding root virtual forces)
    hinge = all_torques[:, 6:]
    stats["hinge_all"] = {
        "mean": float(hinge.mean()),
        "std": float(hinge.std()),
        "abs_mean": float(np.abs(hinge).mean()),
        "abs_max": float(np.abs(hinge).max()),
        "percentiles": {
            str(p): float(np.percentile(np.abs(hinge), p))
            for p in percentiles
        },
    }

    return stats


def print_statistics(stats):
    """Pretty-print torque statistics."""
    print(f"\n{'='*80}")
    print(f" TORQUE DISTRIBUTION — {stats['n_frames']:,} frames")
    print(f"{'='*80}")

    # Body group summary
    print(f"\n{'Group':<15s} {'|τ| mean':>10s} {'|τ| std':>10s} "
          f"{'|τ| 95%':>10s} {'|τ| 99%':>10s} {'|τ| max':>10s}  Unit")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  ----")
    for name, sl in BODY_GROUPS.items():
        g = stats["groups"][name]
        unit = "N" if "trans" in name else "Nm"
        print(f"{name:<15s} {g['abs_mean']:10.2f} {g['std']:10.2f} "
              f"{g['percentiles']['95']:10.2f} "
              f"{g['percentiles']['99']:10.2f} "
              f"{g['abs_max']:10.2f}  {unit}")

    g = stats["hinge_all"]
    print(f"\n{'ALL HINGES':<15s} {g['abs_mean']:10.2f} {g['std']:10.2f} "
          f"{g['percentiles']['95']:10.2f} "
          f"{g['percentiles']['99']:10.2f} "
          f"{g['abs_max']:10.2f}  Nm")

    # Per-DOF table (hinges only, skip root virtual forces)
    print(f"\n{'─'*80}")
    print(f" Per-DOF detail (hinges only):")
    print(f"{'─'*80}")
    print(f"{'DOF':<18s} {'mean':>8s} {'std':>8s} {'|τ| p50':>8s} "
          f"{'|τ| p95':>8s} {'|τ| p99':>8s} {'max':>8s}")
    print(f"{'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for d in range(6, 75):
        name = DOF_NAMES[d]
        s = stats["dof"][name]
        print(f"{name:<18s} {s['mean']:8.2f} {s['std']:8.2f} "
              f"{float(s['percentiles']['50']):8.2f} "
              f"{float(s['percentiles']['95']):8.2f} "
              f"{float(s['percentiles']['99']):8.2f} "
              f"{s['abs_max']:8.2f}")


def plot_distributions(all_torques, save_dir):
    """Create distribution plots and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Overall hinge torque histogram ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    hinge = all_torques[:, 6:].ravel()
    ax.hist(hinge, bins=200, range=(-200, 200), density=True, alpha=0.7,
            color="steelblue", edgecolor="none")
    ax.set_xlabel("Torque (Nm)")
    ax.set_ylabel("Density")
    ax.set_title(f"All Hinge Torques (N={len(hinge):,})")
    ax.axvline(0, color="k", linewidth=0.5)

    # Add percentile lines
    for p, ls in [(95, "--"), (99, ":")]:
        v = np.percentile(np.abs(hinge), p)
        ax.axvline(v, color="red", linestyle=ls, label=f"|τ| p{p}={v:.1f}")
        ax.axvline(-v, color="red", linestyle=ls)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "torque_hist_all.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-body-group box plots ───────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    group_data = []
    group_labels = []
    for name, sl in BODY_GROUPS.items():
        if "trans" in name:
            continue  # skip root position (different units)
        group_data.append(all_torques[:, sl].ravel())
        group_labels.append(name)
    bp = ax.boxplot(group_data, tick_labels=group_labels, showfliers=False,
                    whis=[5, 95], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Torque Distribution by Body Group (5-95% whiskers)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "torque_boxplot_groups.png"), dpi=150)
    plt.close(fig)

    # ── 3. Per-body heatmap (|τ| percentiles) ────────────
    body_names = BODY_NAMES_JOINTS
    n_bodies = len(body_names)
    pcts = [50, 75, 90, 95, 99]
    heatmap = np.zeros((n_bodies, len(pcts)))
    for b in range(n_bodies):
        dof_start = 6 + b * 3
        body_torques = np.abs(all_torques[:, dof_start:dof_start + 3]).ravel()
        for j, p in enumerate(pcts):
            heatmap[b, j] = np.percentile(body_torques, p)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"p{p}" for p in pcts])
    ax.set_yticks(range(n_bodies))
    ax.set_yticklabels(body_names)
    ax.set_title("|τ| Percentiles per Body (Nm)")
    # Annotate cells
    for i in range(n_bodies):
        for j in range(len(pcts)):
            ax.text(j, i, f"{heatmap[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if heatmap[i, j] < heatmap.max() * 0.7 else "white")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Nm")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "torque_heatmap_bodies.png"), dpi=150)
    plt.close(fig)

    # ── 4. Torque time series for a random clip ──────────
    # (Use first 200 frames from all_torques)
    n_show = min(200, all_torques.shape[0])
    t = np.arange(n_show) / FPS
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Legs
    for d in range(6, 18):
        axes[0].plot(t, all_torques[:n_show, d], alpha=0.5, linewidth=0.8)
    axes[0].set_ylabel("Torque (Nm)")
    axes[0].set_title("Left Leg DOFs")
    axes[0].grid(alpha=0.3)

    # Spine
    for d in range(30, 45):
        axes[1].plot(t, all_torques[:n_show, d], alpha=0.5, linewidth=0.8)
    axes[1].set_ylabel("Torque (Nm)")
    axes[1].set_title("Spine/Torso/Neck/Head DOFs")
    axes[1].grid(alpha=0.3)

    # Arms
    for d in range(45, 60):
        axes[2].plot(t, all_torques[:n_show, d], alpha=0.5, linewidth=0.8)
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Left Arm DOFs")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Sample Torque Time Series", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "torque_timeseries_sample.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Compute torque distribution from sampled InterHuman clips"
    )
    parser.add_argument("--n-clips", type=int, default=200,
                        help="Number of clips to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for clip sampling (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/torque_stats)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device (default: cuda:0)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from intermediate results")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--source", type=str, default="gt",
                        choices=["gt", "generated"],
                        help="Data source: 'gt' (InterHuman) or 'generated' (InterMask)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: auto-detected based on --source)")
    args = parser.parse_args()

    # ── Determine output dir ──────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    elif args.source == "generated":
        output_dir = os.path.join(PROJECT_ROOT, "data", "torque_stats_generated")
    else:
        output_dir = os.path.join(PROJECT_ROOT, "data", "torque_stats")
    os.makedirs(output_dir, exist_ok=True)

    # ── Discover clips ────────────────────────────────────
    if args.source == "generated":
        data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data", "generated", "interhuman")
        all_clips = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        print(f"Found {len(all_clips)} generated clips in {data_dir}")
    else:
        data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data", "InterHuman", "motions")
        all_clips = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        print(f"Found {len(all_clips)} GT clips in {data_dir}")

    # ── Sample ────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    if args.n_clips <= 0 or args.n_clips >= len(all_clips):
        n = len(all_clips)
        sampled_clips = all_clips
        print(f"Using ALL {n} clips")
    else:
        n = min(args.n_clips, len(all_clips))
        sampled_idx = rng.choice(len(all_clips), size=n, replace=False)
        sampled_clips = [all_clips[i] for i in sorted(sampled_idx)]
        print(f"Sampled {n} clips (seed={args.seed})")

    # Save sample manifest
    manifest = {
        "n_clips": n,
        "seed": args.seed,
        "clip_ids": [os.path.splitext(os.path.basename(c))[0] for c in sampled_clips],
    }
    with open(os.path.join(output_dir, "sample_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Batch compute ─────────────────────────────────────
    all_torques, clip_stats = batch_compute(
        sampled_clips, device=args.device,
        save_dir=output_dir, resume=args.resume,
        source=args.source,
    )

    if all_torques.shape[0] == 0:
        print("No torques computed. Exiting.")
        return

    # ── Save final results ────────────────────────────────
    np.savez_compressed(
        os.path.join(output_dir, "torque_distribution.npz"),
        torques=all_torques,
        dof_names=np.array(DOF_NAMES),
    )
    print(f"\nSaved torques: {all_torques.shape} to {output_dir}/torque_distribution.npz")

    # ── Statistics ────────────────────────────────────────
    stats = compute_statistics(all_torques)
    print_statistics(stats)

    with open(os.path.join(output_dir, "torque_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ── Clip-level stats ─────────────────────────────────
    if clip_stats:
        with open(os.path.join(output_dir, "clip_stats.json"), "w") as f:
            json.dump(clip_stats, f, indent=2)

    # ── Plots ─────────────────────────────────────────────
    if not args.no_plot:
        plot_distributions(all_torques, output_dir)

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
