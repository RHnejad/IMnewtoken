"""
compare_torque_distributions.py — Compare GT vs Generated torque distributions.

Loads precomputed torque_distribution.npz files from both GT and generated
motion, computes side-by-side statistics, and generates comparison plots.

Usage:
    python prepare4/compare_torque_distributions.py

    # Custom paths
    python prepare4/compare_torque_distributions.py \
        --gt-dir data/torque_stats \
        --gen-dir data/torque_stats_generated \
        --output-dir data/torque_comparison
"""
import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare4.batch_torque_distribution import (
    DOF_NAMES, BODY_GROUPS, BODY_NAMES_JOINTS, FPS,
)

# ═══════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════

def load_torques(stats_dir):
    """Load torque distribution from a directory.

    Returns:
        torques: (N, 75) array
        stats: dict from torque_stats.json (or None)
    """
    npz_path = os.path.join(stats_dir, "torque_distribution.npz")
    data = np.load(npz_path, allow_pickle=True)
    torques = data["torques"]

    json_path = os.path.join(stats_dir, "torque_stats.json")
    stats = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            stats = json.load(f)

    return torques, stats


# ═══════════════════════════════════════════════════════════════
# Comparison Statistics
# ═══════════════════════════════════════════════════════════════

def compute_comparison(gt_torques, gen_torques):
    """Compute side-by-side summary statistics.

    Returns dict with per-group and per-dof comparisons.
    """
    percentiles = [50, 75, 90, 95, 99]
    comp = {
        "gt_frames": gt_torques.shape[0],
        "gen_frames": gen_torques.shape[0],
        "groups": {},
        "dof": {},
    }

    # Per-group comparison
    for name, sl in BODY_GROUPS.items():
        gt_vals = np.abs(gt_torques[:, sl]).ravel()
        gen_vals = np.abs(gen_torques[:, sl]).ravel()
        comp["groups"][name] = {
            "gt_mean": float(gt_vals.mean()),
            "gen_mean": float(gen_vals.mean()),
            "ratio_mean": float(gen_vals.mean() / max(gt_vals.mean(), 1e-8)),
            "gt_p95": float(np.percentile(gt_vals, 95)),
            "gen_p95": float(np.percentile(gen_vals, 95)),
            "ratio_p95": float(np.percentile(gen_vals, 95) / max(np.percentile(gt_vals, 95), 1e-8)),
            "gt_p99": float(np.percentile(gt_vals, 99)),
            "gen_p99": float(np.percentile(gen_vals, 99)),
            "ratio_p99": float(np.percentile(gen_vals, 99) / max(np.percentile(gt_vals, 99), 1e-8)),
            "gt_max": float(gt_vals.max()),
            "gen_max": float(gen_vals.max()),
        }

    # All hinges
    gt_hinge = np.abs(gt_torques[:, 6:]).ravel()
    gen_hinge = np.abs(gen_torques[:, 6:]).ravel()
    comp["hinge_all"] = {
        "gt_mean": float(gt_hinge.mean()),
        "gen_mean": float(gen_hinge.mean()),
        "ratio_mean": float(gen_hinge.mean() / max(gt_hinge.mean(), 1e-8)),
        "gt_p95": float(np.percentile(gt_hinge, 95)),
        "gen_p95": float(np.percentile(gen_hinge, 95)),
        "ratio_p95": float(np.percentile(gen_hinge, 95) / max(np.percentile(gt_hinge, 95), 1e-8)),
        "gt_p99": float(np.percentile(gt_hinge, 99)),
        "gen_p99": float(np.percentile(gen_hinge, 99)),
    }

    # Per-DOF (hinges only)
    for d in range(6, 75):
        gt_v = np.abs(gt_torques[:, d])
        gen_v = np.abs(gen_torques[:, d])
        comp["dof"][DOF_NAMES[d]] = {
            "gt_mean": float(gt_v.mean()),
            "gen_mean": float(gen_v.mean()),
            "ratio_mean": float(gen_v.mean() / max(gt_v.mean(), 1e-8)),
            "gt_p95": float(np.percentile(gt_v, 95)),
            "gen_p95": float(np.percentile(gen_v, 95)),
            "ratio_p95": float(np.percentile(gen_v, 95) / max(np.percentile(gt_v, 95), 1e-8)),
        }

    return comp


def print_comparison(comp):
    """Print side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f" TORQUE COMPARISON: GT ({comp['gt_frames']:,} frames) vs "
          f"Generated ({comp['gen_frames']:,} frames)")
    print(f"{'='*90}")

    # Group summary
    print(f"\n{'Group':<15s} │ {'GT |τ| mean':>10s} {'Gen |τ| mean':>12s} {'Ratio':>7s}"
          f" │ {'GT P95':>8s} {'Gen P95':>8s} {'Ratio':>7s}"
          f" │ {'GT P99':>8s} {'Gen P99':>8s} {'Ratio':>7s}")
    print(f"{'─'*15}─┼─{'─'*10}─{'─'*12}─{'─'*7}"
          f"─┼─{'─'*8}─{'─'*8}─{'─'*7}"
          f"─┼─{'─'*8}─{'─'*8}─{'─'*7}")

    for name in BODY_GROUPS:
        g = comp["groups"][name]
        unit = "N" if "trans" in name else "Nm"
        print(f"{name:<15s} │ {g['gt_mean']:10.1f} {g['gen_mean']:12.1f} {g['ratio_mean']:7.2f}x"
              f" │ {g['gt_p95']:8.1f} {g['gen_p95']:8.1f} {g['ratio_p95']:7.2f}x"
              f" │ {g['gt_p99']:8.1f} {g['gen_p99']:8.1f} {g['ratio_p99']:7.2f}x")

    h = comp["hinge_all"]
    print(f"\n{'ALL HINGES':<15s} │ {h['gt_mean']:10.1f} {h['gen_mean']:12.1f} {h['ratio_mean']:7.2f}x"
          f" │ {h['gt_p95']:8.1f} {h['gen_p95']:8.1f} {h['ratio_p95']:7.2f}x"
          f" │ {h['gt_p99']:8.1f} {h['gen_p99']:8.1f}")

    # Per-DOF detail (selected DOFs)
    print(f"\n{'─'*90}")
    print(f" Per-DOF detail (selected joints):")
    print(f"{'─'*90}")
    print(f"{'DOF':<18s} │ {'GT mean':>8s} {'Gen mean':>9s} {'Ratio':>7s}"
          f" │ {'GT P95':>8s} {'Gen P95':>9s} {'Ratio':>7s}")
    print(f"{'─'*18}─┼─{'─'*8}─{'─'*9}─{'─'*7}─┼─{'─'*8}─{'─'*9}─{'─'*7}")
    for d in range(6, 75):
        name = DOF_NAMES[d]
        dd = comp["dof"][name]
        print(f"{name:<18s} │ {dd['gt_mean']:8.2f} {dd['gen_mean']:9.2f} {dd['ratio_mean']:7.2f}x"
              f" │ {dd['gt_p95']:8.2f} {dd['gen_p95']:9.2f} {dd['ratio_p95']:7.2f}x")


# ═══════════════════════════════════════════════════════════════
# Comparison Plots
# ═══════════════════════════════════════════════════════════════

def plot_comparison(gt_torques, gen_torques, save_dir):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Overlaid histograms (all hinges) ───────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    gt_hinge = gt_torques[:, 6:].ravel()
    gen_hinge = gen_torques[:, 6:].ravel()
    bins = np.linspace(-200, 200, 201)
    ax.hist(gt_hinge, bins=bins, density=True, alpha=0.5, color="steelblue",
            label=f"GT (N={gt_torques.shape[0]:,})")
    ax.hist(gen_hinge, bins=bins, density=True, alpha=0.5, color="orangered",
            label=f"Generated (N={gen_torques.shape[0]:,})")
    ax.set_xlabel("Torque (Nm)")
    ax.set_ylabel("Density")
    ax.set_title("All Hinge Torques: GT vs Generated")
    ax.legend()
    ax.axvline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "compare_hist_all.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-group bar chart comparison ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    group_names = [g for g in BODY_GROUPS if "trans" not in g]
    x = np.arange(len(group_names))
    width = 0.35

    for ax, (metric, title) in zip(axes, [
        ("mean", "|τ| Mean (Nm)"),
        ("p95", "|τ| P95 (Nm)"),
        ("p99", "|τ| P99 (Nm)"),
    ]):
        gt_vals = []
        gen_vals = []
        for name in group_names:
            gt_v = np.abs(gt_torques[:, BODY_GROUPS[name]]).ravel()
            gen_v = np.abs(gen_torques[:, BODY_GROUPS[name]]).ravel()
            if metric == "mean":
                gt_vals.append(gt_v.mean())
                gen_vals.append(gen_v.mean())
            elif metric == "p95":
                gt_vals.append(np.percentile(gt_v, 95))
                gen_vals.append(np.percentile(gen_v, 95))
            else:
                gt_vals.append(np.percentile(gt_v, 99))
                gen_vals.append(np.percentile(gen_v, 99))

        ax.bar(x - width/2, gt_vals, width, label="GT", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, gen_vals, width, label="Generated", color="orangered", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Torque Comparison by Body Group", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "compare_bar_groups.png"), dpi=150)
    plt.close(fig)

    # ── 3. Per-body ratio heatmap ─────────────────────────
    body_names = BODY_NAMES_JOINTS
    n_bodies = len(body_names)
    pcts = [50, 75, 90, 95, 99]
    ratio_map = np.zeros((n_bodies, len(pcts)))
    for b in range(n_bodies):
        dof_start = 6 + b * 3
        gt_body = np.abs(gt_torques[:, dof_start:dof_start + 3]).ravel()
        gen_body = np.abs(gen_torques[:, dof_start:dof_start + 3]).ravel()
        for j, p in enumerate(pcts):
            gt_p = np.percentile(gt_body, p)
            gen_p = np.percentile(gen_body, p)
            ratio_map[b, j] = gen_p / max(gt_p, 1e-8)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(ratio_map, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.5, vmax=2.0)
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"p{p}" for p in pcts])
    ax.set_yticks(range(n_bodies))
    ax.set_yticklabels(body_names)
    ax.set_title("Gen/GT Torque Ratio per Body (green=similar, red=higher)")
    # Annotate cells
    for i in range(n_bodies):
        for j in range(len(pcts)):
            v = ratio_map[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.7 < v < 1.5 else "white")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Gen/GT ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "compare_ratio_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── 4. CDF comparison for key DOFs ───────────────────
    key_dofs = [
        ("L_Hip_x", 6), ("L_Knee_x", 9), ("Torso_x", 30),
        ("L_Shoulder_x", 42), ("R_Hip_x", 18), ("Spine_x", 33),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for (name, idx), ax in zip(key_dofs, axes.ravel()):
        gt_v = np.sort(np.abs(gt_torques[:, idx]))
        gen_v = np.sort(np.abs(gen_torques[:, idx]))
        gt_cdf = np.linspace(0, 1, len(gt_v))
        gen_cdf = np.linspace(0, 1, len(gen_v))
        ax.plot(gt_v, gt_cdf, color="steelblue", label="GT", linewidth=1.5)
        ax.plot(gen_v, gen_cdf, color="orangered", label="Gen", linewidth=1.5)
        ax.set_xlabel(f"|τ| {name} (Nm)")
        ax.set_ylabel("CDF")
        ax.set_title(name)
        ax.set_xlim(0, np.percentile(np.concatenate([gt_v, gen_v]), 99) * 1.2)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("CDF Comparison: GT vs Generated (key DOFs)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "compare_cdf_key_dofs.png"), dpi=150)
    plt.close(fig)

    print(f"\nComparison plots saved to {save_dir}/")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare GT vs Generated torque distributions"
    )
    parser.add_argument("--gt-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "torque_stats"),
                        help="GT torque stats directory")
    parser.add_argument("--gen-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "torque_stats_generated"),
                        help="Generated torque stats directory")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "torque_comparison"),
                        help="Output directory for comparison results")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print(f"Loading GT torques from {args.gt_dir}...")
    gt_torques, gt_stats = load_torques(args.gt_dir)
    print(f"  GT: {gt_torques.shape[0]:,} frames × {gt_torques.shape[1]} DOFs")

    print(f"Loading Generated torques from {args.gen_dir}...")
    gen_torques, gen_stats = load_torques(args.gen_dir)
    print(f"  Gen: {gen_torques.shape[0]:,} frames × {gen_torques.shape[1]} DOFs")

    # Compare
    comp = compute_comparison(gt_torques, gen_torques)
    print_comparison(comp)

    # Save comparison JSON
    comp_path = os.path.join(args.output_dir, "torque_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\nComparison saved to {comp_path}")

    # Save log
    log_path = os.path.join(args.output_dir, "comparison.log")
    with open(log_path, "w") as f:
        # Redirect print to file
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_comparison(comp)
        sys.stdout = old_stdout
        f.write(buf.getvalue())
        f.write(f"\n\nGT source: {args.gt_dir}\n")
        f.write(f"Gen source: {args.gen_dir}\n")
    print(f"Log saved to {log_path}")

    # Plots
    if not args.no_plot:
        plot_comparison(gt_torques, gen_torques, args.output_dir)

    print(f"\nAll comparison results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
