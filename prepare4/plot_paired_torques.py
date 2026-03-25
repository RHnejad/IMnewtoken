"""
plot_paired_torques.py — Visualization for paired-vs-solo torque analysis.

Generates:
  1. Per-clip torque time-series (paired vs solo vs delta)
  2. Root force comparison (paired vs solo)
  3. Torque delta heatmap (body-group × time)
  4. GT vs Generated aggregate bar chart
  5. SII distribution histogram
  6. Newton's 3rd law compliance plot
  7. Summary comparison table

Usage:
    from prepare4.plot_paired_torques import (
        plot_paired_vs_solo_timeseries,
        plot_root_force_comparison,
        plot_torque_delta_heatmap,
        plot_gt_vs_gen_comparison,
        plot_sii_distribution,
    )
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Constants ──
FPS = 30
GRAVITY = 9.81
DEFAULT_BODY_MASS = 75.0

BODY_GROUPS = {
    "L_Leg":       slice(6, 18),
    "R_Leg":       slice(18, 30),
    "Spine/Torso": slice(30, 45),
    "L_Arm":       slice(45, 60),
    "R_Arm":       slice(60, 75),
}

GROUP_COLORS = {
    "L_Leg": "#1f77b4",
    "R_Leg": "#ff7f0e",
    "Spine/Torso": "#2ca02c",
    "L_Arm": "#d62728",
    "R_Arm": "#9467bd",
}

JOINT_NAMES_23 = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]


def plot_paired_vs_solo_timeseries(result, clip_id="", text="",
                                    save_path=None, person="A"):
    """Plot torque time-series comparing paired vs solo for one person.

    3 rows × 5 columns (one per body group):
      Row 1: Paired torques (|τ| per body group)
      Row 2: Solo torques
      Row 3: Delta (|τ_paired - τ_solo|) with shaded regions

    Args:
        result: dict from compute_paired_vs_solo()
        clip_id: clip identifier for title
        text: motion description text
        save_path: file path to save figure
        person: 'A' or 'B'
    """
    tau_p = result[f'torques_paired_{person}']
    tau_s = result[f'torques_solo_{person}']
    T = tau_p.shape[0]
    time_axis = np.arange(T) / FPS

    fig, axes = plt.subplots(3, 5, figsize=(22, 10), sharex=True)
    fig.suptitle(f"Paired vs Solo Torques — Person {person}\n"
                 f"Clip {clip_id}: {text}", fontsize=12)

    groups = list(BODY_GROUPS.items())

    for col, (gname, gslice) in enumerate(groups):
        color = GROUP_COLORS[gname]

        # Row 0: Paired
        ax = axes[0, col]
        group_tau = np.abs(tau_p[:, gslice]).mean(axis=-1)
        ax.plot(time_axis, group_tau, color=color, linewidth=0.8)
        ax.set_ylabel("Paired |τ| (Nm)" if col == 0 else "")
        ax.set_title(gname, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 1: Solo
        ax = axes[1, col]
        group_tau_s = np.abs(tau_s[:, gslice]).mean(axis=-1)
        ax.plot(time_axis, group_tau_s, color=color, linewidth=0.8)
        ax.set_ylabel("Solo |τ| (Nm)" if col == 0 else "")
        ax.grid(True, alpha=0.3)

        # Row 2: Delta
        ax = axes[2, col]
        delta = np.abs(tau_p[:, gslice] - tau_s[:, gslice]).mean(axis=-1)
        ax.fill_between(time_axis, 0, delta, alpha=0.4, color=color)
        ax.plot(time_axis, delta, color=color, linewidth=0.8)
        ax.set_ylabel("|Δτ| (Nm)" if col == 0 else "")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_root_force_comparison(result, clip_id="", text="",
                                save_path=None):
    """Plot root translational forces: paired vs solo for both persons.

    2×2 grid: rows = person A/B, cols = paired/solo.
    Plus a 5th panel showing the delta (interaction force proxy).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    fig.suptitle(f"Root Translational Forces — Clip {clip_id}: {text}",
                 fontsize=12)

    axis_labels = ['X', 'Y', 'Z']
    axis_colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for row, person in enumerate(['A', 'B']):
        tau_p = result[f'torques_paired_{person}']
        tau_s = result[f'torques_solo_{person}']
        T = tau_p.shape[0]
        time_axis = np.arange(T) / FPS

        # Paired
        ax = axes[row, 0]
        for d in range(3):
            ax.plot(time_axis, tau_p[:, d], color=axis_colors[d],
                    label=axis_labels[d], linewidth=0.8)
        ax.set_title(f"Person {person} — Paired" if row == 0 else "")
        ax.set_ylabel(f"Person {person}\nForce (N)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Body weight reference
        ax.axhline(DEFAULT_BODY_MASS * GRAVITY, color='gray',
                    linestyle='--', alpha=0.5, label='BW')

        # Solo
        ax = axes[row, 1]
        for d in range(3):
            ax.plot(time_axis, tau_s[:, d], color=axis_colors[d],
                    linewidth=0.8)
        ax.set_title("Solo" if row == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.axhline(DEFAULT_BODY_MASS * GRAVITY, color='gray',
                    linestyle='--', alpha=0.5)

        # Delta (interaction force proxy)
        ax = axes[row, 2]
        delta = tau_p[:, :3] - tau_s[:, :3]
        for d in range(3):
            ax.plot(time_axis, delta[:, d], color=axis_colors[d],
                    linewidth=0.8)
        delta_mag = np.linalg.norm(delta, axis=-1)
        ax.plot(time_axis, delta_mag, 'k-', linewidth=1.0, label='|ΔF|')
        ax.set_title("Delta (interaction proxy)" if row == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5)

    axes[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_torque_delta_heatmap(result, clip_id="", text="",
                               save_path=None, person="A"):
    """Heatmap of |τ_paired - τ_solo| over joints × time.

    Each row = one of 23 joints (mean across 3 DOFs per joint).
    Each column = one frame.
    """
    tau_p = result[f'torques_paired_{person}']
    tau_s = result[f'torques_solo_{person}']
    T = tau_p.shape[0]

    # Compute per-joint delta (23 joints × T frames)
    heatmap = np.zeros((23, T), dtype=np.float32)
    for j in range(23):
        dof_start = 6 + j * 3
        dof_end = dof_start + 3
        heatmap[j] = np.abs(tau_p[:, dof_start:dof_end]
                           - tau_s[:, dof_start:dof_end]).mean(axis=-1)

    fig, ax = plt.subplots(figsize=(16, 8))
    time_axis = np.arange(T) / FPS

    im = ax.imshow(heatmap, aspect='auto', interpolation='nearest',
                   extent=[0, time_axis[-1], 22.5, -0.5],
                   cmap='hot')
    ax.set_yticks(range(23))
    ax.set_yticklabels(JOINT_NAMES_23, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Torque Delta Heatmap — Person {person}\n"
                 f"Clip {clip_id}: {text}")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("|Δτ| (Nm)")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_newton3_compliance(result, clip_id="", text="",
                             save_path=None):
    """Plot Newton's 3rd law compliance over time.

    Shows |F_int_A + F_int_B| (should be ~0 for consistent interactions)
    and the N3LV normalized metric.
    """
    from prepare4.interaction_metrics import newtons_third_law_violation

    tau_p_A = result['torques_paired_A']
    tau_s_A = result['torques_solo_A']
    tau_p_B = result['torques_paired_B']
    tau_s_B = result['torques_solo_B']

    n3lv, f_int_A, f_int_B = newtons_third_law_violation(
        tau_p_A, tau_s_A, tau_p_B, tau_s_B)

    T = n3lv.shape[0]
    time_axis = np.arange(T) / FPS

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Newton's 3rd Law Analysis — Clip {clip_id}: {text}",
                 fontsize=12)

    # Panel 1: Interaction force magnitudes
    ax = axes[0]
    f_int_A_mag = np.linalg.norm(f_int_A, axis=-1)
    f_int_B_mag = np.linalg.norm(f_int_B, axis=-1)
    ax.plot(time_axis, f_int_A_mag, 'b-', label='|F_int_A|', linewidth=0.8)
    ax.plot(time_axis, f_int_B_mag, 'r-', label='|F_int_B|', linewidth=0.8)
    ax.set_ylabel("Force (N)")
    ax.set_title("Interaction Force Proxy Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Violation vector
    ax = axes[1]
    violation_mag = np.linalg.norm(f_int_A + f_int_B, axis=-1)
    ax.fill_between(time_axis, 0, violation_mag, alpha=0.3, color='red')
    ax.plot(time_axis, violation_mag, 'r-', linewidth=0.8)
    ax.set_ylabel("|F_A + F_B| (N)")
    ax.set_title("Newton's 3rd Law Violation (should be ~0)")
    ax.grid(True, alpha=0.3)

    # Panel 3: Normalized N3LV
    ax = axes[2]
    ax.plot(time_axis, n3lv, 'k-', linewidth=0.8)
    ax.fill_between(time_axis, 0, n3lv, alpha=0.2, color='orange')
    ax.set_ylabel("N3LV")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Normalized N3LV (mean={n3lv.mean():.3f})")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_gt_vs_gen_comparison(gt_agg, gen_agg, save_path=None):
    """Bar chart comparing GT vs Generated across key metrics.

    Args:
        gt_agg: aggregated stats dict for GT
        gen_agg: aggregated stats dict for Generated
        save_path: file path to save
    """
    metrics_to_plot = [
        ('sii_A', 'SII (A)', '', True),
        ('sii_B', 'SII (B)', '', True),
        ('n3lv_mean', 'N3LV', '', False),
        ('bps_paired_A', 'BPS paired (A)', '', True),
        ('bps_solo_A', 'BPS solo (A)', '', True),
        ('torque_delta_A_mean', 'TD hinge (A)', 'Nm', False),
        ('root_force_delta_trans_A_mean', 'Root ΔF (A)', 'N', False),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
    fig.suptitle("GT vs Generated — Interaction Plausibility Metrics",
                 fontsize=13)

    for i, (key, label, unit, is_pct) in enumerate(metrics_to_plot):
        ax = axes[i]
        gt_val = gt_agg.get(key, {}).get('mean', 0)
        gen_val = gen_agg.get(key, {}).get('mean', 0)
        gt_std = gt_agg.get(key, {}).get('std', 0)
        gen_std = gen_agg.get(key, {}).get('std', 0)

        bars = ax.bar(['GT', 'Gen'], [gt_val, gen_val],
                       yerr=[gt_std, gen_std],
                       color=['#2196F3', '#FF5722'], capsize=5)
        ax.set_title(label, fontsize=10)
        if is_pct:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        if unit:
            ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_sii_distribution(gt_results, gen_results, save_path=None):
    """Histogram of Solo Impossibility Index, GT vs Generated.

    Args:
        gt_results: list of per-clip result dicts (GT)
        gen_results: list of per-clip result dicts (Generated)
    """
    gt_sii = [(r['metrics']['sii_A'] + r['metrics']['sii_B']) / 2
              for r in gt_results if 'sii_A' in r.get('metrics', {})]
    gen_sii = [(r['metrics']['sii_A'] + r['metrics']['sii_B']) / 2
               for r in gen_results if 'sii_A' in r.get('metrics', {})]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)

    if gt_sii:
        ax.hist(gt_sii, bins=bins, alpha=0.6, color='#2196F3',
                label=f'GT (n={len(gt_sii)})', density=True)
    if gen_sii:
        ax.hist(gen_sii, bins=bins, alpha=0.6, color='#FF5722',
                label=f'Generated (n={len(gen_sii)})', density=True)

    ax.set_xlabel("Solo Impossibility Index (SII)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Solo Impossibility Index\n"
                 "(fraction of frames impossible without partner)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if gt_sii:
        ax.axvline(np.mean(gt_sii), color='#2196F3', linestyle='--',
                   label=f'GT mean={np.mean(gt_sii):.2f}')
    if gen_sii:
        ax.axvline(np.mean(gen_sii), color='#FF5722', linestyle='--',
                   label=f'Gen mean={np.mean(gen_sii):.2f}')
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_per_group_torque_delta(gt_agg, gen_agg, save_path=None):
    """Grouped bar chart: per-body-group mean torque delta, GT vs Gen."""
    groups = ["L_Leg", "R_Leg", "Spine/Torso", "L_Arm", "R_Arm"]

    gt_vals = []
    gen_vals = []
    for g in groups:
        key = f'td_A_{g}'
        # These are per-clip metrics; need to check if they're in agg
        # Fallback: use overall torque_delta_A_mean
        gt_v = gt_agg.get(key, {}).get('mean', 0)
        gen_v = gen_agg.get(key, {}).get('mean', 0)
        gt_vals.append(gt_v)
        gen_vals.append(gen_v)

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, gt_vals, width, label='GT', color='#2196F3')
    ax.bar(x + width/2, gen_vals, width, label='Generated', color='#FF5722')

    ax.set_xlabel("Body Group")
    ax.set_ylabel("Mean Torque Delta (Nm)")
    ax.set_title("Paired-Solo Torque Delta by Body Group (Person A)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def generate_all_clip_plots(result, clip_id, text="", save_dir=None):
    """Generate all per-clip plots.

    Args:
        result: dict from compute_paired_vs_solo()
        clip_id: clip identifier
        text: motion description
        save_dir: directory to save plots (default: output/paired_analysis/{clip_id})
    """
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "output", "paired_analysis",
                                str(clip_id))
    os.makedirs(save_dir, exist_ok=True)

    plot_paired_vs_solo_timeseries(
        result, clip_id, text, person="A",
        save_path=os.path.join(save_dir, "torque_timeseries_A.png"))
    plot_paired_vs_solo_timeseries(
        result, clip_id, text, person="B",
        save_path=os.path.join(save_dir, "torque_timeseries_B.png"))

    plot_root_force_comparison(
        result, clip_id, text,
        save_path=os.path.join(save_dir, "root_forces.png"))

    plot_torque_delta_heatmap(
        result, clip_id, text, person="A",
        save_path=os.path.join(save_dir, "delta_heatmap_A.png"))
    plot_torque_delta_heatmap(
        result, clip_id, text, person="B",
        save_path=os.path.join(save_dir, "delta_heatmap_B.png"))

    plot_newton3_compliance(
        result, clip_id, text,
        save_path=os.path.join(save_dir, "newton3_compliance.png"))

    print(f"  Plots saved to {save_dir}/")
