"""
plot_clip_torques.py — Compute and plot per-clip torque & inter-person force time series.

For each clip, computes inverse dynamics torques for both GT and Generated motions
(both persons), then generates:
  1. Torque magnitude over time (per body group)
  2. Inter-person applied force over time (from CoM accelerations)

Usage:
    conda run -n mimickit --no-capture-output python prepare4/plot_clip_torques.py \
        --clips 1129 1147 1187 121 1006 1441
"""
import os
import sys
import pickle
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ── Constants ──
FPS = 30
DT = 1.0 / FPS
BODY_MASS = 75.0  # kg

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


BODY_NAMES_JOINTS = [
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]


def load_gt_persons(clip_id):
    """Load GT persons from InterHuman dataset."""
    from prepare4.retarget import load_interhuman_pkl
    data_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
    return load_interhuman_pkl(data_dir, str(clip_id))


def load_gen_persons(clip_id):
    """Load generated persons from InterMask output."""
    path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman",
                        f"{clip_id}.pkl")
    if not os.path.isfile(path):
        return None, None
    with open(path, "rb") as f:
        raw = pickle.load(f)

    text = raw.get("text", f"Clip {clip_id}")
    results = []
    for pkey in ["person1", "person2"]:
        if pkey not in raw:
            return None, text
        p = raw[pkey]
        d = {k: p[k].astype(np.float64) for k in
             ["root_orient", "pose_body", "trans", "betas"]}
        if "positions_zup" in p:
            d["positions_zup"] = p["positions_zup"].astype(np.float64)
        results.append(d)
    return results, text


def compute_torques_for_person(person_data, source, device="cuda:0"):
    """Compute inverse dynamics torques for one person.

    Returns (T, 75) torque array or None.
    """
    from prepare4.retarget import rotation_retarget, ik_retarget
    from prepare4.dynamics import inverse_dynamics

    if source == "generated":
        if "positions_zup" not in person_data:
            return None
        joint_q, _ = ik_retarget(
            person_data["positions_zup"], person_data["betas"],
            ik_iters=50, device=device, sequential=True,
        )
    else:
        joint_q = rotation_retarget(
            person_data["root_orient"], person_data["pose_body"],
            person_data["trans"], person_data["betas"],
        )
        joint_q = joint_q[::2]  # 60fps → 30fps

    if joint_q.shape[0] < 11:
        return None

    torques, _, _ = inverse_dynamics(
        joint_q, DT, person_data["betas"],
        total_mass=BODY_MASS, device=device, verbose=False,
    )
    return torques


def compute_com_acceleration(positions_zup):
    """Compute center-of-mass acceleration from joint positions.

    Uses SavGol differentiation on the mean joint position (≈ CoM proxy).
    positions_zup: (T, 22, 3)
    Returns: (T, 3) acceleration in m/s²
    """
    com = positions_zup.mean(axis=1)  # (T, 3)
    T = com.shape[0]
    if T < 11:
        return np.zeros_like(com)
    accel = np.zeros_like(com)
    for d in range(3):
        accel[:, d] = savgol_filter(com[:, d], min(11, T if T % 2 == 1 else T - 1),
                                     5, deriv=2, delta=DT)
    return accel


def compute_interaction_force(positions_p1, positions_p2, mass=BODY_MASS):
    """Estimate interaction force between two persons.

    Net external force on each person = m * a_com
    Gravity contributes m*g downward.
    Interaction force ≈ m * a_com - m * g (removing gravity)

    For the horizontal/interaction component, we look at the force
    along the inter-person axis (connecting the two CoMs).

    Returns dict with time series of force magnitudes.
    """
    com1 = positions_p1.mean(axis=1)  # (T, 3)
    com2 = positions_p2.mean(axis=1)

    T = min(com1.shape[0], com2.shape[0])
    com1 = com1[:T]
    com2 = com2[:T]

    accel1 = compute_com_acceleration(positions_p1[:T])
    accel2 = compute_com_acceleration(positions_p2[:T])

    # Net force = m * a  (includes all forces: gravity, ground reaction, interaction)
    force1 = mass * accel1  # (T, 3)
    force2 = mass * accel2

    # Remove gravity (Z-up: gravity along -Z)
    gravity = np.array([0.0, 0.0, -9.81])
    ext_force1 = force1 - mass * gravity  # External forces excluding gravity
    ext_force2 = force2 - mass * gravity

    # Force magnitude (total external excluding gravity)
    force_mag1 = np.linalg.norm(ext_force1, axis=1)
    force_mag2 = np.linalg.norm(ext_force2, axis=1)

    # Inter-person direction (CoM1 → CoM2)
    delta = com2 - com1
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    dist = np.clip(dist, 0.01, None)
    direction = delta / dist

    # Force along inter-person axis
    force_along1 = np.sum(ext_force1 * direction, axis=1)
    force_along2 = np.sum(ext_force2 * (-direction), axis=1)

    # Horizontal force magnitude (XY plane in Z-up)
    horiz_force1 = np.linalg.norm(ext_force1[:, :2], axis=1)
    horiz_force2 = np.linalg.norm(ext_force2[:, :2], axis=1)

    return {
        "force_mag_p1": force_mag1,
        "force_mag_p2": force_mag2,
        "force_along_p1": force_along1,
        "force_along_p2": force_along2,
        "horiz_force_p1": horiz_force1,
        "horiz_force_p2": horiz_force2,
        "com_dist": dist.squeeze(),
    }


def load_positions_zup(clip_id, source):
    """Load positions_zup for both persons."""
    if source == "generated":
        path = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman",
                            f"{clip_id}.pkl")
        if not os.path.isfile(path):
            return None, None
        with open(path, "rb") as f:
            raw = pickle.load(f)
        p1 = raw["person1"].get("positions_zup")
        p2 = raw["person2"].get("positions_zup")
        return p1, p2
    else:
        # GT: load from motions_processed, convert to Z-up
        from data.utils import trans_matrix as TRANS_MATRIX_TORCH
        INV_TRANS = np.linalg.inv(TRANS_MATRIX_TORCH.numpy().astype(np.float64))

        gt_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")
        positions = []
        for pidx in [1, 2]:
            npy_path = os.path.join(gt_dir, "motions_processed",
                                    f"person{pidx}", f"{clip_id}.npy")
            if not os.path.isfile(npy_path):
                return None, None
            raw = np.load(npy_path).astype(np.float64)
            pos_yup = raw[:, :66].reshape(-1, 22, 3)
            pos_zup = np.einsum("mn,...n->...m", INV_TRANS, pos_yup)
            positions.append(pos_zup)
        return positions[0], positions[1]


def plot_comprehensive(gt_torques, gen_torques, forces_gt, forces_gen,
                       text, clip_id, out_dir):
    """Generate comprehensive multi-panel comparison plots for a single clip.

    Produces 3 files:
      1. {clip_id}_torque_comparison.png — 4 panels:
         - Overlaid histograms (all hinges)
         - Per-group bar chart (mean + P95)
         - Per-body ratio heatmap
         - Torque time series (per group, GT vs Gen overlaid)
      2. {clip_id}_forces.png — 3 panels:
         - Horizontal force over time
         - Force along inter-person axis
         - CoM distance over time
      3. {clip_id}_per_body_heatmap.png — Detailed heatmap of |τ| percentiles
    """
    from matplotlib.gridspec import GridSpec

    # ═══════════════════════════════════════════════════════
    # Figure 1: Torque Comparison (4 panels)
    # ═══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f'Torque Analysis — Clip {clip_id}\n"{text}"',
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Overlaid histograms ──
    ax1 = fig.add_subplot(gs[0, 0])
    gt_hinge = gt_torques[:, 6:].ravel()
    gen_hinge = gen_torques[:, 6:].ravel()
    bins = np.linspace(-100, 100, 101)
    ax1.hist(gt_hinge, bins=bins, density=True, alpha=0.5, color="steelblue",
             label=f"GT (N={gt_torques.shape[0]})")
    ax1.hist(gen_hinge, bins=bins, density=True, alpha=0.5, color="orangered",
             label=f"Gen (N={gen_torques.shape[0]})")
    ax1.set_xlabel("Torque (Nm)")
    ax1.set_ylabel("Density")
    ax1.set_title("All Hinge Torques: Distribution")
    ax1.legend(fontsize=9)
    ax1.axvline(0, color="k", linewidth=0.5)
    for p, ls in [(95, "--"), (99, ":")]:
        gt_v = np.percentile(np.abs(gt_hinge), p)
        gen_v = np.percentile(np.abs(gen_hinge), p)
        ax1.axvline(gt_v, color="steelblue", linestyle=ls, alpha=0.6)
        ax1.axvline(gen_v, color="orangered", linestyle=ls, alpha=0.6)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Per-group bar chart (mean + P95) ──
    ax2 = fig.add_subplot(gs[0, 1])
    group_names = list(BODY_GROUPS.keys())
    x = np.arange(len(group_names))
    width = 0.2
    for offset, (metric, label_suffix, alpha) in enumerate([
        ("mean", "mean", 0.9), ("p95", "P95", 0.6)
    ]):
        gt_vals, gen_vals = [], []
        for name in group_names:
            sl = BODY_GROUPS[name]
            gt_v = np.abs(gt_torques[:, sl]).ravel()
            gen_v = np.abs(gen_torques[:, sl]).ravel()
            if metric == "mean":
                gt_vals.append(gt_v.mean())
                gen_vals.append(gen_v.mean())
            else:
                gt_vals.append(np.percentile(gt_v, 95))
                gen_vals.append(np.percentile(gen_v, 95))
        ax2.bar(x - width - width/2 + offset * width, gt_vals, width,
                label=f"GT {label_suffix}", color="steelblue", alpha=alpha)
        ax2.bar(x + width/2 + offset * width, gen_vals, width,
                label=f"Gen {label_suffix}", color="orangered", alpha=alpha)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("|τ| (Nm)")
    ax2.set_title("Torque by Body Group")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Per-body ratio heatmap ──
    ax3 = fig.add_subplot(gs[1, 0])
    n_bodies = len(BODY_NAMES_JOINTS)
    pcts = [50, 75, 90, 95]
    ratio_map = np.zeros((n_bodies, len(pcts)))
    for b in range(n_bodies):
        dof_start = 6 + b * 3
        gt_body = np.abs(gt_torques[:, dof_start:dof_start + 3]).ravel()
        gen_body = np.abs(gen_torques[:, dof_start:dof_start + 3]).ravel()
        for j, p in enumerate(pcts):
            gt_p = np.percentile(gt_body, p)
            gen_p = np.percentile(gen_body, p)
            ratio_map[b, j] = gen_p / max(gt_p, 0.1)
    im = ax3.imshow(ratio_map, aspect="auto", cmap="RdYlGn_r",
                    vmin=0.5, vmax=3.0)
    ax3.set_xticks(range(len(pcts)))
    ax3.set_xticklabels([f"P{p}" for p in pcts])
    ax3.set_yticks(range(n_bodies))
    ax3.set_yticklabels(BODY_NAMES_JOINTS, fontsize=7)
    ax3.set_title("Gen/GT Ratio per Body")
    for i in range(n_bodies):
        for j in range(len(pcts)):
            v = ratio_map[i, j]
            color = "black" if 0.7 < v < 2.0 else "white"
            ax3.text(j, i, f"{v:.1f}", ha="center", va="center",
                     fontsize=6, color=color)
    fig.colorbar(im, ax=ax3, shrink=0.7, label="Gen/GT")

    # ── Panel 4: Per-body |τ| heatmap (GT and Gen side by side) ──
    ax4 = fig.add_subplot(gs[1, 1])
    pcts_abs = [50, 75, 90, 95]
    heatmap_combined = np.zeros((n_bodies, len(pcts_abs) * 2))
    xtick_labels = []
    for j, p in enumerate(pcts_abs):
        xtick_labels.append(f"GT\nP{p}")
        xtick_labels.append(f"Gen\nP{p}")
        for b in range(n_bodies):
            dof_start = 6 + b * 3
            gt_body = np.abs(gt_torques[:, dof_start:dof_start + 3]).ravel()
            gen_body = np.abs(gen_torques[:, dof_start:dof_start + 3]).ravel()
            heatmap_combined[b, j * 2] = np.percentile(gt_body, p)
            heatmap_combined[b, j * 2 + 1] = np.percentile(gen_body, p)
    im2 = ax4.imshow(heatmap_combined, aspect="auto", cmap="YlOrRd")
    ax4.set_xticks(range(len(pcts_abs) * 2))
    ax4.set_xticklabels(xtick_labels, fontsize=6)
    ax4.set_yticks(range(n_bodies))
    ax4.set_yticklabels(BODY_NAMES_JOINTS, fontsize=7)
    ax4.set_title("|τ| Percentiles (Nm)")
    for i in range(n_bodies):
        for j in range(len(pcts_abs) * 2):
            v = heatmap_combined[i, j]
            color = "black" if v < heatmap_combined.max() * 0.65 else "white"
            ax4.text(j, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=5, color=color)
    fig.colorbar(im2, ax=ax4, shrink=0.7, label="Nm")

    # ── Panel 5+6: Torque time series (overlaid GT vs Gen) ──
    time_gt = np.arange(gt_torques.shape[0]) / FPS
    time_gen = np.arange(gen_torques.shape[0]) / FPS

    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    for group_name, sl in list(BODY_GROUPS.items())[:3]:  # Legs + Spine
        color = GROUP_COLORS[group_name]
        tau_gt = np.abs(gt_torques[:, sl]).mean(axis=1)
        tau_gen = np.abs(gen_torques[:, sl]).mean(axis=1)
        ax5.plot(time_gt, tau_gt, color=color, linewidth=1.0, alpha=0.8,
                 label=f"{group_name}")
        ax5.plot(time_gen, tau_gen, color=color, linewidth=1.0, alpha=0.4,
                 linestyle="--")

    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("|τ| mean (Nm)")
    ax5.set_title("Torque Time Series — Legs & Spine (solid=GT, dashed=Gen)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    for group_name, sl in list(BODY_GROUPS.items())[3:]:  # Arms
        color = GROUP_COLORS[group_name]
        tau_gt = np.abs(gt_torques[:, sl]).mean(axis=1)
        tau_gen = np.abs(gen_torques[:, sl]).mean(axis=1)
        ax6.plot(time_gt, tau_gt, color=color, linewidth=1.0, alpha=0.8,
                 label=f"{group_name}")
        ax6.plot(time_gen, tau_gen, color=color, linewidth=1.0, alpha=0.4,
                 linestyle="--")

    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("|τ| mean (Nm)")
    ax6.set_title("Torque Time Series — Arms (solid=GT, dashed=Gen)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    torque_path = os.path.join(out_dir, f"{clip_id}_torque_comparison.png")
    fig.savefig(torque_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {torque_path}")

    # ═══════════════════════════════════════════════════════
    # Figure 2: Inter-Person Forces (3 panels)
    # ═══════════════════════════════════════════════════════
    if forces_gt is not None and forces_gen is not None:
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        fig2.suptitle(f'Inter-Person Forces — Clip {clip_id}\n"{text}"',
                      fontsize=13, fontweight='bold')

        T_gt = len(forces_gt["force_mag_p1"])
        T_gen = len(forces_gen["force_mag_p1"])
        t_gt = np.arange(T_gt) / FPS
        t_gen = np.arange(T_gen) / FPS

        # Panel A: Horizontal force
        ax = axes2[0, 0]
        ax.plot(t_gt, forces_gt["horiz_force_p1"], color="#1f77b4",
                linewidth=1.0, alpha=0.8, label='GT P1')
        ax.plot(t_gt, forces_gt["horiz_force_p2"], color="#d62728",
                linewidth=1.0, alpha=0.8, label='GT P2')
        ax.plot(t_gen, forces_gen["horiz_force_p1"], color="#1f77b4",
                linewidth=1.0, alpha=0.4, linestyle="--", label='Gen P1')
        ax.plot(t_gen, forces_gen["horiz_force_p2"], color="#d62728",
                linewidth=1.0, alpha=0.4, linestyle="--", label='Gen P2')
        ax.set_ylabel("Force (N)")
        ax.set_title("Horizontal Force (solid=GT, dashed=Gen)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel B: Force along inter-person axis
        ax = axes2[0, 1]
        ax.plot(t_gt, forces_gt["force_along_p1"], color="#2ca02c",
                linewidth=1.0, alpha=0.8, label='GT P1')
        ax.plot(t_gt, forces_gt["force_along_p2"], color="#ff7f0e",
                linewidth=1.0, alpha=0.8, label='GT P2')
        ax.plot(t_gen, forces_gen["force_along_p1"], color="#2ca02c",
                linewidth=1.0, alpha=0.4, linestyle="--", label='Gen P1')
        ax.plot(t_gen, forces_gen["force_along_p2"], color="#ff7f0e",
                linewidth=1.0, alpha=0.4, linestyle="--", label='Gen P2')
        ax.set_ylabel("Force (N)")
        ax.set_title("Force Along Inter-Person Axis")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)

        # Panel C: CoM distance
        ax = axes2[1, 0]
        ax.plot(t_gt, forces_gt["com_dist"], color="#9467bd",
                linewidth=1.5, label='GT')
        ax.plot(t_gen, forces_gen["com_dist"], color="#9467bd",
                linewidth=1.5, linestyle="--", label='Gen')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("CoM Distance Between Persons")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel D: Summary bar chart
        ax = axes2[1, 1]
        metrics = {
            'Horiz F\nmean': (
                np.mean([forces_gt["horiz_force_p1"].mean(),
                         forces_gt["horiz_force_p2"].mean()]),
                np.mean([forces_gen["horiz_force_p1"].mean(),
                         forces_gen["horiz_force_p2"].mean()]),
            ),
            'Horiz F\nP95': (
                np.mean([np.percentile(forces_gt["horiz_force_p1"], 95),
                         np.percentile(forces_gt["horiz_force_p2"], 95)]),
                np.mean([np.percentile(forces_gen["horiz_force_p1"], 95),
                         np.percentile(forces_gen["horiz_force_p2"], 95)]),
            ),
            'CoM dist\nmean': (
                forces_gt["com_dist"].mean(),
                forces_gen["com_dist"].mean(),
            ),
        }
        x_bar = np.arange(len(metrics))
        w = 0.35
        gt_bar = [v[0] for v in metrics.values()]
        gen_bar = [v[1] for v in metrics.values()]
        ax.bar(x_bar - w/2, gt_bar, w, label="GT", color="steelblue", alpha=0.8)
        ax.bar(x_bar + w/2, gen_bar, w, label="Gen", color="orangered", alpha=0.8)
        ax.set_xticks(x_bar)
        ax.set_xticklabels(list(metrics.keys()), fontsize=9)
        ax.set_title("Force Summary")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for i, (gt_v, gen_v) in enumerate(metrics.values()):
            ratio = gen_v / max(gt_v, 0.01)
            ax.text(i, max(gt_v, gen_v) * 1.05, f"×{ratio:.2f}",
                    ha="center", fontsize=8, color="gray")

        fig2.tight_layout(rect=[0, 0, 1, 0.93])
        force_path = os.path.join(out_dir, f"{clip_id}_forces.png")
        fig2.savefig(force_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"  Saved: {force_path}")


def process_clip(clip_id, out_dir, device="cuda:0"):
    """Process one clip: compute torques and forces, generate plots."""
    print(f"\n{'='*60}")
    print(f" Clip {clip_id}")
    print(f"{'='*60}")

    # Load generated data
    gen_persons, text = load_gen_persons(clip_id)
    if gen_persons is None:
        print(f"  No generated data for clip {clip_id}")
        return False

    # Load GT data
    gt_persons = load_gt_persons(clip_id)
    if gt_persons is None:
        print(f"  No GT data for clip {clip_id}")
        return False

    # ── Torques ──
    print(f"  Computing GT torques...")
    gt_torques_list = []
    for i, p in enumerate(gt_persons):
        t = compute_torques_for_person(p, "gt", device)
        if t is None:
            print(f"    Person {i+1}: FAILED")
            return False
        gt_torques_list.append(t)
        print(f"    Person {i+1}: {t.shape[0]} frames, "
              f"|τ_hinge| mean={np.abs(t[:, 6:]).mean():.1f} Nm")

    print(f"  Computing Generated torques...")
    gen_torques_list = []
    for i, p in enumerate(gen_persons):
        t = compute_torques_for_person(p, "generated", device)
        if t is None:
            print(f"    Person {i+1}: FAILED")
            return False
        gen_torques_list.append(t)
        print(f"    Person {i+1}: {t.shape[0]} frames, "
              f"|τ_hinge| mean={np.abs(t[:, 6:]).mean():.1f} Nm")

    # Average torques across persons for plotting
    T_gt = min(t.shape[0] for t in gt_torques_list)
    T_gen = min(t.shape[0] for t in gen_torques_list)
    gt_torques_avg = np.mean([t[:T_gt] for t in gt_torques_list], axis=0)
    gen_torques_avg = np.mean([t[:T_gen] for t in gen_torques_list], axis=0)

    # ── Inter-person forces ──
    print(f"  Computing inter-person forces...")
    gt_pos1, gt_pos2 = load_positions_zup(clip_id, "gt")
    gen_pos1, gen_pos2 = load_positions_zup(clip_id, "generated")

    forces_gt = None
    forces_gen = None
    if gt_pos1 is not None and gen_pos1 is not None:
        forces_gt = compute_interaction_force(gt_pos1, gt_pos2)
        forces_gen = compute_interaction_force(gen_pos1, gen_pos2)

    # ── Generate comprehensive plots ──
    print(f"  Generating plots...")
    plot_comprehensive(gt_torques_avg, gen_torques_avg,
                       forces_gt, forces_gen,
                       text, clip_id, out_dir)

    # Save raw data for later analysis
    np.savez(os.path.join(out_dir, f"{clip_id}_data.npz"),
             gt_torques_p1=gt_torques_list[0],
             gt_torques_p2=gt_torques_list[1],
             gen_torques_p1=gen_torques_list[0],
             gen_torques_p2=gen_torques_list[1],
             text=text, clip_id=clip_id)
    print(f"  Saved raw data: {clip_id}_data.npz")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot per-clip torque & force time series")
    parser.add_argument("--clips", type=str, nargs="+", required=True,
                        help="Clip IDs to process")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: prepare4/newton_videos/)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, "prepare4", "newton_videos")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for clip_id in args.clips:
        ok = process_clip(clip_id, out_dir, device=args.device)
        results[clip_id] = "OK" if ok else "FAILED"

    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"{'='*60}")
    for cid, status in results.items():
        print(f"  Clip {cid}: {status}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
