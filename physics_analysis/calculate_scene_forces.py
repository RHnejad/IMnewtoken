import os
import sys
import torch
import numpy as np
import trimesh
import smplx
import matplotlib.pyplot as plt

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.data_adapter import InterHumanPairDataset
from newton_vqvae.physics_losses import SMPL_SEGMENT_MASS_RATIOS

# Physics Constants
UP_AXIS = 1
GRAVITY = 9.81
FPS = 30.0
HUMAN_DENSITY_KG_M3 = 1000.0
# Foot contact flags are stored in motion channels 258:262
# (pre-computed during data processing using velocity + height criteria)

# ── Motion vector layout (per person, 262 dims total) ──
# [0:66]    = joint positions  (22 joints × 3)
# [66:132]  = joint velocities (22 joints × 3)
# [132:258] = joint rotations  (21 joints × 6, continuous 6d)
# [258:262] = foot contacts    (4 binary flags)
POS_SLICE = slice(0, 66)

seg_mass = torch.tensor(SMPL_SEGMENT_MASS_RATIOS, dtype=torch.float32)
seg_mass = seg_mass / seg_mass.sum()  # shape (22,)


def get_com(positions: torch.Tensor, device) -> torch.Tensor:
    """Compute Center of Mass.  positions: (T, 22, 3) -> (T, 3)"""
    w = seg_mass.to(device).view(1, -1, 1)
    return (positions * w).sum(dim=1)


def finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    return (x[1:] - x[:-1]) / dt


def detect_ground_contact(motion: torch.Tensor) -> torch.Tensor:
    """Detect ground contact from pre-computed foot contact flags.
    motion: (T, 262) per-person motion vector
    Returns: (T,) bool — True if any foot contact flag is active
    """
    foot_contacts = motion[:, 258:262]  # (T, 4) pre-computed contact flags
    return (foot_contacts > 0.5).any(dim=-1)


def estimate_mass_from_betas(smpl_model, betas: torch.Tensor) -> float:
    with torch.no_grad():
        output = smpl_model(betas=betas)
        vertices = output.vertices[0].cpu().numpy()
        faces = smpl_model.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return float(mesh.volume * HUMAN_DENSITY_KG_M3)


def analyze_scene_and_interaction_forces(motion_A: torch.Tensor, motion_B: torch.Tensor,
                                          mass_A: float, mass_B: float,
                                          seq_idx, save_dir: str):
    """
    Compute scene (ground) forces and interaction forces using
    contact-state-aware Newton's laws.

    System dynamics (always valid — interaction forces cancel by 3rd law):
        (m_A + m_B) · a_sys = F_ground_total + (m_A + m_B) · g
        => F_ground_total = m_A·a_A + m_B·a_B − (m_A+m_B)·g

    Distribution of ground force to individual people depends on contact state:

    ┌────────────────┬─────────────────────────────────────────────────────┐
    │ Both grounded  │ F_ground_total is known from system eq.            │
    │                │ Individual F_ground_A, F_ground_B underdetermined. │
    │                │ Total scene force = F_ground_total.                │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │ A floating     │ F_ground_A = 0 → all scene force on B:            │
    │                │ F_ground_B = F_ground_total                        │
    │                │ F_{B→A} = m_A(a_A − g),  F_{A→B} = −F_{B→A}      │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │ B floating     │ F_ground_B = 0 → all scene force on A:            │
    │                │ F_ground_A = F_ground_total                        │
    │                │ F_{A→B} = m_B(a_B − g),  F_{B→A} = −F_{A→B}      │
    ├────────────────┼─────────────────────────────────────────────────────┤
    │ Both floating  │ F_ground_total SHOULD be zero (free fall).         │
    │                │ Any non-zero value = scene force violation.        │
    │                │ Interaction from each eq; check 3rd law.           │
    └────────────────┴─────────────────────────────────────────────────────┘
    """
    T = motion_A.shape[0]
    dt = 1.0 / FPS
    device = motion_A.device

    g_vec = torch.zeros(3, device=device)
    g_vec[UP_AXIS] = -GRAVITY

    # ── Positions, COM, accelerations ──
    pos_A = motion_A[:, POS_SLICE].reshape(T, 22, 3)
    pos_B = motion_B[:, POS_SLICE].reshape(T, 22, 3)

    com_A = get_com(pos_A, device)
    com_B = get_com(pos_B, device)

    acc_A = finite_diff(finite_diff(com_A, dt), dt)  # (Ta, 3)
    acc_B = finite_diff(finite_diff(com_B, dt), dt)
    Ta = acc_A.shape[0]

    # ── System-level total ground force (always valid) ──
    F_ground_total = mass_A * acc_A + mass_B * acc_B - (mass_A + mass_B) * g_vec

    # ── Ground contact (trim to match acceleration frames) ──
    contact_A = detect_ground_contact(motion_A)[1:-1]
    contact_B = detect_ground_contact(motion_B)[1:-1]

    A_float = ~contact_A
    B_float = ~contact_B
    both_float = A_float & B_float
    A_float_only = A_float & contact_B
    B_float_only = contact_A & B_float
    both_ground = contact_A & contact_B

    # ── Distribute ground force by contact state ──
    F_ground_A = torch.full((Ta, 3), float('nan'), device=device)
    F_ground_B = torch.full((Ta, 3), float('nan'), device=device)

    # A floating: all ground force goes to B
    if A_float.any():
        F_ground_A[A_float] = 0.0
    if A_float_only.any():
        F_ground_B[A_float_only] = F_ground_total[A_float_only]

    # B floating: all ground force goes to A
    if B_float.any():
        F_ground_B[B_float] = 0.0
    if B_float_only.any():
        F_ground_A[B_float_only] = F_ground_total[B_float_only]

    # Both floating: ground force should be zero (free-fall)
    if both_float.any():
        F_ground_A[both_float] = 0.0
        F_ground_B[both_float] = 0.0

    # Both grounded: total is known, individual split is underdetermined
    # Leave F_ground_A, F_ground_B as NaN for these frames

    # ── Interaction forces (only solvable when ≥1 floating) ──
    F_B_on_A = torch.full((Ta, 3), float('nan'), device=device)
    F_A_on_B = torch.full((Ta, 3), float('nan'), device=device)
    newton3_err = torch.full((Ta,), float('nan'), device=device)

    if A_float_only.any():
        F_B_on_A[A_float_only] = mass_A * (acc_A[A_float_only] - g_vec)
        F_A_on_B[A_float_only] = -F_B_on_A[A_float_only]
        newton3_err[A_float_only] = 0.0

    if B_float_only.any():
        F_A_on_B[B_float_only] = mass_B * (acc_B[B_float_only] - g_vec)
        F_B_on_A[B_float_only] = -F_A_on_B[B_float_only]
        newton3_err[B_float_only] = 0.0

    if both_float.any():
        F_BA_from_A = mass_A * (acc_A[both_float] - g_vec)
        F_AB_from_B = mass_B * (acc_B[both_float] - g_vec)
        newton3_err[both_float] = (F_BA_from_A + F_AB_from_B).norm(dim=-1)
        F_B_on_A[both_float] = F_BA_from_A
        F_A_on_B[both_float] = F_AB_from_B

    # ── Scene force violation: when both floating, total ground force should ≈ 0 ──
    scene_violation = torch.zeros(Ta, device=device)
    if both_float.any():
        scene_violation[both_float] = F_ground_total[both_float].norm(dim=-1)

    # ── Ground reaction force sanity: vertical component ≥ 0 ──
    grf_neg_A = torch.zeros(Ta, device=device)
    grf_neg_B = torch.zeros(Ta, device=device)
    solv_A = ~torch.isnan(F_ground_A[:, 0])
    solv_B = ~torch.isnan(F_ground_B[:, 0])
    if solv_A.any():
        grf_neg_A[solv_A] = torch.relu(-F_ground_A[solv_A, UP_AXIS])
    if solv_B.any():
        grf_neg_B[solv_B] = torch.relu(-F_ground_B[solv_B, UP_AXIS])

    # ── Plotting (4-panel) ──
    time_axis = np.arange(Ta) * dt
    sys_weight = (mass_A + mass_B) * GRAVITY

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Panel 1: Contact state
    ax = axes[0]
    state = np.full(Ta, 3)
    state[A_float_only.cpu().numpy()] = 1
    state[B_float_only.cpu().numpy()] = 2
    state[both_float.cpu().numpy()] = 0
    colors_map = {0: '#e74c3c', 1: '#3498db', 2: '#e67e22', 3: '#2ecc71'}
    labels_map = {0: 'Both floating', 1: 'A floating', 2: 'B floating', 3: 'Both grounded'}
    for sv, c in colors_map.items():
        m = state == sv
        if m.any():
            ax.fill_between(time_axis, 0, 1, where=m, color=c, alpha=0.6, label=labels_map[sv])
    ax.set_ylabel('Contact')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Scene & Interaction Forces (Contact-Aware) — Seq {seq_idx}')

    # Panel 2: Total system ground force
    ax = axes[1]
    F_total_mag = F_ground_total.norm(dim=-1).cpu().numpy()
    F_total_vert = F_ground_total[:, UP_AXIS].cpu().numpy()
    ax.plot(time_axis, F_total_mag, label='|F_ground_total| (system)', color='green', linewidth=1.5)
    ax.plot(time_axis, F_total_vert, label='F_ground_total vertical', color='green', linestyle='--', alpha=0.6)
    ax.axhline(y=sys_weight, color='grey', linestyle=':', alpha=0.5, label=f'System weight ({sys_weight:.0f} N)')
    ax.set_ylabel('Force (N)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Interaction forces (only where solvable)
    ax = axes[2]
    solvable = ~torch.isnan(F_B_on_A[:, 0])
    F_BA_mag = torch.where(solvable, F_B_on_A.norm(dim=-1), torch.tensor(0.0, device=device)).cpu().numpy()
    F_AB_mag = torch.where(solvable, F_A_on_B.norm(dim=-1), torch.tensor(0.0, device=device)).cpu().numpy()
    ax.plot(time_axis, F_BA_mag, label='|F_{B→A}|', color='blue', alpha=0.8)
    ax.plot(time_axis, F_AB_mag, label='|F_{A→B}|', color='red', alpha=0.8, linestyle='--')
    ax.fill_between(time_axis, 0, max(F_BA_mag.max(), F_AB_mag.max(), 1),
                    where=both_ground.cpu().numpy(), color='grey', alpha=0.15, label='Undetermined')
    ax.set_ylabel('Interaction (N)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Physics violations
    ax = axes[3]
    n3 = torch.where(~torch.isnan(newton3_err), newton3_err, torch.tensor(0.0, device=device)).cpu().numpy()
    sv = scene_violation.cpu().numpy()
    gn_A = grf_neg_A.cpu().numpy()
    gn_B = grf_neg_B.cpu().numpy()
    ax.plot(time_axis, n3, label="Newton's 3rd law err", color='black', linewidth=1.5)
    ax.plot(time_axis, sv, label='Scene violation (both floating)', color='red', alpha=0.7)
    ax.plot(time_axis, gn_A, label='Negative GRF_A', color='blue', alpha=0.5, linestyle=':')
    ax.plot(time_axis, gn_B, label='Negative GRF_B', color='orange', alpha=0.5, linestyle=':')
    ax.set_ylabel('Violation (N)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'scene_interaction_seq_{seq_idx}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    # ── Return summary stats ──
    return {
        'sys_weight': sys_weight,
        'avg_scene_force': float(F_total_mag.mean()),
        'max_scene_force': float(F_total_mag.max()),
        'avg_interact': float(F_BA_mag[solvable.cpu().numpy()].mean()) if solvable.any() else 0.0,
        'avg_newton3': float(n3[both_float.cpu().numpy()].mean()) if both_float.any() else 0.0,
        'avg_scene_violation': float(sv[both_float.cpu().numpy()].mean()) if both_float.any() else 0.0,
        'pct_both_ground': both_ground.sum().item() / Ta * 100,
        'pct_any_float': (Ta - both_ground.sum().item()) / Ta * 100,
    }


def main():
    print("Loading dataset...")
    dataset = InterHumanPairDataset(
        data_root=os.path.join(_PROJECT_ROOT, "data/InterHuman"),
        mode='train',
        window_size=64,
        window_stride=64,
    )

    save_dir = os.path.join(_PROJECT_ROOT, 'physics_analysis')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize SMPL-X model
    model_path = os.path.join(_PROJECT_ROOT, "data", "body_model")
    try:
        smpl_model = smplx.create(model_path=model_path, model_type='smplx', gender='neutral', batch_size=1)
        smpl_model.eval()
    except Exception as e:
        print(f"Error loading SMPLX model: {e}")
        return

    target_clip_ids = ["1000", "2001"]

    # Pre-search for clips in the dataset
    found_clips = {}
    for pidx, (m1_full, m2_full, b1, b2, clip_id) in enumerate(dataset._pairs):
        if clip_id in target_clip_ids:
            found_clips[clip_id] = (pidx, m1_full, m2_full, b1, b2)

    for clip_id in target_clip_ids:
        if clip_id not in found_clips:
            print(f"Clip ID '{clip_id}' not found.")
            continue

        pidx, m1_full, m2_full, b1, b2 = found_clips[clip_id]
        print(f"\nClip '{clip_id}' — {m1_full.shape[0]} frames")

        betas_A = torch.from_numpy(b1).unsqueeze(0).float()
        betas_B = torch.from_numpy(b2).unsqueeze(0).float()
        mass_A = estimate_mass_from_betas(smpl_model, betas_A)
        mass_B = estimate_mass_from_betas(smpl_model, betas_B)

        # _pairs stores already-unnormalized data (output of _process_motion_np)
        # Do NOT apply normalizer.backward — that would double-unnormalize
        m1_u = torch.from_numpy(m1_full).float()
        m2_u = torch.from_numpy(m2_full).float()

        s = analyze_scene_and_interaction_forces(
            m1_u, m2_u, mass_A, mass_B, f"clip_{clip_id}_full", save_dir
        )

        print(f"  Mass A = {mass_A:.1f} kg,  Mass B = {mass_B:.1f} kg")
        print(f"  System weight        : {s['sys_weight']:.1f} N")
        print(f"  Avg |F_ground_total| : {s['avg_scene_force']:.1f} N")
        print(f"  Max |F_ground_total| : {s['max_scene_force']:.1f} N")
        print(f"  Avg interaction      : {s['avg_interact']:.1f} N  (only solvable frames)")
        print(f"  Both grounded        : {s['pct_both_ground']:.1f}%")
        print(f"  ≥1 floating          : {s['pct_any_float']:.1f}%")
        print(f"  Avg Newton 3rd err   : {s['avg_newton3']:.2f} N  (both-float frames)")
        print(f"  Avg scene violation  : {s['avg_scene_violation']:.2f} N  (both-float frames)")

if __name__ == "__main__":
    main()
