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


def analyze_interaction_forces(motion_A: torch.Tensor, motion_B: torch.Tensor,
                               mass_A: float, mass_B: float,
                               seq_idx, save_dir: str):
    """
    Compute interaction forces using contact-state-aware Newton's laws.

    For each person:  m · a = F_ground + m · g + F_interaction

    The interaction force F_{B→A} is only uniquely solvable when we know
    the ground force, which requires knowing the contact state:

    ┌────────────────┬──────────────────────────────────────────────────┐
    │ A floating     │ F_ground_A = 0                                  │
    │                │ F_{B→A} = m_A · (a_A − g)                       │
    │                │ F_{A→B} = −F_{B→A}   (Newton's 3rd law)         │
    ├────────────────┼──────────────────────────────────────────────────┤
    │ B floating     │ F_ground_B = 0                                  │
    │                │ F_{A→B} = m_B · (a_B − g)                       │
    │                │ F_{B→A} = −F_{A→B}                              │
    ├────────────────┼──────────────────────────────────────────────────┤
    │ Both floating  │ Solve from each: check 3rd law consistency      │
    ├────────────────┼──────────────────────────────────────────────────┤
    │ Both grounded  │ Under-determined. Cannot uniquely solve.        │
    └────────────────┴──────────────────────────────────────────────────┘
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

    # ── Ground contact (trim to match acceleration frames) ──
    contact_A = detect_ground_contact(motion_A)[1:-1]  # (Ta,)
    contact_B = detect_ground_contact(motion_B)[1:-1]

    A_float = ~contact_A
    B_float = ~contact_B
    both_float = A_float & B_float
    A_float_only = A_float & contact_B
    B_float_only = contact_A & B_float
    both_ground = contact_A & contact_B

    # ── Solve interaction forces per contact regime ──
    F_B_on_A = torch.full((Ta, 3), float('nan'), device=device)
    F_A_on_B = torch.full((Ta, 3), float('nan'), device=device)
    newton3_err = torch.full((Ta,), float('nan'), device=device)

    # Case 1: A floating → solve from A's equation, use 3rd law for B
    if A_float_only.any():
        F_B_on_A[A_float_only] = mass_A * (acc_A[A_float_only] - g_vec)
        F_A_on_B[A_float_only] = -F_B_on_A[A_float_only]
        newton3_err[A_float_only] = 0.0  # exact by construction

    # Case 2: B floating → solve from B's equation, use 3rd law for A
    if B_float_only.any():
        F_A_on_B[B_float_only] = mass_B * (acc_B[B_float_only] - g_vec)
        F_B_on_A[B_float_only] = -F_A_on_B[B_float_only]
        newton3_err[B_float_only] = 0.0  # exact by construction

    # Case 3: Both floating → solve independently from each equation, CHECK 3rd law
    if both_float.any():
        F_B_on_A_from_A = mass_A * (acc_A[both_float] - g_vec)
        F_A_on_B_from_B = mass_B * (acc_B[both_float] - g_vec)
        # Newton's 3rd law says these should sum to zero
        newton3_err[both_float] = (F_B_on_A_from_A + F_A_on_B_from_B).norm(dim=-1)
        F_B_on_A[both_float] = F_B_on_A_from_A
        F_A_on_B[both_float] = F_A_on_B_from_B

    # Case 4: Both grounded → interaction is underdetermined, leave as NaN

    # ── Also compute ground reaction forces where solvable ──
    F_ground_A = torch.full((Ta, 3), float('nan'), device=device)
    F_ground_B = torch.full((Ta, 3), float('nan'), device=device)

    F_ground_A[A_float] = 0.0
    F_ground_B[B_float] = 0.0

    # When B floating & A grounded: F_ground_A = m_A·a_A − m_A·g − F_{B→A}
    if B_float_only.any():
        F_ground_A[B_float_only] = (
            mass_A * acc_A[B_float_only] - mass_A * g_vec - F_B_on_A[B_float_only]
        )
    # When A floating & B grounded: F_ground_B = m_B·a_B − m_B·g − F_{A→B}
    if A_float_only.any():
        F_ground_B[A_float_only] = (
            mass_B * acc_B[A_float_only] - mass_B * g_vec - F_A_on_B[A_float_only]
        )

    # ── Build plotting arrays ──
    time_axis = np.arange(Ta) * dt
    solvable = ~torch.isnan(F_B_on_A[:, 0])

    F_BA_mag = torch.where(solvable, F_B_on_A.norm(dim=-1), torch.tensor(0.0)).cpu().numpy()
    F_AB_mag = torch.where(solvable, F_A_on_B.norm(dim=-1), torch.tensor(0.0)).cpu().numpy()
    n3_err = torch.where(~torch.isnan(newton3_err), newton3_err, torch.tensor(0.0)).cpu().numpy()

    # ── Plotting ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Contact state timeline
    ax = axes[0]
    state = np.full(Ta, 3)  # default = both grounded
    state[A_float_only.cpu().numpy()] = 1
    state[B_float_only.cpu().numpy()] = 2
    state[both_float.cpu().numpy()] = 0
    colors = {0: '#e74c3c', 1: '#3498db', 2: '#e67e22', 3: '#2ecc71'}
    labels = {0: 'Both floating', 1: 'A floating', 2: 'B floating', 3: 'Both grounded'}
    for s_val, color in colors.items():
        mask = state == s_val
        if mask.any():
            ax.fill_between(time_axis, 0, 1, where=mask, color=color, alpha=0.6, label=labels[s_val])
    ax.set_ylabel('Contact State')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Contact-Aware Interaction Forces — Seq {seq_idx}')

    # Panel 2: Solved interaction forces (only where solvable)
    ax = axes[1]
    ax.plot(time_axis, F_BA_mag, label=f'|F_{{B→A}}| (m_A={mass_A:.1f}kg)', color='blue', alpha=0.8)
    ax.plot(time_axis, F_AB_mag, label=f'|F_{{A→B}}| (m_B={mass_B:.1f}kg)', color='red', alpha=0.8, linestyle='--')
    # Grey out undetermined regions
    ax.fill_between(time_axis, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
                    where=both_ground.cpu().numpy(), color='grey', alpha=0.15, label='Undetermined (both grounded)')
    ax.set_ylabel('Interaction Force (N)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Newton's 3rd law error (only in both-floating frames)
    ax = axes[2]
    ax.plot(time_axis, n3_err, color='black', linewidth=1.5, label='|F_{B→A} + F_{A→B}|  (should ≈ 0)')
    ax.fill_between(time_axis, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10,
                    where=(~both_float).cpu().numpy(), color='grey', alpha=0.15, label='N/A (not both floating)')
    ax.set_ylabel("Newton's 3rd Law Error (N)")
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'interaction_forces_seq_{seq_idx}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    # ── Statistics ──
    n_solvable = solvable.sum().item()
    n_both_float = both_float.sum().item()

    stats = {
        'n_total': Ta,
        'n_solvable': n_solvable,
        'n_both_ground': both_ground.sum().item(),
        'n_both_float': n_both_float,
        'avg_F_BA': F_BA_mag[solvable.cpu().numpy()].mean() if n_solvable > 0 else 0.0,
        'avg_F_AB': F_AB_mag[solvable.cpu().numpy()].mean() if n_solvable > 0 else 0.0,
        'avg_newton3_err': n3_err[both_float.cpu().numpy()].mean() if n_both_float > 0 else 0.0,
        'max_newton3_err': n3_err[both_float.cpu().numpy()].max() if n_both_float > 0 else 0.0,
    }
    return stats


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

    target_indices = [1000, 2000]

    header = (f"{'Seq':<6} | {'Solvable':<10} | {'BothGnd':<10} | {'BothFlt':<10} | "
              f"{'Avg|F_BA| (N)':<15} | {'Avg|F_AB| (N)':<15} | {'Mean N3 Err':<12} | {'Max N3 Err':<12}")
    print(f"\n{header}")
    print("-" * len(header))

    for idx in target_indices:
        if idx < len(dataset):
            item = dataset[idx]

            betas_A = item['betas_p1'].unsqueeze(0).float()
            betas_B = item['betas_p2'].unsqueeze(0).float()
            mass_A = estimate_mass_from_betas(smpl_model, betas_A)
            mass_B = estimate_mass_from_betas(smpl_model, betas_B)

            m1 = dataset.normalizer.backward(item['motion_p1'].unsqueeze(0)).squeeze(0)
            m2 = dataset.normalizer.backward(item['motion_p2'].unsqueeze(0)).squeeze(0)

            s = analyze_interaction_forces(m1, m2, mass_A, mass_B, idx, save_dir)

            print(f"{idx:<6} | {s['n_solvable']:>4}/{s['n_total']:<4} | "
                  f"{s['n_both_ground']:>4}/{s['n_total']:<4} | "
                  f"{s['n_both_float']:>4}/{s['n_total']:<4} | "
                  f"{s['avg_F_BA']:<15.1f} | {s['avg_F_AB']:<15.1f} | "
                  f"{s['avg_newton3_err']:<12.2f} | {s['max_newton3_err']:<12.2f}")
        else:
            print(f"Index {idx} is out of bounds.")

if __name__ == "__main__":
    main()
