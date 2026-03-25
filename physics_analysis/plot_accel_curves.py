import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.data_adapter import InterHumanPairDataset
from newton_vqvae.physics_losses import SMPL_SEGMENT_MASS_RATIOS

UP_AXIS = 1
FPS = 30.0

# ── Motion vector layout (per person, 262 dims total) ──
# [0:66]    = joint positions  (22 joints × 3)
# [66:132]  = joint velocities (22 joints × 3)
# [132:258] = joint rotations  (21 joints × 6, continuous 6d)
# [258:262] = foot contacts    (4 binary flags)
POS_SLICE = slice(0, 66)

seg_mass = torch.tensor(SMPL_SEGMENT_MASS_RATIOS, dtype=torch.float32)
seg_mass = seg_mass / seg_mass.sum()  # shape (22,)

def get_com(positions: torch.Tensor) -> torch.Tensor:
    """Compute Center of Mass. positions: (T, 22, 3) -> (T, 3)"""
    w = seg_mass.to(positions.device).view(1, -1, 1)
    return (positions * w).sum(dim=1)

def finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    return (x[1:] - x[:-1]) / dt

def analyze_and_plot(motion_A: torch.Tensor, motion_B: torch.Tensor,
                     seq_idx: int, save_dir: str):
    """
    Plot acceleration curves for two interacting people.

    Args:
        motion_A: (T, 262) single-person motion for Person A
        motion_B: (T, 262) single-person motion for Person B

    Each 262-dim vector (per person) contains:
        [0:66]    joint positions  (22×3)
        [66:132]  joint velocities (22×3)
        [132:258] joint rotations  (21×6)
        [258:262] foot contacts    (4)
    """
    T = motion_A.shape[0]
    dt = 1.0 / FPS

    # Extract positions from each person's own 262-dim vector
    pos_A = motion_A[:, POS_SLICE].reshape(T, 22, 3)
    pos_B = motion_B[:, POS_SLICE].reshape(T, 22, 3)

    # Combine into system (44 joints)
    pos_sys = torch.cat([pos_A, pos_B], dim=1)  # (T, 44, 3)
    sys_mass = torch.cat([seg_mass * 0.5, seg_mass * 0.5], dim=0).to(motion_A.device)

    # COMs
    com_A = get_com(pos_A)
    com_B = get_com(pos_B)
    com_sys = (pos_sys * sys_mass.view(1, -1, 1)).sum(dim=1)

    # Accelerations (T-2, 3)
    acc_A = finite_diff(finite_diff(com_A, dt), dt)
    acc_B = finite_diff(finite_diff(com_B, dt), dt)
    acc_sys = finite_diff(finite_diff(com_sys, dt), dt)

    # Get magnitude of acceleration
    acc_A_mag = acc_A.norm(dim=-1).cpu().numpy()
    acc_B_mag = acc_B.norm(dim=-1).cpu().numpy()
    acc_sys_mag = acc_sys.norm(dim=-1).cpu().numpy()

    # Time axis
    time_axis = np.arange(acc_A.shape[0]) * dt

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, acc_A_mag, label='Person A (Individual)', color='blue', alpha=0.7)
    plt.plot(time_axis, acc_B_mag, label='Person B (Individual)', color='red', alpha=0.7)
    plt.plot(time_axis, acc_sys_mag, label='Combined System (A+B)', color='black', linewidth=2.5)

    # Gravity reference line
    plt.axhline(y=9.81, color='g', linestyle='--', label='1g (9.81 m/s²)', alpha=0.5)

    plt.title(f'Center of Mass Acceleration Norm - Sequence {seq_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration Magnitude (m/s²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'accel_curve_seq_{seq_idx}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")

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

    # Indices to plot
    target_indices = [1000, 2000]

    for idx in target_indices:
        if idx < len(dataset):
            # Get item — returns separate per-person motions
            item = dataset[idx]
            m1_norm = item['motion_p1']  # (T, 262)
            m2_norm = item['motion_p2']  # (T, 262)

            # Unnormalize each person's motion independently
            m1 = dataset.normalizer.backward(m1_norm.unsqueeze(0)).squeeze(0)
            m2 = dataset.normalizer.backward(m2_norm.unsqueeze(0)).squeeze(0)

            analyze_and_plot(m1, m2, idx, save_dir)
        else:
            print(f"Index {idx} is out of bounds (dataset size: {len(dataset)})")

if __name__ == "__main__":
    main()
