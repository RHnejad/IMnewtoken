import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.data_adapter import InterHumanPairDataset
from newton_vqvae.physics_losses import SMPL_SEGMENT_MASS_RATIOS

# ═══════════════════════════════════════════════════════════════
# Physics Constants & Helpers
# ═══════════════════════════════════════════════════════════════
UP_AXIS = 1
GRAVITY = 9.81
FPS = 30.0
TOTAL_MASS = 75.0
# Foot contact flags are stored in motion channels 258:262
# (pre-computed during data processing using velocity + height criteria)

# ── Motion vector layout (per person, 262 dims total) ──
# [0:66]    = joint positions  (22 joints × 3)
# [66:132]  = joint velocities (22 joints × 3)
# [132:258] = joint rotations  (21 joints × 6, continuous 6d)
# [258:262] = foot contacts    (4 binary flags)
POS_SLICE = slice(0, 66)

# Initialize segment mass tensor
seg_mass = torch.tensor(SMPL_SEGMENT_MASS_RATIOS, dtype=torch.float32)
seg_mass = seg_mass / seg_mass.sum()  # shape (22,)


def get_com(positions: torch.Tensor) -> torch.Tensor:
    """Compute Center of Mass.  positions: (T, 22, 3) -> (T, 3)"""
    w = seg_mass.to(positions.device).view(1, -1, 1)
    return (positions * w).sum(dim=1)


def finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    return (x[1:] - x[:-1]) / dt


def detect_ground_contact(motion: torch.Tensor) -> torch.Tensor:
    """
    Determine whether a person has at least one foot on the ground,
    using pre-computed foot contact flags from the motion vector.

    Channels 258:262 were computed during data processing using both
    velocity and height criteria (height < [0.12, 0.05] m for
    [ankle, toe] AND velocity < threshold), making them more robust
    than a simple height threshold.

    Args:
        motion: (T, 262) per-person motion vector

    Returns:
        grounded: (T,) bool — True if any foot contact flag is active
    """
    foot_contacts = motion[:, 258:262]  # (T, 4) pre-computed contact flags
    return (foot_contacts > 0.5).any(dim=-1)  # (T,) bool


# ═══════════════════════════════════════════════════════════════
# Contact-state-aware dyadic physics analysis
# ═══════════════════════════════════════════════════════════════

def analyze_pair(motion_A: torch.Tensor, motion_B: torch.Tensor):
    """
    Analyse a pair of interacting people using proper contact-aware dynamics.

    Newton's 2nd law for each person:
        m_A · a_A = F_ground_A + m_A · g + F_{B→A}
        m_B · a_B = F_ground_B + m_B · g + F_{A→B}
    Newton's 3rd law:  F_{A→B} = −F_{B→A}

    The equations are solvable depending on each person's contact state:
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ Contact state    │ What we can compute                           │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ A floating only  │ F_{B→A} = m_A(a_A − g)  (F_ground_A = 0)     │
    │                  │ F_{A→B} = −F_{B→A}  (3rd law)                 │
    │                  │ F_ground_B = m_B·a_B − m_B·g − F_{A→B}       │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ B floating only  │ F_{A→B} = m_B(a_B − g)  (F_ground_B = 0)     │
    │                  │ F_{B→A} = −F_{A→B}                            │
    │                  │ F_ground_A = m_A·a_A − m_A·g − F_{B→A}       │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ Both floating    │ From A: F_{B→A} = m_A(a_A − g)               │
    │                  │ From B: F_{A→B} = m_B(a_B − g)               │
    │                  │ 3rd law CHECK: F_{B→A} + F_{A→B} ≈ 0?        │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ Both grounded    │ Under-determined (9 unknowns, 6 equations).   │
    │                  │ Can only compute total system ground force:    │
    │                  │ F_gnd_total = (m_A+m_B)·a_sys − (m_A+m_B)·g  │
    └──────────────────┴────────────────────────────────────────────────┘

    Returns dict with per-frame results for each contact regime.
    """
    T = motion_A.shape[0]
    device = motion_A.device
    dt = 1.0 / FPS
    g_vec = torch.zeros(3, device=device)
    g_vec[UP_AXIS] = -GRAVITY

    # ── Positions & COM ──
    pos_A = motion_A[:, POS_SLICE].reshape(T, 22, 3)
    pos_B = motion_B[:, POS_SLICE].reshape(T, 22, 3)

    com_A = get_com(pos_A)  # (T, 3)
    com_B = get_com(pos_B)
    com_sys = 0.5 * (com_A + com_B)  # equal mass assumption

    # ── Accelerations  (T-2, 3) ──
    acc_A = finite_diff(finite_diff(com_A, dt), dt)
    acc_B = finite_diff(finite_diff(com_B, dt), dt)
    acc_sys = finite_diff(finite_diff(com_sys, dt), dt)
    Ta = acc_A.shape[0]

    # ── Ground contact  (trim to match accel frames) ──
    contact_A = detect_ground_contact(motion_A)[1:-1]  # (Ta,) bool
    contact_B = detect_ground_contact(motion_B)[1:-1]

    A_float = ~contact_A
    B_float = ~contact_B

    both_float = A_float & B_float
    A_float_only = A_float & contact_B
    B_float_only = contact_A & B_float
    both_ground = contact_A & contact_B

    # ── Interaction forces (only where solvable) ──
    F_B_on_A = torch.full((Ta, 3), float('nan'), device=device)
    F_A_on_B = torch.full((Ta, 3), float('nan'), device=device)
    newton3_err = torch.zeros(Ta, device=device)

    # Case: A floating (A_float_only OR both_float)
    mask_A_fl = A_float  # includes both_float
    if mask_A_fl.any():
        F_B_on_A[mask_A_fl] = TOTAL_MASS * (acc_A[mask_A_fl] - g_vec)

    # Case: B floating (B_float_only OR both_float)
    mask_B_fl = B_float
    if mask_B_fl.any():
        F_A_on_B[mask_B_fl] = TOTAL_MASS * (acc_B[mask_B_fl] - g_vec)

    # Use Newton's 3rd law to fill the other force where only one is floating
    if A_float_only.any():
        F_A_on_B[A_float_only] = -F_B_on_A[A_float_only]
    if B_float_only.any():
        F_B_on_A[B_float_only] = -F_A_on_B[B_float_only]

    # Both-floating: check Newton's 3rd law consistency
    if both_float.any():
        newton3_err[both_float] = (F_B_on_A[both_float] + F_A_on_B[both_float]).norm(dim=-1)

    # ── Ground reaction forces (where solvable) ──
    F_ground_A = torch.full((Ta, 3), float('nan'), device=device)
    F_ground_B = torch.full((Ta, 3), float('nan'), device=device)

    # When A is floating → F_ground_A = 0
    F_ground_A[A_float] = 0.0
    # When B is floating → F_ground_B = 0
    F_ground_B[B_float] = 0.0

    # When A grounded & B floating → F_ground_A = m_A·a_A − m_A·g − F_{B→A}
    if B_float_only.any():
        F_ground_A[B_float_only] = (
            TOTAL_MASS * acc_A[B_float_only]
            - TOTAL_MASS * g_vec
            - F_B_on_A[B_float_only]
        )
    # When B grounded & A floating → F_ground_B = m_B·a_B − m_B·g − F_{A→B}
    if A_float_only.any():
        F_ground_B[A_float_only] = (
            TOTAL_MASS * acc_B[A_float_only]
            - TOTAL_MASS * g_vec
            - F_A_on_B[A_float_only]
        )

    # ── System-level skyhook (always computable) ──
    # Total upward force the ground must provide to the system:
    #   F_gnd_up = (m_A+m_B)(a_sys_y + g)
    # If this is negative → system accelerates down faster than gravity → violation
    sys_required_up = 2 * TOTAL_MASS * (acc_sys[:, UP_AXIS] + GRAVITY)

    # System skyhook = needed upward force when BOTH floating (no ground available)
    sys_floating = both_float.float()
    skyhook_sys = F.relu(sys_required_up) * sys_floating

    # Individual skyhook = needed upward force when that person is floating
    indiv_up_A = TOTAL_MASS * (acc_A[:, UP_AXIS] + GRAVITY)
    indiv_up_B = TOTAL_MASS * (acc_B[:, UP_AXIS] + GRAVITY)
    skyhook_A = F.relu(indiv_up_A) * A_float.float()
    skyhook_B = F.relu(indiv_up_B) * B_float.float()

    # ── Ground-force sanity (vertical component should be ≥ 0) ──
    grf_violation_A = torch.zeros(Ta, device=device)
    grf_violation_B = torch.zeros(Ta, device=device)
    # Only check where we could solve for GRF
    solvable_A = ~torch.isnan(F_ground_A[:, 0])
    solvable_B = ~torch.isnan(F_ground_B[:, 0])
    if solvable_A.any():
        grf_violation_A[solvable_A] = F.relu(-F_ground_A[solvable_A, UP_AXIS])
    if solvable_B.any():
        grf_violation_B[solvable_B] = F.relu(-F_ground_B[solvable_B, UP_AXIS])

    return {
        # Contact state counts
        'n_both_ground': both_ground.sum().item(),
        'n_A_float_only': A_float_only.sum().item(),
        'n_B_float_only': B_float_only.sum().item(),
        'n_both_float': both_float.sum().item(),
        'n_total': Ta,
        # Forces
        'F_B_on_A': F_B_on_A, 'F_A_on_B': F_A_on_B,
        'F_ground_A': F_ground_A, 'F_ground_B': F_ground_B,
        'newton3_err': newton3_err,
        # Skyhook
        'skyhook_A': skyhook_A, 'skyhook_B': skyhook_B, 'skyhook_sys': skyhook_sys,
        # GRF violations
        'grf_violation_A': grf_violation_A, 'grf_violation_B': grf_violation_B,
        # Accelerations
        'acc_A': acc_A, 'acc_B': acc_B, 'acc_sys': acc_sys,
    }


# ═══════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading InterHuman pair dataset...")
    dataset = InterHumanPairDataset(
        data_root=os.path.join(_PROJECT_ROOT, "data/InterHuman"),
        mode='train',
        window_size=64,
        window_stride=64,
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch = next(iter(loader))

    normalizer = dataset.normalizer
    motions_p1 = normalizer.backward(batch['motion_p1'])
    motions_p2 = normalizer.backward(batch['motion_p2'])

    B = motions_p1.shape[0]

    # Accumulators
    totals = {k: 0.0 for k in [
        'n_both_ground', 'n_A_float_only', 'n_B_float_only', 'n_both_float', 'n_total',
        'sky_A', 'sky_B', 'sky_sys',
        'newton3', 'grf_viol_A', 'grf_viol_B',
        'acc_A_horiz', 'acc_sys_horiz',
    ]}

    for i in range(B):
        r = analyze_pair(motions_p1[i], motions_p2[i])

        for k in ['n_both_ground', 'n_A_float_only', 'n_B_float_only', 'n_both_float', 'n_total']:
            totals[k] += r[k]

        totals['sky_A'] += r['skyhook_A'].mean().item()
        totals['sky_B'] += r['skyhook_B'].mean().item()
        totals['sky_sys'] += r['skyhook_sys'].mean().item()
        totals['newton3'] += r['newton3_err'].mean().item()
        totals['grf_viol_A'] += r['grf_violation_A'].mean().item()
        totals['grf_viol_B'] += r['grf_violation_B'].mean().item()
        totals['acc_A_horiz'] += r['acc_A'][:, [0, 2]].norm(dim=-1).mean().item()
        totals['acc_sys_horiz'] += r['acc_sys'][:, [0, 2]].norm(dim=-1).mean().item()

    n = totals['n_total']
    print(f"\n{'='*65}")
    print(f"  Dyadic Physics Analysis  ({B} sequences, 64 frames each)")
    print(f"{'='*65}")

    print(f"\n── Contact State Distribution ──")
    print(f"  Both grounded    : {totals['n_both_ground']/n*100:5.1f}%  ({int(totals['n_both_ground'])} frames)")
    print(f"  A floating only  : {totals['n_A_float_only']/n*100:5.1f}%  ({int(totals['n_A_float_only'])} frames)")
    print(f"  B floating only  : {totals['n_B_float_only']/n*100:5.1f}%  ({int(totals['n_B_float_only'])} frames)")
    print(f"  Both floating    : {totals['n_both_float']/n*100:5.1f}%  ({int(totals['n_both_float'])} frames)")

    print(f"\n── Skyhook Force (upward force needed while floating) ──")
    print(f"  Person A (indiv) : {totals['sky_A']/B:8.2f} N  (mean across sequences)")
    print(f"  Person B (indiv) : {totals['sky_B']/B:8.2f} N")
    print(f"  System (A+B)     : {totals['sky_sys']/B:8.2f} N  (only when BOTH floating)")

    print(f"\n── Newton's 3rd Law Violation (both floating frames) ──")
    print(f"  Mean |F_{{B→A}} + F_{{A→B}}| : {totals['newton3']/B:8.2f} N")

    print(f"\n── Ground Reaction Force Violations ──")
    print(f"  (negative vertical GRF = ground pulling down = impossible)")
    print(f"  Person A : {totals['grf_viol_A']/B:8.2f} N")
    print(f"  Person B : {totals['grf_viol_B']/B:8.2f} N")

    print(f"\n── Horizontal COM Acceleration ──")
    print(f"  Person A (indiv) : {totals['acc_A_horiz']/B:8.2f} m/s²")
    print(f"  System (A+B)     : {totals['acc_sys_horiz']/B:8.2f} m/s²")

    print(f"\n{'='*65}")
    print("Key: Interaction forces are only solvable when ≥1 person is floating.")
    print("When both grounded, we have 9 unknowns / 6 equations → under-determined.")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
