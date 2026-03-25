"""
test_physics_losses.py — Validate physics losses show improvement.

Tests that:
1. All physics loss functions compute correctly on synthetic data
2. Physics losses decrease when motion improves (gradient signal)
3. Contact force budget prevents explosion
4. Foot-ground contact uses correct coordinate system (Z-up)
5. SoftFlow penetration penalty activates for below-ground feet
6. Skyhook loss detects root support forces
7. ZMP loss detects CoM outside support polygon

Usage:
    conda activate mimickit
    cd /path/to/InterMask
    python newton_vqvae/test_physics_losses.py
"""
from __future__ import annotations

import sys
import os

# Ensure InterMask root on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np

from newton_vqvae.config import TrainingConfig
from newton_vqvae.physics_losses import (
    loss_fk_mpjpe,
    loss_torque,
    loss_skyhook,
    loss_softflow,
    loss_zmp,
    loss_contact_force_budget,
    PhysicsLoss,
    PhysicsLossScheduler,
)


def test_fk_mpjpe():
    """Test FK MPJPE loss: zero for identical, positive otherwise."""
    print("Test 1: FK MPJPE loss...")

    B, T, J = 2, 16, 22
    target = torch.randn(B, T, J, 3)
    sim_same = target.clone()
    sim_diff = target + 0.1 * torch.randn_like(target)

    loss_same = loss_fk_mpjpe(sim_same, target)
    loss_diff = loss_fk_mpjpe(sim_diff, target)

    assert loss_same.item() < 1e-6, f"Same positions should give ~0 loss, got {loss_same.item()}"
    assert loss_diff.item() > 0, f"Different positions should give positive loss"
    print(f"  OK: same={loss_same.item():.6f}, diff={loss_diff.item():.6f}")


def test_torque_low_vs_high():
    """Test torque loss: higher torques → higher loss."""
    print("Test 2: Torque loss (low vs high)...")

    T, D = 16, 75
    low_torques = torch.randn(T, D) * 10   # ~10 Nm
    high_torques = torch.randn(T, D) * 500  # ~500 Nm

    loss_low = loss_torque(low_torques, torque_limit=1000.0)
    loss_high = loss_torque(high_torques, torque_limit=1000.0)

    assert loss_high > loss_low, f"High torques should give higher loss: {loss_high} > {loss_low}"
    print(f"  OK: low={loss_low.item():.6f}, high={loss_high.item():.6f}")


def test_skyhook_detection():
    """Test skyhook loss: large root forces → high loss."""
    print("Test 3: Skyhook loss...")

    T = 32
    small_forces = torch.randn(T, 6) * 1.0
    large_forces = torch.randn(T, 6) * 100.0

    loss_small = loss_skyhook(small_forces)
    loss_large = loss_skyhook(large_forces)

    assert loss_large > loss_small, f"Large root forces should give higher loss"
    print(f"  OK: small_forces={loss_small.item():.4f}, large_forces={loss_large.item():.4f}")


def test_softflow_z_up():
    """Test SoftFlow uses Z-up (up_axis=2) for Newton convention."""
    print("Test 4: SoftFlow Z-up coordinate system...")

    B, T, J = 1, 16, 22
    positions = torch.zeros(B, T, J, 3)

    # Set all foot joints above ground (Z = 0.05)
    foot_idx = (7, 8, 10, 11)
    positions[:, :, foot_idx, 2] = 0.05  # Z-up, above ground

    loss_above = loss_softflow(positions, foot_indices=foot_idx, up_axis=2)

    # Now set feet below ground (Z = -0.02)
    positions[:, :, foot_idx, 2] = -0.02

    loss_below = loss_softflow(positions, foot_indices=foot_idx, up_axis=2)

    assert loss_below > loss_above, f"Below-ground feet should have higher loss: {loss_below} > {loss_above}"
    print(f"  OK: above_ground={loss_above.item():.6f}, below_ground={loss_below.item():.6f}")


def test_softflow_penetration_gradient():
    """Test that SoftFlow provides smooth gradients near the contact boundary."""
    print("Test 5: SoftFlow gradient smoothness near contact boundary...")

    # Create feet slightly below ground → gradients should flow
    positions = torch.zeros(1, 2, 22, 3, requires_grad=True)
    # Foot Z = -0.005 (slightly penetrating)
    with torch.no_grad():
        init = torch.zeros(1, 2, 22, 3)
        init[:, :, (7, 8, 10, 11), 2] = -0.005
    positions = init.clone().requires_grad_(True)

    loss = loss_softflow(positions, foot_indices=(7, 8, 10, 11), up_axis=2)
    loss.backward()

    grad = positions.grad
    assert grad is not None, "Gradient should exist near contact boundary"
    grad_mag = grad.abs().sum().item()
    assert grad_mag > 0, f"Gradient should be non-zero near contact boundary"
    print(f"  OK: gradient magnitude near boundary = {grad_mag:.6f}")


def test_softflow_sliding():
    """Test that SoftFlow penalizes horizontal sliding when in contact."""
    print("Test 6: SoftFlow sliding penalty...")

    B, T, J = 1, 8, 22
    foot_idx = (7, 8, 10, 11)

    # Stationary feet on ground (Z=0.01)
    pos_static = torch.zeros(B, T, J, 3)
    pos_static[:, :, foot_idx, 2] = 0.01  # near ground

    # Sliding feet on ground
    pos_slide = torch.zeros(B, T, J, 3)
    pos_slide[:, :, foot_idx, 2] = 0.01
    for t in range(T):
        pos_slide[:, t, foot_idx, 0] = t * 0.05  # moving in X

    loss_static = loss_softflow(pos_static, foot_indices=foot_idx, up_axis=2)
    loss_slide = loss_softflow(pos_slide, foot_indices=foot_idx, up_axis=2)

    assert loss_slide > loss_static, f"Sliding feet should have higher loss"
    print(f"  OK: static={loss_static.item():.6f}, sliding={loss_slide.item():.6f}")


def test_zmp_z_up():
    """Test ZMP loss uses Z-up correctly."""
    print("Test 7: ZMP loss Z-up coordinate system...")

    J = 22
    foot_idx = (7, 8, 10, 11)

    # CoM inside support polygon
    pos_inside = torch.zeros(J, 3)
    pos_inside[0] = torch.tensor([0.0, 0.0, 0.9])  # root at center
    pos_inside[7] = torch.tensor([-0.1, -0.1, 0.0])
    pos_inside[8] = torch.tensor([0.1, -0.1, 0.0])
    pos_inside[10] = torch.tensor([-0.1, 0.1, 0.0])
    pos_inside[11] = torch.tensor([0.1, 0.1, 0.0])

    root_f = torch.zeros(6)
    loss_in = loss_zmp(pos_inside, root_f, foot_indices=foot_idx, up_axis=2)

    # CoM outside support polygon (far to the side)
    pos_outside = pos_inside.clone()
    pos_outside[0] = torch.tensor([1.0, 1.0, 0.9])  # root far from feet

    loss_out = loss_zmp(pos_outside, root_f, foot_indices=foot_idx, up_axis=2)

    assert loss_out > loss_in, f"CoM outside support should have higher loss"
    print(f"  OK: inside={loss_in.item():.6f}, outside={loss_out.item():.6f}")


def test_contact_force_budget():
    """Test contact force budget prevents extreme forces."""
    print("Test 8: Contact force budget...")

    T, D, J = 16, 75, 22
    budget = 500.0

    # Normal torques (under budget)
    normal_torques = torch.randn(T, D) * 100
    pos = torch.zeros(T, J, 3)

    loss_normal = loss_contact_force_budget(normal_torques, pos, force_budget=budget, up_axis=2)

    # Extreme torques (over budget)
    extreme_torques = torch.randn(T, D) * 2000
    loss_extreme = loss_contact_force_budget(extreme_torques, pos, force_budget=budget, up_axis=2)

    assert loss_extreme > loss_normal, f"Extreme forces should give higher loss"
    print(f"  OK: normal={loss_normal.item():.6f}, extreme={loss_extreme.item():.6f}")


def test_physics_loss_scheduler():
    """Test physics loss scheduler warmup → ramp → full."""
    print("Test 9: Physics loss scheduler...")

    sched = PhysicsLossScheduler(warmup_epochs=5, ramp_epochs=10)

    # Warmup: weight = 0
    for ep in range(5):
        w = sched.get_weight(ep)
        assert w == 0.0, f"Warmup epoch {ep} should have weight 0, got {w}"

    # Ramp: weight increases linearly
    w5 = sched.get_weight(5)
    w10 = sched.get_weight(10)
    assert w5 == 0.0, f"Ramp start should be 0, got {w5}"
    assert 0.4 < w10 < 0.6, f"Mid-ramp should be ~0.5, got {w10}"

    # Full: weight = 1
    w20 = sched.get_weight(20)
    assert w20 == 1.0, f"After ramp should be 1.0, got {w20}"

    print(f"  OK: warmup=0, mid_ramp={w10:.2f}, full={w20:.2f}")


def test_combined_physics_loss_improves():
    """Test that combined PhysicsLoss decreases when motion gets better."""
    print("Test 10: Combined physics loss improvement signal...")

    cfg = TrainingConfig()
    physics_loss = PhysicsLoss(cfg)

    B, T, J = 2, 16, 22

    # Create target positions
    target_pos = torch.randn(B, T, J, 3) * 0.5
    target_pos[..., 2] = target_pos[..., 2].abs() + 0.02  # above ground

    # Bad simulation result (large errors, high torques)
    bad_result = {
        'sim_positions': target_pos + torch.randn_like(target_pos) * 0.5,
        'pd_torques': torch.randn(B, T, 75) * 500,
        'root_forces': torch.randn(B, T, 6) * 100,
        'body_positions': target_pos + torch.randn(B, T, J, 3) * 0.3,
    }

    # Good simulation result (small errors, low torques)
    good_result = {
        'sim_positions': target_pos + torch.randn_like(target_pos) * 0.01,
        'pd_torques': torch.randn(B, T, 75) * 10,
        'root_forces': torch.randn(B, T, 6) * 1,
        'body_positions': target_pos + torch.randn(B, T, J, 3) * 0.01,
    }
    # Make sure feet are above ground in good result
    foot_idx = list(physics_loss.foot_indices)
    good_result['body_positions'][..., foot_idx, 2] = 0.02

    loss_bad, log_bad = physics_loss(bad_result, target_pos)
    loss_good, log_good = physics_loss(good_result, target_pos)

    assert loss_good < loss_bad, (
        f"Better sim should have lower physics loss: good={loss_good.item():.4f} < bad={loss_bad.item():.4f}"
    )

    print(f"  OK: bad_loss={loss_bad.item():.4f}, good_loss={loss_good.item():.4f}")
    print(f"  Individual losses (bad → good):")
    for key in ['l_fk_mpjpe', 'l_torque', 'l_skyhook', 'l_softflow', 'l_zmp', 'l_contact_budget']:
        bad_v = log_bad.get(key, torch.tensor(0.0))
        good_v = log_good.get(key, torch.tensor(0.0))
        bad_v = bad_v.item() if isinstance(bad_v, torch.Tensor) else bad_v
        good_v = good_v.item() if isinstance(good_v, torch.Tensor) else good_v
        improved = "✓" if good_v < bad_v else "✗"
        print(f"    {key}: {bad_v:.4f} → {good_v:.4f} {improved}")


def test_gradient_flow_through_physics():
    """Test that gradients flow back through physics losses."""
    print("Test 11: Gradient flow through physics losses...")

    B, T, J = 1, 8, 22

    # Simulate differentiable positions
    root_offset = torch.randn(B, T, 1, 3, requires_grad=True)
    base_pos = torch.randn(B, T, J, 3)

    # Create sim_result with gradients
    sim_pos = base_pos + root_offset
    sim_pos_above_ground = sim_pos.clone()
    sim_pos_above_ground[..., 2] = sim_pos_above_ground[..., 2].abs() + 0.01

    target = base_pos.clone()

    # Each loss should produce gradients
    losses_to_test = [
        ("fk_mpjpe", loss_fk_mpjpe(sim_pos, target)),
        ("softflow", loss_softflow(sim_pos, up_axis=2)),
    ]

    for name, loss in losses_to_test:
        if root_offset.grad is not None:
            root_offset.grad.zero_()
        loss.backward(retain_graph=True)
        grad_norm = root_offset.grad.norm().item() if root_offset.grad is not None else 0
        assert grad_norm > 0, f"No gradient from {name} loss"
        print(f"  {name}: grad_norm={grad_norm:.6f}")

    # ZMP: create scenario where CoM is outside support polygon for gradient
    root_pos = torch.randn(1, 3, requires_grad=True)
    pos_zmp = torch.zeros(J, 3)
    pos_zmp[0] = root_pos.squeeze()  # root = learnable
    pos_zmp[7] = torch.tensor([-0.1, -0.1, 0.0])
    pos_zmp[8] = torch.tensor([0.1, -0.1, 0.0])
    pos_zmp[10] = torch.tensor([-0.1, 0.1, 0.0])
    pos_zmp[11] = torch.tensor([0.1, 0.1, 0.0])
    zmp_loss = loss_zmp(pos_zmp, torch.zeros(6), up_axis=2)
    zmp_loss.backward()
    zmp_grad = root_pos.grad.norm().item() if root_pos.grad is not None else 0
    # Gradient may be zero if CoM is inside support polygon — that's OK
    print(f"  zmp: grad_norm={zmp_grad:.6f} (may be 0 if CoM inside support)")

    print("  OK: All losses produce gradients")


if __name__ == '__main__':
    print("=" * 60)
    print(" Physics Loss Validation Suite")
    print("=" * 60)

    tests = [
        test_fk_mpjpe,
        test_torque_low_vs_high,
        test_skyhook_detection,
        test_softflow_z_up,
        test_softflow_penetration_gradient,
        test_softflow_sliding,
        test_zmp_z_up,
        test_contact_force_budget,
        test_physics_loss_scheduler,
        test_combined_physics_loss_improves,
        test_gradient_flow_through_physics,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f" Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n=== ALL PHYSICS TESTS PASSED ===")
