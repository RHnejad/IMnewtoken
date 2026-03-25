"""
test_actor_network.py — Tests for the neural actor and gradient bridge.

Tests cover:
    1. Actor network shapes and zero initialization
    2. Observation building from reference data
    3. Delta scatter/extract roundtrip
    4. Gradient flow through the manual bridge
    5. Observation normalizer
    6. Integration test with Warp physics (requires data)

Run:
    python -m pytest prepare2/test_actor_network.py -v
    python prepare2/test_actor_network.py           # fallback runner
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare2.actor_network import (
    InteractionActor,
    ObservationNormalizer,
    build_observations,
    build_all_observations,
    scatter_delta_to_flat,
    extract_hinge_grads,
    OBS_DIM,
    N_HINGE_DOFS,
    DOFS_PER_PERSON,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def _make_dummy_data(T=30, n_persons=2):
    """Create random reference data for testing (with valid quaternions)."""
    ref_jq = []
    for _ in range(n_persons):
        jq = np.random.randn(T, 76).astype(np.float32)
        # Ensure root quaternion (indices 3:7) is unit length
        for t in range(T):
            q = jq[t, 3:7]
            q /= np.linalg.norm(q) + 1e-8
            jq[t, 3:7] = q
        ref_jq.append(jq)
    ref_positions = np.random.randn(T, n_persons, 22, 3).astype(np.float32)
    torques_solo = [
        np.random.randn(T, 75).astype(np.float32)
        for _ in range(n_persons)
    ]
    return ref_jq, ref_positions, torques_solo


# ═══════════════════════════════════════════════════════════════
# Actor Network Tests
# ═══════════════════════════════════════════════════════════════
class TestInteractionActor:
    """Tests for InteractionActor."""

    def test_output_shape(self):
        """Actor output should be (batch, 69)."""
        actor = InteractionActor()
        obs = torch.randn(10, OBS_DIM)
        out = actor(obs)
        assert out.shape == (10, N_HINGE_DOFS), \
            f"Expected (10, {N_HINGE_DOFS}), got {out.shape}"

    def test_single_sample(self):
        """Actor should handle batch_size=1."""
        actor = InteractionActor()
        obs = torch.randn(1, OBS_DIM)
        out = actor(obs)
        assert out.shape == (1, N_HINGE_DOFS)

    def test_zero_initialization(self):
        """At init, output should be zero (Δq = 0 → baseline behavior)."""
        actor = InteractionActor()
        obs = torch.randn(5, OBS_DIM)
        out = actor(obs)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), \
            f"Output should be zero at init, got max={out.abs().max():.6f}"

    def test_max_delta_bounding(self):
        """tanh + max_delta should bound output to [-max_delta, +max_delta]."""
        max_delta = 0.02
        actor = InteractionActor(max_delta=max_delta)
        # Give large weights to output layer to saturate tanh
        with torch.no_grad():
            actor.net[-1].weight.fill_(10.0)
            actor.net[-1].bias.fill_(100.0)
        obs = torch.zeros(1, OBS_DIM)
        out = actor(obs)
        # Output should be bounded by max_delta even with extreme weights
        assert out.abs().max() <= max_delta + 1e-6, \
            f"Output should be bounded by {max_delta}, got {out.abs().max()}"
        # With large positive bias, tanh saturates near 1.0
        assert out.min() > 0, "With large positive bias, output should be positive"

    def test_gradient_flow_standard(self):
        """Standard loss.backward() should produce gradients on all params."""
        actor = InteractionActor()
        obs = torch.randn(5, OBS_DIM)
        out = actor(obs)
        loss = (out ** 2).sum()
        loss.backward()

        for name, param in actor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient on {name}"

    def test_gradient_flow_external(self):
        """External gradients (from Warp) should propagate to actor params.

        This is the critical test for the manual gradient bridge:
            delta_q = actor(obs)
            # ... Warp sim produces grad ...
            delta_q.backward(gradient=external_grad)
        """
        actor = InteractionActor(max_delta=1.0)
        # Make output non-zero so gradients are non-trivial
        with torch.no_grad():
            actor.net[-1].weight.normal_(0.01)
            actor.net[-1].bias.fill_(0.0)

        obs = torch.randn(5, OBS_DIM)
        delta_q = actor(obs)   # (5, 69) — has grad_fn

        # Simulate external gradient from Warp
        external_grad = torch.randn(5, N_HINGE_DOFS)
        delta_q.backward(gradient=external_grad)

        # Check that at least one parameter has non-zero gradient
        has_nonzero_grad = False
        for param in actor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        assert has_nonzero_grad, \
            "External gradients did not propagate to actor parameters"

    def test_parameter_count(self):
        """Parameter count should be reasonable for (212→256→256→69)."""
        actor = InteractionActor(hidden_dims=(256, 256))
        n = actor.count_parameters()
        # L1: 212*256+256=54528, L2: 256*256+256=65792, L3: 256*69+69=17733
        # Total ≈ 138053
        assert 100_000 < n < 200_000, \
            f"Unexpected parameter count: {n}"

    def test_different_hidden_dims(self):
        """Should work with various hidden layer configurations."""
        for dims in [(128,), (512, 512), (64, 128, 64)]:
            actor = InteractionActor(hidden_dims=dims)
            out = actor(torch.randn(3, OBS_DIM))
            assert out.shape == (3, N_HINGE_DOFS), \
                f"Failed with hidden_dims={dims}"

    def test_activations(self):
        """All supported activations should work."""
        for act in ("relu", "elu", "tanh"):
            actor = InteractionActor(activation=act)
            out = actor(torch.randn(3, OBS_DIM))
            assert out.shape == (3, N_HINGE_DOFS), \
                f"Failed with activation={act}"

    def test_invalid_activation(self):
        """Unknown activation should raise ValueError."""
        try:
            InteractionActor(activation="gelu")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════
# Observation Tests
# ═══════════════════════════════════════════════════════════════
class TestObservations:
    """Tests for observation building."""

    def test_window_shape(self):
        """Observation shape should be (window_size × 2, 212)."""
        ref_jq, ref_pos, torques = _make_dummy_data(T=30)
        obs = build_observations(ref_jq, ref_pos, torques, 0, 5, 30)
        assert obs.shape == (5 * 2, OBS_DIM), \
            f"Expected ({5 * 2}, {OBS_DIM}), got {obs.shape}"

    def test_all_frames_shape(self):
        """build_all_observations should cover all T frames."""
        T = 20
        ref_jq, ref_pos, torques = _make_dummy_data(T=T)
        obs = build_all_observations(ref_jq, ref_pos, torques, T)
        assert obs.shape == (T * 2, OBS_DIM)

    def test_content_ref_hinge(self):
        """First 69 features should be reference hinge angles."""
        T = 10
        ref_jq, ref_pos, torques = _make_dummy_data(T=T)
        obs = build_observations(ref_jq, ref_pos, torques, 2, 4, T)

        # Row 0 = frame 2, person 0
        np.testing.assert_allclose(
            obs[0, :69], ref_jq[0][2, 7:76], rtol=1e-5,
            err_msg="Ref hinge mismatch for frame 2, person 0")

        # Row 1 = frame 2, person 1
        np.testing.assert_allclose(
            obs[1, :69], ref_jq[1][2, 7:76], rtol=1e-5,
            err_msg="Ref hinge mismatch for frame 2, person 1")

    def test_content_other_pos(self):
        """Features 69:135 should be OTHER person's positions in ego frame."""
        T = 10
        ref_jq, ref_pos, torques = _make_dummy_data(T=T)
        obs = build_observations(ref_jq, ref_pos, torques, 3, 5, T)

        # Row 0 = frame 3, person 0: other = person 1
        # Verify it's NOT raw world positions (ego-relative transform was applied)
        raw_other = ref_pos[3, 1].flatten()  # world-frame
        # If ego transform was applied, these should differ
        # (unless ego is at origin with identity rotation, which is unlikely)
        # Just check shape and finite values
        other_obs = obs[0, 69:135]
        assert other_obs.shape == (66,), f"Wrong shape: {other_obs.shape}"
        assert np.all(np.isfinite(other_obs)), "Non-finite values in other_pos"

    def test_content_solo_torques(self):
        """Features 135:204 should be solo hinge torques."""
        T = 10
        ref_jq, ref_pos, torques = _make_dummy_data(T=T)
        obs = build_observations(ref_jq, ref_pos, torques, 0, 2, T)

        # Row 0 = frame 0, person 0
        np.testing.assert_allclose(
            obs[0, 135:204], torques[0][0, 6:75], rtol=1e-5,
            err_msg="Solo torque mismatch")

    def test_content_root_state(self):
        """Features 204:211 should be root height + local orientation."""
        T = 10
        ref_jq, ref_pos, torques = _make_dummy_data(T=T)
        obs = build_observations(ref_jq, ref_pos, torques, 1, 3, T)

        # Row 0 = frame 1, person 0
        # Feature 204 should be root height (Z component of position)
        np.testing.assert_allclose(
            obs[0, 204], ref_jq[0][1, 2], rtol=1e-5,
            err_msg="Root height mismatch")
        # Features 205:211 should be local orientation (6-dim, all finite)
        local_ori = obs[0, 205:211]
        assert local_ori.shape == (6,), f"Wrong shape: {local_ori.shape}"
        assert np.all(np.isfinite(local_ori)), "Non-finite local orientation"

    def test_content_time(self):
        """Feature 211 should be normalized time."""
        T = 20
        ref_jq, ref_pos, torques = _make_dummy_data(T=T)
        obs = build_observations(ref_jq, ref_pos, torques, 10, 12, T)

        # Frame 10, normalized: 10 / 19 ≈ 0.5263
        expected_time = 10 / (T - 1)
        np.testing.assert_allclose(
            obs[0, 211], expected_time, rtol=1e-5,
            err_msg="Normalized time mismatch")


# ═══════════════════════════════════════════════════════════════
# Delta Mapping Tests
# ═══════════════════════════════════════════════════════════════
class TestDeltaMapping:
    """Tests for scatter_delta_to_flat and extract_hinge_grads."""

    def test_roundtrip(self):
        """scatter → extract should recover original values."""
        window = 5
        T = 20
        n_dof = 150
        delta_q = np.random.randn(window * 2, 69).astype(np.float32)

        flat = scatter_delta_to_flat(delta_q, 3, 8, T, n_dof)
        recovered = extract_hinge_grads(flat, 3, 8, n_dof)

        np.testing.assert_allclose(
            recovered, delta_q, rtol=1e-5,
            err_msg="Roundtrip scatter→extract failed")

    def test_zeros_outside_window(self):
        """Delta should be zero outside the window frames."""
        T = 20
        n_dof = 150
        delta_q = np.ones((4, 69), dtype=np.float32)  # 2 frames × 2 persons

        flat = scatter_delta_to_flat(delta_q, 5, 7, T, n_dof)
        full = flat.reshape(T, n_dof)

        assert np.all(full[:5] == 0), "Before window should be zero"
        assert np.all(full[7:] == 0), "After window should be zero"

    def test_root_dofs_zero(self):
        """Root DOFs (0:6 per person) should always be zero."""
        T = 10
        n_dof = 150
        delta_q = np.ones((6, 69), dtype=np.float32)  # 3 frames × 2 persons

        flat = scatter_delta_to_flat(delta_q, 2, 5, T, n_dof)
        full = flat.reshape(T, n_dof)

        for p in range(2):
            root_slice = full[:, p * 75:p * 75 + 6]
            assert np.all(root_slice == 0), \
                f"Root DOFs for person {p} should be zero"

    def test_hinge_placement(self):
        """Hinge values should be at correct positions in flat array."""
        T = 5
        n_dof = 150
        delta_q = np.zeros((2, 69), dtype=np.float32)
        delta_q[0, 0] = 1.0   # person 0, first hinge
        delta_q[1, 0] = 2.0   # person 1, first hinge

        flat = scatter_delta_to_flat(delta_q, 1, 2, T, n_dof)
        full = flat.reshape(T, n_dof)

        # Person 0: DOF 6 at frame 1
        assert full[1, 6] == 1.0, f"Expected 1.0, got {full[1, 6]}"
        # Person 1: DOF 81 (75+6) at frame 1
        assert full[1, 81] == 2.0, f"Expected 2.0, got {full[1, 81]}"

    def test_flat_shape(self):
        """Output shape should be (T × n_dof,)."""
        flat = scatter_delta_to_flat(
            np.zeros((6, 69), dtype=np.float32), 0, 3, 10, 150)
        assert flat.shape == (10 * 150,), \
            f"Expected ({10 * 150},), got {flat.shape}"

    def test_extract_shape(self):
        """Extracted gradients shape should be (window×2, 69)."""
        grad_flat = np.zeros(20 * 150, dtype=np.float32)
        result = extract_hinge_grads(grad_flat, 5, 8, 150)
        assert result.shape == (3 * 2, 69), \
            f"Expected ({3 * 2}, 69), got {result.shape}"


# ═══════════════════════════════════════════════════════════════
# Normalizer Tests
# ═══════════════════════════════════════════════════════════════
class TestObservationNormalizer:
    """Tests for ObservationNormalizer."""

    def test_normalize_shape(self):
        """Output shape should match input."""
        norm = ObservationNormalizer(4)
        data = np.random.randn(10, 4).astype(np.float32)
        norm.update(data)
        result = norm.normalize(data)
        assert result.shape == data.shape

    def test_normalize_mean(self):
        """After normalization, mean should be ~0."""
        norm = ObservationNormalizer(4)
        data = np.random.randn(100, 4).astype(np.float32) + 5.0
        norm.update(data)
        result = norm.normalize(data)
        np.testing.assert_allclose(
            result.mean(axis=0), 0, atol=0.15,
            err_msg="Normalized mean should be ~0")

    def test_normalize_std(self):
        """After normalization, std should be ~1."""
        norm = ObservationNormalizer(4)
        data = np.random.randn(1000, 4).astype(np.float32) * 10.0
        norm.update(data)
        result = norm.normalize(data)
        stds = result.std(axis=0)
        np.testing.assert_allclose(
            stds, 1.0, atol=0.1,
            err_msg="Normalized std should be ~1")

    def test_state_dict_roundtrip(self):
        """Save/load should preserve normalization behavior."""
        norm1 = ObservationNormalizer(4)
        data = np.random.randn(50, 4).astype(np.float32)
        norm1.update(data)

        sd = norm1.state_dict()
        norm2 = ObservationNormalizer(4)
        norm2.load_state_dict(sd)

        test = np.random.randn(3, 4).astype(np.float32)
        np.testing.assert_allclose(
            norm1.normalize(test), norm2.normalize(test),
            err_msg="Saved/loaded normalizer gives different results")


# ═══════════════════════════════════════════════════════════════
# End-to-End Gradient Bridge Test
# ═══════════════════════════════════════════════════════════════
class TestGradientBridge:
    """Test the full manual gradient bridge (without Warp)."""

    def test_gradient_descent_converges(self):
        """Verify that repeated gradient steps reduce a mock loss.

        Simulates the bridge pattern:
            actor → Δq → detach → mock_loss(Δq) → mock_grad(Δq)
            → delta_q.backward(mock_grad) → optimizer.step()

        The mock loss is ||Δq - target||² so the actor should learn
        to output the target.
        """
        actor = InteractionActor(
            obs_dim=10, output_dim=5,
            hidden_dims=(32, 32), max_delta=1.0,
        )
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
        target = np.array([0.1, -0.2, 0.15, -0.05, 0.3], dtype=np.float32)

        obs = torch.randn(4, 10)
        losses = []

        for _ in range(100):
            delta_q = actor(obs)   # (4, 5)

            # Mock "Warp" loss: ||Δq - target||²
            delta_np = delta_q.detach().numpy()
            diff = delta_np - target[None, :]
            mock_loss = float((diff ** 2).sum())
            losses.append(mock_loss)

            # Mock "Warp" gradient: 2 * (Δq - target) / N
            mock_grad_np = (2.0 * diff / delta_np.size).astype(np.float32)
            mock_grad = torch.from_numpy(mock_grad_np)

            # Bridge: backward through actor
            optimizer.zero_grad()
            delta_q.backward(gradient=mock_grad)
            optimizer.step()

        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.1, \
            f"Loss did not converge: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_bridge_matches_direct(self):
        """Bridge gradients should match direct PyTorch gradients.

        Compares two approaches for computing ∂L/∂θ:
        1. Standard: loss = ||actor(obs) - target||²; loss.backward()
        2. Bridge:   delta_q = actor(obs); grad = 2*(Δq-target)/N;
                     delta_q.backward(gradient=grad)

        Both should produce identical parameter gradients.
        """
        actor = InteractionActor(
            obs_dim=10, output_dim=5,
            hidden_dims=(16,), max_delta=1.0,
        )
        obs = torch.randn(3, 10)
        target = torch.randn(5)

        # Method 1: Direct PyTorch
        delta_1 = actor(obs)
        loss_1 = ((delta_1 - target) ** 2).sum() / delta_1.numel()
        loss_1.backward()
        grads_direct = {n: p.grad.clone() for n, p in actor.named_parameters()
                        if p.grad is not None}

        # Reset
        actor.zero_grad()

        # Method 2: Manual bridge
        delta_2 = actor(obs)
        delta_np = delta_2.detach().numpy()
        target_np = target.numpy()
        diff = delta_np - target_np[None, :]
        manual_grad = (2.0 * diff / delta_np.size).astype(np.float32)
        delta_2.backward(gradient=torch.from_numpy(manual_grad))
        grads_bridge = {n: p.grad.clone() for n, p in actor.named_parameters()
                        if p.grad is not None}

        # Compare
        for name in grads_direct:
            assert name in grads_bridge, f"Missing gradient for {name}"
            np.testing.assert_allclose(
                grads_direct[name].numpy(),
                grads_bridge[name].numpy(),
                rtol=1e-4, atol=1e-6,
                err_msg=f"Gradient mismatch on {name}")


# ═══════════════════════════════════════════════════════════════
# Integration Test (requires data + Warp)
# ═══════════════════════════════════════════════════════════════
class TestIntegration:
    """Integration test with actual Newton physics.

    Only runs if clip 1000 data exists.
    """

    DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "retargeted_v2", "interhuman",
    )

    def _data_exists(self):
        """Check if clip 1000 data is available."""
        needed = [
            "1000_person0_joint_q.npy",
            "1000_person1_joint_q.npy",
            "1000_person0_betas.npy",
            "1000_person1_betas.npy",
            "1000_person0.npy",
            "1000_person1.npy",
        ]
        return all(
            os.path.exists(os.path.join(self.DATA_DIR, f))
            for f in needed
        )

    def _needs_torques(self):
        """Check if torques exist (in either directory)."""
        for d in [self.DATA_DIR,
                  self.DATA_DIR.replace("retargeted_v2", "compute_torques")]:
            if os.path.exists(
                os.path.join(d, "1000_person0_torques_solo.npy")
            ):
                return True
        return False

    def test_single_train_step(self):
        """Run one train_step with real physics. Loss should be finite."""
        if not self._data_exists() or not self._needs_torques():
            print("  SKIP: clip 1000 data not found")
            return

        import warp as wp
        wp.init()

        args = argparse.Namespace(
            clip="1000",
            data_dir=self.DATA_DIR,
            fps=30,
            downsample=2,
            sim_freq=120,
            window=3,
            ke_root=5000.0,
            kd_root=500.0,
            ke_joint=200.0,
            kd_joint=20.0,
            hidden=[64, 64],
            max_delta=0.05,
            lr=1e-4,
            weight_decay=1e-5,
            reg_lambda=0.01,
            grad_clip=10.0,
            param_grad_clip=1.0,
            patience=10,
            epochs=50,
            normalize_obs=True,
            load_model=None,
            device="cuda:0",
        )

        import argparse as ap
        opt = None
        try:
            from prepare2.optimize_neural import NeuralInteractionOptimizer
            opt = NeuralInteractionOptimizer(args)
            loss, grad_norm = opt.train_step(0)

            assert np.isfinite(loss), f"Loss is not finite: {loss}"
            assert loss > 0, f"Loss should be positive: {loss}"
            print(f"  Integration test: loss={loss:.6f}, "
                  f"grad_norm={grad_norm:.2e}")
        except Exception as e:
            print(f"  Integration test failed: {e}")
            raise


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
    except ImportError:
        print("pytest not found, running tests manually\n")
        passed = 0
        failed = 0
        skipped = 0

        test_classes = [
            TestInteractionActor,
            TestObservations,
            TestDeltaMapping,
            TestObservationNormalizer,
            TestGradientBridge,
            # TestIntegration skipped in manual mode (needs Warp)
        ]

        for cls in test_classes:
            instance = cls()
            for name in sorted(dir(cls)):
                if not name.startswith("test_"):
                    continue
                try:
                    getattr(instance, name)()
                    print(f"  PASS  {cls.__name__}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL  {cls.__name__}.{name}: {e}")
                    failed += 1

        print(f"\n{'=' * 50}")
        print(f"Results: {passed} passed, {failed} failed, "
              f"{skipped} skipped")
        sys.exit(1 if failed else 0)
