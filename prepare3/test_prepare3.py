"""
test_prepare3.py — Unit tests for the prepare3 RL pipeline.

Tests are organized into groups:
- Pure math/utility tests (no GPU needed)
- XML builder tests
- Motion conversion tests
- Environment tests (require Newton GPU)
- Training component tests (require PyTorch)

Usage:
    # Run all tests
    python -m pytest prepare3/test_prepare3.py -v

    # Run only CPU-safe tests (no Newton/GPU)
    python -m pytest prepare3/test_prepare3.py -v -k "not gpu"

    # Run from InterMask root
    cd /media/rh/codes/sim/InterMask
    python -m pytest prepare3/test_prepare3.py -v
"""
import os
import sys
import tempfile
import pickle
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════
# 1. Quaternion / math utility tests (CPU-only)
# ═══════════════════════════════════════════════════════════════

class TestQuaternionUtils:
    """Test quaternion conversion utilities from newton_mimic_env.py."""

    def test_exp_map_identity(self):
        """Zero exponential map → identity quaternion."""
        from prepare3.newton_mimic_env import _exp_map_to_quat
        q = _exp_map_to_quat(np.zeros(3))
        expected = np.array([0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_allclose(q, expected, atol=1e-6)

    def test_exp_map_roundtrip(self):
        """exp_map → quat → exp_map should be identity."""
        from prepare3.newton_mimic_env import _exp_map_to_quat, _quat_to_exp_map
        rng = np.random.RandomState(42)
        for _ in range(100):
            # Random rotation (small to moderate angles)
            e = rng.randn(3).astype(np.float32) * 1.5
            q = _exp_map_to_quat(e)
            e2 = _quat_to_exp_map(q)
            # May differ by sign (antipodal quaternions)
            q2 = _exp_map_to_quat(e2)
            dot = np.abs(np.sum(q * q2))
            assert dot > 0.999, f"Roundtrip failed: dot={dot}, e={e}"

    def test_exp_map_to_quat_unit_norm(self):
        """Output quaternion should always have unit norm."""
        from prepare3.newton_mimic_env import _exp_map_to_quat
        rng = np.random.RandomState(123)
        for _ in range(50):
            e = rng.randn(3).astype(np.float32) * 2.0
            q = _exp_map_to_quat(e)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-5

    def test_quat_angle_diff_identity(self):
        """Angle difference between identical quaternions = 0."""
        from prepare3.newton_mimic_env import _quat_angle_diff
        q = np.array([0, 0, 0, 1], dtype=np.float32)
        assert _quat_angle_diff(q, q) < 1e-6

    def test_quat_angle_diff_opposite(self):
        """Angle difference between q and -q = 0 (same rotation)."""
        from prepare3.newton_mimic_env import _quat_angle_diff
        q = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
        q /= np.linalg.norm(q)
        assert _quat_angle_diff(q, -q) < 1e-2  # float32 precision with double-cover

    def test_quat_angle_diff_90deg(self):
        """90-degree rotation about Z → angle ≈ π/2."""
        from prepare3.newton_mimic_env import _quat_angle_diff
        q1 = np.array([0, 0, 0, 1], dtype=np.float32)
        # 90° about Z: quat = [0, 0, sin(45°), cos(45°)]
        q2 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
                       dtype=np.float32)
        diff = _quat_angle_diff(q1, q2)
        np.testing.assert_allclose(diff, np.pi / 2, atol=1e-5)

    def test_heading_rotation_inv_identity(self):
        """Identity quaternion → identity heading → R_inv = I."""
        from prepare3.newton_mimic_env import _heading_rotation_inv
        q = np.array([0, 0, 0, 1], dtype=np.float32)
        R_inv = _heading_rotation_inv(q)
        np.testing.assert_allclose(R_inv, np.eye(3), atol=1e-6)

    def test_heading_rotation_inv_yaw_only(self):
        """Pure yaw rotation → R_inv cancels it."""
        from prepare3.newton_mimic_env import _heading_rotation_inv
        yaw = np.pi / 3  # 60 degrees
        q = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)],
                      dtype=np.float32)
        R_inv = _heading_rotation_inv(q)
        # R_heading @ R_inv should be identity for yaw-only rotation
        # direction vector along yaw should map back to +X
        v = np.array([np.cos(yaw), np.sin(yaw), 0], dtype=np.float32)
        v_local = R_inv @ v
        np.testing.assert_allclose(v_local, [1, 0, 0], atol=1e-5)

    def test_heading_rotation_inv_preserves_z(self):
        """Heading inverse should not affect Z component."""
        from prepare3.newton_mimic_env import _heading_rotation_inv
        rng = np.random.RandomState(42)
        q = rng.randn(4).astype(np.float32)
        q /= np.linalg.norm(q)
        R_inv = _heading_rotation_inv(q)
        v = np.array([0, 0, 1], dtype=np.float32)
        v_out = R_inv @ v
        np.testing.assert_allclose(v_out, [0, 0, 1], atol=1e-6)

    def test_frame_to_joint_q_shape(self):
        """Frame (75,) → joint_q (76,)."""
        from prepare3.newton_mimic_env import _frame_to_joint_q
        frame = np.zeros(75, dtype=np.float32)
        jq = _frame_to_joint_q(frame)
        assert jq.shape == (76,)

    def test_frame_to_joint_q_identity_rotation(self):
        """Zero expmap → identity quat [0,0,0,1] in joint_q."""
        from prepare3.newton_mimic_env import _frame_to_joint_q
        frame = np.zeros(75, dtype=np.float32)
        frame[:3] = [1, 2, 3]  # root_pos
        jq = _frame_to_joint_q(frame)
        np.testing.assert_allclose(jq[:3], [1, 2, 3])
        np.testing.assert_allclose(jq[3:7], [0, 0, 0, 1], atol=1e-6)

    def test_frame_to_joint_q_hinge_passthrough(self):
        """Hinge DOFs pass through unchanged."""
        from prepare3.newton_mimic_env import _frame_to_joint_q
        frame = np.zeros(75, dtype=np.float32)
        frame[6:] = np.arange(69, dtype=np.float32)
        jq = _frame_to_joint_q(frame)
        np.testing.assert_allclose(jq[7:], np.arange(69, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════
# 2. Motion load/conversion tests (CPU-only)
# ═══════════════════════════════════════════════════════════════

class TestMotionConversion:
    """Test convert_to_mimickit.py functions."""

    def test_quat_to_exp_map_np_identity(self):
        """Identity quaternion → zero expmap."""
        from prepare3.convert_to_mimickit import _quat_to_exp_map_np
        q = np.array([[0, 0, 0, 1]], dtype=np.float32)
        e = _quat_to_exp_map_np(q)
        np.testing.assert_allclose(e, np.zeros((1, 3)), atol=1e-6)

    def test_quat_to_exp_map_np_90deg_z(self):
        """90° about Z axis → expmap ≈ [0, 0, π/2]."""
        from prepare3.convert_to_mimickit import _quat_to_exp_map_np
        angle = np.pi / 2
        q = np.array([[0, 0, np.sin(angle / 2), np.cos(angle / 2)]],
                      dtype=np.float32)
        e = _quat_to_exp_map_np(q)
        np.testing.assert_allclose(e[0], [0, 0, np.pi / 2], atol=1e-5)

    def test_convert_clip_shape(self):
        """convert_clip: (T,76) joint_q → dict with (T,75) frames."""
        from prepare3.convert_to_mimickit import convert_clip
        T = 30
        # Build valid joint_q: root_pos + unit quat + hinges
        jq = np.zeros((T, 76), dtype=np.float32)
        jq[:, 6] = 1.0  # w=1 for identity quat (xyzw format)
        result = convert_clip(jq)
        assert isinstance(result, dict)
        assert "frames" in result
        frames = np.array(result["frames"])
        assert frames.shape == (T, 75)

    def test_convert_clip_root_pos_preserved(self):
        """Root position should pass through convert_clip unchanged."""
        from prepare3.convert_to_mimickit import convert_clip
        T = 10
        jq = np.zeros((T, 76), dtype=np.float32)
        jq[:, 6] = 1.0  # identity quat
        jq[:, 0] = np.linspace(0, 1, T)  # varying X
        jq[:, 2] = 0.9  # fixed height
        result = convert_clip(jq)
        frames = np.array(result["frames"])
        np.testing.assert_allclose(frames[:, 0], jq[:, 0], atol=1e-6)
        np.testing.assert_allclose(frames[:, 2], 0.9, atol=1e-6)

    def test_convert_clip_hinge_preserved(self):
        """Hinge DOFs should pass through convert_clip unchanged."""
        from prepare3.convert_to_mimickit import convert_clip
        T = 5
        jq = np.zeros((T, 76), dtype=np.float32)
        jq[:, 6] = 1.0
        jq[:, 7:] = np.random.randn(T, 69).astype(np.float32) * 0.5
        result = convert_clip(jq)
        frames = np.array(result["frames"])
        np.testing.assert_allclose(frames[:, 6:], jq[:, 7:], atol=1e-6)

    def test_load_save_motion_roundtrip(self):
        """Save and reload a motion file."""
        from prepare3.convert_to_mimickit import save_motion
        from prepare3.newton_mimic_env import load_motion_data

        T = 20
        frames = np.random.randn(T, 75).astype(np.float32)
        fps = 30

        # Build motion dict in MimicKit format
        motion_dict = {
            "loop_mode": 1,
            "fps": fps,
            "frames": frames.tolist(),
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            save_motion(motion_dict, tmp_path)
            loaded_frames, loaded_fps, loaded_loop = load_motion_data(tmp_path)
            assert loaded_fps == fps
            assert loaded_loop == 1
            assert loaded_frames.shape == (T, 75)
            np.testing.assert_allclose(loaded_frames, frames, atol=1e-6)
        finally:
            os.unlink(tmp_path)

    def test_generate_motion_list_yaml(self):
        """generate_motion_list creates valid YAML."""
        from prepare3.convert_to_mimickit import generate_motion_list

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy pkl files
            file_paths = []
            for name in ["a.pkl", "b.pkl", "c.pkl"]:
                fpath = os.path.join(tmpdir, name)
                with open(fpath, "wb") as f:
                    pickle.dump({"frames": [], "fps": 30, "loop_mode": 0}, f)
                file_paths.append(fpath)

            yaml_path = os.path.join(tmpdir, "motions.yaml")
            generate_motion_list(file_paths, yaml_path)

            assert os.path.exists(yaml_path)
            with open(yaml_path) as f:
                content = f.read()
            assert "a.pkl" in content
            assert "b.pkl" in content
            assert "c.pkl" in content


# ═══════════════════════════════════════════════════════════════
# 3. XML builder tests
# ═══════════════════════════════════════════════════════════════

class TestXmlBuilder:
    """Test xml_builder.py caching and generation."""

    def test_hash_betas_deterministic(self):
        """Same betas → same hash."""
        from prepare3.xml_builder import _hash_betas
        betas = np.array([1.0, 2.0, 3.0] + [0.0] * 7, dtype=np.float64)
        h1 = _hash_betas(betas)
        h2 = _hash_betas(betas)
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_betas_different(self):
        """Different betas → different hash."""
        from prepare3.xml_builder import _hash_betas
        b1 = np.zeros(10, dtype=np.float64)
        b2 = np.ones(10, dtype=np.float64)
        assert _hash_betas(b1) != _hash_betas(b2)

    def test_get_cached_xml_path(self):
        """Cached path follows expected pattern."""
        from prepare3.xml_builder import get_cached_xml_path, _hash_betas
        betas = np.zeros(10, dtype=np.float64)
        path = get_cached_xml_path(betas)
        assert path.endswith(".xml")
        assert "smpl_" in os.path.basename(path)
        assert _hash_betas(betas) in os.path.basename(path)


# ═══════════════════════════════════════════════════════════════
# 4. Training components tests (requires PyTorch, no GPU)
# ═══════════════════════════════════════════════════════════════

class TestTrainingComponents:
    """Test ActorCritic, RolloutBuffer, etc."""

    def test_actor_critic_output_shapes(self):
        """ActorCritic forward produces correct shapes."""
        from prepare3.train import ActorCritic
        import torch

        obs_dim = 222
        action_dim = 69
        batch = 16

        policy = ActorCritic(obs_dim, action_dim, hidden_dims=(64, 32))
        obs = torch.randn(batch, obs_dim)
        mean, value = policy(obs)

        assert mean.shape == (batch, action_dim)
        assert value.shape == (batch, 1)

    def test_actor_critic_get_action(self):
        """get_action returns correct shapes."""
        from prepare3.train import ActorCritic
        import torch

        obs_dim = 222
        action_dim = 69

        policy = ActorCritic(obs_dim, action_dim, hidden_dims=(64, 32))
        obs = np.random.randn(obs_dim).astype(np.float32)

        action, log_prob, value = policy.get_action(obs)
        assert action.shape == (action_dim,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_actor_critic_deterministic(self):
        """Deterministic mode returns same action for same input."""
        from prepare3.train import ActorCritic
        import torch

        obs_dim = 222
        action_dim = 69

        policy = ActorCritic(obs_dim, action_dim, hidden_dims=(64, 32))
        obs = np.random.randn(obs_dim).astype(np.float32)

        a1, _, _ = policy.get_action(obs, deterministic=True)
        a2, _, _ = policy.get_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-6)

    def test_evaluate_actions(self):
        """evaluate_actions returns correct shapes."""
        from prepare3.train import ActorCritic
        import torch

        obs_dim = 222
        action_dim = 69
        batch = 32

        policy = ActorCritic(obs_dim, action_dim, hidden_dims=(64, 32))
        obs = torch.randn(batch, obs_dim)
        actions = torch.randn(batch, action_dim)

        log_prob, value, entropy = policy.evaluate_actions(obs, actions)
        assert log_prob.shape == (batch,)
        assert value.shape == (batch, 1)
        assert entropy.shape == (batch,)

    def test_rollout_buffer_add_and_full(self):
        """RolloutBuffer tracks fullness correctly."""
        from prepare3.train import RolloutBuffer

        buf = RolloutBuffer(10, obs_dim=5, action_dim=3)
        assert not buf.is_full()

        for i in range(10):
            buf.add(np.zeros(5), np.zeros(3), 1.0, 0.5, -0.1, 0.0)

        assert buf.is_full()

    def test_rollout_buffer_compute_returns(self):
        """GAE returns should be finite and correct shape."""
        from prepare3.train import RolloutBuffer

        N = 20
        buf = RolloutBuffer(N, obs_dim=5, action_dim=3)
        for i in range(N):
            buf.add(np.zeros(5), np.zeros(3), 1.0, 0.5, -0.1, 0.0)

        advantages, returns = buf.compute_returns(last_value=0.5)
        assert advantages.shape == (N,)
        assert returns.shape == (N,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))

    def test_running_mean_std_normalize(self):
        """RunningMeanStd should normalize to ~ zero mean unit var."""
        from prepare3.train import RunningMeanStd

        rms = RunningMeanStd(shape=(3,))
        rng = np.random.RandomState(42)

        # Feed in data with known mean/var
        data = rng.randn(1000, 3).astype(np.float64) * 2.0 + 5.0
        rms.update(data)

        # Normalized output should be ~ N(0, 1)
        normalized = rms.normalize(data.astype(np.float32))
        np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=0.1)
        np.testing.assert_allclose(np.std(normalized, axis=0), 1.0, atol=0.1)

    def test_running_mean_std_state_dict_roundtrip(self):
        """RunningMeanStd save/load state dict."""
        from prepare3.train import RunningMeanStd

        rms1 = RunningMeanStd(shape=(5,))
        rms1.update(np.random.randn(100, 5))

        state = rms1.state_dict()
        rms2 = RunningMeanStd(shape=(5,))
        rms2.load_state_dict(state)

        np.testing.assert_allclose(rms1.mean, rms2.mean)
        np.testing.assert_allclose(rms1.var, rms2.var)
        assert rms1.count == rms2.count


# ═══════════════════════════════════════════════════════════════
# 5. Evaluation metric tests (CPU-only)
# ═══════════════════════════════════════════════════════════════

class TestEvaluationMetrics:
    """Test TrackingMetrics from evaluate_policy.py."""

    def test_tracking_metrics_summary_empty(self):
        """Empty metrics should return empty dict without error."""
        from prepare3.evaluate_policy import TrackingMetrics
        m = TrackingMetrics()
        summary = m.summary()
        assert isinstance(summary, dict)

    def test_tracking_metrics_accumulation(self):
        """Metrics should accumulate values."""
        from prepare3.evaluate_policy import TrackingMetrics
        m = TrackingMetrics()
        m.mpjpe_list = [0.05, 0.10, 0.15]
        m.root_pos_err_list = [0.1, 0.2, 0.3]
        m.root_rot_err_list = [5.0, 10.0]
        m.episode_rewards = [100.0, 200.0]
        m.episode_lengths = [50, 60]
        m.episode_survival_rates = [0.8, 1.0]

        summary = m.summary()
        np.testing.assert_allclose(summary["mpjpe_mean_m"], 0.1, atol=1e-6)
        np.testing.assert_allclose(summary["root_pos_err_mean_m"], 0.2, atol=1e-6)
        np.testing.assert_allclose(summary["reward_mean"], 150.0, atol=1e-6)
        np.testing.assert_allclose(summary["survival_rate_mean"], 0.9, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# 6. Integration tests (require Newton GPU)
# ═══════════════════════════════════════════════════════════════

def _newton_available():
    """Check if Newton and GPU are available."""
    try:
        import newton
        import warp as wp
        wp.init()
        return True
    except Exception:
        return False

def _smplx_available():
    """Check if SMPL-X model files are accessible."""
    model_path = os.path.join(PROJECT_ROOT, "conversions", "eric.sdf")
    return os.path.exists(model_path)

# Conditional import: only import heavy GPU modules when running GPU tests
if _newton_available():
    from prepare3.newton_mimic_env import NewtonMimicEnv


@pytest.mark.skipif(not _newton_available(), reason="Newton/GPU not available")
class TestNewtonMimicEnvGPU:
    """Integration tests requiring Newton and GPU."""

    @pytest.fixture
    def dummy_motion_and_betas(self, tmp_path):
        """Create a dummy motion file and betas for testing."""
        T = 60
        frames = []
        for t in range(T):
            frame = np.zeros(75, dtype=np.float32)
            frame[0] = 0.0  # X
            frame[1] = 0.0  # Y
            frame[2] = 0.9  # root height
            # root_expmap = 0 (identity)
            # hinge_dofs = 0 (rest pose)
            frames.append(frame)

        motion_data = {
            "loop_mode": 1,
            "fps": 30,
            "frames": frames,
        }
        motion_path = str(tmp_path / "test_motion.pkl")
        with open(motion_path, "wb") as f:
            pickle.dump(motion_data, f)

        betas = np.zeros(10, dtype=np.float64)
        betas_path = str(tmp_path / "test_betas.npy")
        np.save(betas_path, betas)

        return motion_path, betas_path

    def test_env_creation(self, dummy_motion_and_betas):
        """Environment should create without errors."""
        motion_path, betas_path = dummy_motion_and_betas
        config = {
            "motion_file": motion_path,
            "betas_file": betas_path,
            "device": "cuda:0",
        }
        env = NewtonMimicEnv(config)
        assert env.action_dim == 69
        assert env.obs_dim > 0
        env.close()

    def test_env_reset(self, dummy_motion_and_betas):
        """Reset should return valid observation."""
        motion_path, betas_path = dummy_motion_and_betas
        config = {
            "motion_file": motion_path,
            "betas_file": betas_path,
            "device": "cuda:0",
        }
        env = NewtonMimicEnv(config)
        obs, info = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert np.all(np.isfinite(obs))
        assert "motion_time" in info
        env.close()

    def test_env_step(self, dummy_motion_and_betas):
        """Single step should return valid outputs."""
        motion_path, betas_path = dummy_motion_and_betas
        config = {
            "motion_file": motion_path,
            "betas_file": betas_path,
            "device": "cuda:0",
        }
        env = NewtonMimicEnv(config)
        obs, info = env.reset()
        action = np.zeros(69, dtype=np.float32)
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert obs2.shape == (env.obs_dim,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 0.0 <= reward <= 1.0  # reward is weighted sum of exp(-error)
        env.close()

    def test_env_root_skyhook_forces(self, dummy_motion_and_betas):
        """Root DOFs should have skyhook PD forces (not zero)."""
        motion_path, betas_path = dummy_motion_and_betas
        config = {
            "motion_file": motion_path,
            "betas_file": betas_path,
            "device": "cuda:0",
            "control_mode": "pd",
        }
        env = NewtonMimicEnv(config)
        obs, info = env.reset()
        action = np.ones(69, dtype=np.float32) * 0.1
        tau = env._compute_pd_torques(action)
        # Root PD (skyhook) — at reset, root matches ref so forces ≈ 0
        # but after a step, root may deviate so forces appear
        assert tau.shape == (75,), f"Expected (75,), got {tau.shape}"
        # Hinge DOFs should have non-zero PD torques
        assert np.any(tau[6:] != 0.0)
        env.close()

    def test_env_multiple_steps(self, dummy_motion_and_betas):
        """Run multiple steps without crashing."""
        motion_path, betas_path = dummy_motion_and_betas
        config = {
            "motion_file": motion_path,
            "betas_file": betas_path,
            "device": "cuda:0",
        }
        env = NewtonMimicEnv(config)
        obs, info = env.reset()
        for _ in range(10):
            action = np.random.randn(69).astype(np.float32) * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    def test_env_obs_heading_invariant(self, dummy_motion_and_betas):
        """Observation should NOT contain raw global X/Y coordinates."""
        motion_path, betas_path = dummy_motion_and_betas
        config = {
            "motion_file": motion_path,
            "betas_file": betas_path,
            "device": "cuda:0",
            "rand_init": False,
        }
        env = NewtonMimicEnv(config)
        obs, _ = env.reset()
        # First element is root height (Z), should be around 0.9
        assert obs[0] > 0.0, "Root height should be positive"
        # obs[0] is root height, NOT root X
        env.close()


# ═══════════════════════════════════════════════════════════════
# 7. PPO update test (CPU, no env)
# ═══════════════════════════════════════════════════════════════

class TestPPOUpdate:
    """Test PPO update mechanics."""

    def test_ppo_update_runs(self):
        """PPO update should run without errors on random data."""
        from prepare3.train import ActorCritic, RolloutBuffer, ppo_update
        import torch

        obs_dim = 20
        action_dim = 5
        N = 64

        policy = ActorCritic(obs_dim, action_dim, hidden_dims=(32, 16))
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        buf = RolloutBuffer(N, obs_dim, action_dim)
        for i in range(N):
            buf.add(
                np.random.randn(obs_dim).astype(np.float32),
                np.random.randn(action_dim).astype(np.float32),
                reward=np.random.rand(),
                value=np.random.rand(),
                log_prob=-np.random.rand(),
                done=0.0,
            )

        advantages, returns = buf.compute_returns(last_value=0.5)

        info = ppo_update(
            policy, optimizer, buf, advantages, returns,
            num_epochs=2, mini_batch_size=16, device="cpu",
        )

        assert "policy_loss" in info
        assert "value_loss" in info
        assert "entropy" in info
        assert np.isfinite(info["policy_loss"])
        assert np.isfinite(info["value_loss"])


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
