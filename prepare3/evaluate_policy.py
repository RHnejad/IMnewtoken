"""
evaluate_policy.py — Evaluate a trained policy and compute tracking metrics.

Loads a trained checkpoint from train.py and runs deterministic rollouts,
computing per-frame tracking metrics:

- MPJPE (Mean Per-Joint Position Error): average Euclidean distance
  between simulated and reference body positions (in meters)
- Root position error
- Root orientation error (degrees)
- Joint angle error (radians)
- CoM trajectory drift

Also supports optional Newton viewer visualization of the rollout.

Usage:
    # Quick evaluation (10 episodes, no visualization)
    python prepare3/evaluate_policy.py \\
        --checkpoint output/prepare3/1000_p0/best_policy.pt \\
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \\
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy

    # Detailed evaluation with visualization
    python prepare3/evaluate_policy.py \\
        --checkpoint output/prepare3/1000_p0/best_policy.pt \\
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \\
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \\
        --visualize \\
        --num-episodes 50 \\
        --save-trajectory
"""
import os
import sys
import json
import argparse
import numpy as np
import time

import torch

try:
    import warp as wp
    import newton
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare3.newton_mimic_env import (
    NewtonMimicEnv, _quat_angle_diff, DOFS_PER_PERSON,
)
from prepare3.train import ActorCritic, RunningMeanStd


# ═══════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════
class TrackingMetrics:
    """Accumulates per-step tracking error statistics."""

    def __init__(self):
        self.mpjpe_list = []
        self.root_pos_err_list = []
        self.root_rot_err_list = []
        self.joint_err_list = []
        self.com_drift_list = []   # cumulative CoM X/Y drift
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_survival_rates = []  # fraction of ref motion completed

        # Per-episode accumulators
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_start_root_xy = None

    def step(self, env, reward):
        """Collect metrics for current timestep."""
        cq = env.state_0.joint_q.numpy()
        ref_q = env._get_ref_joint_q(env.motion_time)

        # Root position error
        root_pos_err = np.linalg.norm(cq[:3] - ref_q[:3])
        self.root_pos_err_list.append(root_pos_err)

        # Root orientation error (degrees)
        root_rot_err = np.degrees(_quat_angle_diff(cq[3:7], ref_q[3:7]))
        self.root_rot_err_list.append(root_rot_err)

        # Joint angle error (mean absolute per-DOF)
        joint_err = np.mean(np.abs(cq[7:] - ref_q[7:]))
        self.joint_err_list.append(joint_err)

        # MPJPE: need body positions from FK
        sim_body_pos = env._get_body_positions()
        ref_body_pos = env._get_ref_body_positions(env.motion_time)
        if sim_body_pos is not None and ref_body_pos is not None:
            n_bodies = min(sim_body_pos.shape[0], ref_body_pos.shape[0])
            diffs = sim_body_pos[:n_bodies] - ref_body_pos[:n_bodies]
            per_joint_err = np.linalg.norm(diffs, axis=-1)  # (n_bodies,)
            mpjpe = np.mean(per_joint_err)
            self.mpjpe_list.append(mpjpe)

        # CoM drift
        if self._ep_start_root_xy is None:
            self._ep_start_root_xy = cq[:2].copy()
        com_drift = np.linalg.norm(cq[:2] - self._ep_start_root_xy)
        self.com_drift_list.append(com_drift)

        self._ep_reward += reward
        self._ep_length += 1

    def end_episode(self, env):
        """Finalize episode-level metrics."""
        self.episode_rewards.append(self._ep_reward)
        self.episode_lengths.append(self._ep_length)

        # Survival rate: fraction of reference motion completed
        if env.ref_duration > 0:
            survival = min(env.time / env.ref_duration, 1.0)
        else:
            survival = 1.0
        self.episode_survival_rates.append(survival)

        # Reset per-episode state
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_start_root_xy = None

    def summary(self):
        """Return summary statistics dict."""
        results = {}

        if self.mpjpe_list:
            results["mpjpe_mean_m"] = float(np.mean(self.mpjpe_list))
            results["mpjpe_std_m"] = float(np.std(self.mpjpe_list))
            results["mpjpe_median_m"] = float(np.median(self.mpjpe_list))

        if self.root_pos_err_list:
            results["root_pos_err_mean_m"] = float(np.mean(self.root_pos_err_list))
            results["root_pos_err_std_m"] = float(np.std(self.root_pos_err_list))

        if self.root_rot_err_list:
            results["root_rot_err_mean_deg"] = float(np.mean(self.root_rot_err_list))
            results["root_rot_err_std_deg"] = float(np.std(self.root_rot_err_list))

        if self.joint_err_list:
            results["joint_err_mean_rad"] = float(np.mean(self.joint_err_list))

        if self.com_drift_list:
            results["com_drift_mean_m"] = float(np.mean(self.com_drift_list))
            results["com_drift_max_m"] = float(np.max(self.com_drift_list))

        if self.episode_rewards:
            results["reward_mean"] = float(np.mean(self.episode_rewards))
            results["reward_std"] = float(np.std(self.episode_rewards))

        if self.episode_lengths:
            results["ep_length_mean"] = float(np.mean(self.episode_lengths))

        if self.episode_survival_rates:
            results["survival_rate_mean"] = float(np.mean(self.episode_survival_rates))
            results["survival_rate_min"] = float(np.min(self.episode_survival_rates))

        return results


# ═══════════════════════════════════════════════════════════════
# Evaluation loop
# ═══════════════════════════════════════════════════════════════
def evaluate(args):
    """Run deterministic policy evaluation."""
    print("=" * 60)
    print("prepare3 — Policy Evaluation")
    print("=" * 60)

    # ── Load checkpoint ──────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    print(f"Checkpoint: {args.checkpoint}")
    if "iteration" in ckpt:
        print(f"  Iteration: {ckpt['iteration']}")
    if "total_steps" in ckpt:
        print(f"  Total steps: {ckpt['total_steps']:,}")
    if "best_mean_reward" in ckpt:
        print(f"  Best reward: {ckpt['best_mean_reward']:.4f}")
    print()

    # ── Build environment ────────────────────────────────────
    env_config = {
        "motion_file": args.motion,
        "betas_file": args.betas,
        "device": args.device,
        "sim_freq": args.sim_freq,
        "control_freq": args.control_freq,
        "control_mode": args.control_mode,
        "max_episode_length": args.max_ep_length,
        "enable_early_termination": True,
        "rand_init": not args.from_start,
        "enable_tar_obs": True,  # match training config
    }

    env = NewtonMimicEnv(env_config)
    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")
    print(f"Ref motion:      {env.ref_T} frames, {env.ref_duration:.2f}s")
    print()

    # ── Build policy ─────────────────────────────────────────
    # Infer hidden dims from checkpoint
    policy_state = ckpt["policy"]
    # Detect hidden dims from backbone layer shapes
    hidden_dims = []
    i = 0
    while f"backbone.{i * 2}.weight" in policy_state:
        hidden_dims.append(policy_state[f"backbone.{i * 2}.weight"].shape[0])
        i += 1
    if not hidden_dims:
        hidden_dims = [512, 256, 128]  # fallback

    print(f"Detected hidden dims: {hidden_dims}")

    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(hidden_dims),
    ).to(args.device)
    policy.load_state_dict(policy_state)
    policy.eval()

    # ── Observation normalizer ───────────────────────────────
    obs_normalizer = RunningMeanStd(obs_dim)
    if "obs_normalizer" in ckpt:
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    # ── Metrics tracker ──────────────────────────────────────
    metrics = TrackingMetrics()

    # ── Trajectory storage ───────────────────────────────────
    all_trajectories = []

    # ── Run evaluation ───────────────────────────────────────
    print(f"Running {args.num_episodes} evaluation episodes...")
    print("-" * 60)

    for ep in range(args.num_episodes):
        if args.from_start:
            obs, info = env.reset(seed=ep)
        else:
            obs, info = env.reset()

        done = False
        ep_trajectory = []

        while not done:
            obs_norm = obs_normalizer.normalize(obs)
            action, _, _ = policy.get_action(obs_norm, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            metrics.step(env, reward)

            if args.save_trajectory:
                cq = env.state_0.joint_q.numpy().copy()
                ep_trajectory.append(cq)

        metrics.end_episode(env)

        if args.save_trajectory:
            all_trajectories.append(np.array(ep_trajectory))

        # Progress
        if (ep + 1) % max(1, args.num_episodes // 10) == 0:
            print(f"  Episode {ep + 1}/{args.num_episodes}  "
                  f"reward={metrics.episode_rewards[-1]:.3f}  "
                  f"length={metrics.episode_lengths[-1]}  "
                  f"survival={metrics.episode_survival_rates[-1]:.2%}")

    # ── Print results ────────────────────────────────────────
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    summary = metrics.summary()
    for key, val in summary.items():
        if "deg" in key:
            print(f"  {key:30s}  {val:8.2f}°")
        elif "_m" in key:
            print(f"  {key:30s}  {val:8.4f} m")
        elif "rate" in key:
            print(f"  {key:30s}  {val:8.2%}")
        else:
            print(f"  {key:30s}  {val:8.4f}")

    # ── Save results ─────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save trajectories
    if args.save_trajectory and all_trajectories:
        traj_path = os.path.join(output_dir, "eval_trajectories.npz")
        np.savez_compressed(traj_path,
                            **{f"ep_{i}": t for i, t in enumerate(all_trajectories)})
        print(f"Trajectories saved: {traj_path}")

    # ── Autonomous balance check ─────────────────────────────
    print()
    print("-" * 60)
    survival_mean = summary.get("survival_rate_mean", 0)
    mpjpe = summary.get("mpjpe_mean_m", float("inf"))

    if survival_mean >= 0.9:
        print("✓ Character maintains balance (>90% survival rate)")
    else:
        print(f"✗ Character struggles to balance ({survival_mean:.0%} survival)")

    if mpjpe < 0.10:
        print(f"✓ Excellent tracking (MPJPE={mpjpe:.4f}m < 0.10m)")
    elif mpjpe < 0.20:
        print(f"~ Reasonable tracking (MPJPE={mpjpe:.4f}m < 0.20m)")
    else:
        print(f"✗ Poor tracking (MPJPE={mpjpe:.4f}m >= 0.20m)")

    root_drift = summary.get("com_drift_mean_m", float("inf"))
    if root_drift < 1.0:
        print(f"✓ Minimal CoM drift ({root_drift:.3f}m)")
    else:
        print(f"✗ Significant CoM drift ({root_drift:.3f}m) — root may need correction")

    print("=" * 60)

    env.close()
    return summary


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Newton motion tracking policy (prepare3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained policy checkpoint (.pt)")
    parser.add_argument("--motion", required=True,
                        help="Path to .pkl motion file")
    parser.add_argument("--betas", required=True,
                        help="Path to .npy betas file")

    # Evaluation
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--from-start", action="store_true",
                        help="Always start from t=0 (no random init)")
    parser.add_argument("--save-trajectory", action="store_true",
                        help="Save joint_q trajectory for each episode")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable Newton viewer (requires display)")

    # Environment settings (should match training)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-freq", type=int, default=480)
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--control-mode", default="pd",
                        choices=["pd", "torque"])
    parser.add_argument("--max-ep-length", type=float, default=10.0)

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Directory for results (default: checkpoint dir)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
