"""
rl_tracker.py — PPO RL-based motion tracker (per-clip training).

Orchestrates:
  1. Build vectorised Newton env (RolloutEnv)
  2. Train PPO agent on the clip (~500k timesteps, ~5 min on RTX 3090)
  3. Evaluate policy → sim_positions vs ref_positions
  4. Return same metrics schema as prepare5/PHCTracker

Usage:
    from prepare6.rl_tracker import RLTracker

    tracker = RLTracker(device="cuda:0")
    result  = tracker.train_and_evaluate(joint_q, betas)
    # result['mpjpe_mm']      float
    # result['sim_positions'] (T, 22, 3)
    # result['ref_positions'] (T, 22, 3)
    # result['training_curve'] list of dicts
"""
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare6.rl_config import (
    N_ENVS, N_STEPS, TOTAL_TIMESTEPS,
    EARLY_STOP_REWARD, REWARD_WINDOW,
    OBS_DIM_SOLO, ACT_DIM_SOLO,
)
from prepare6.rollout_env import RolloutEnv
from prepare6.ppo_agent import PPOAgent
from prepare5.phc_reward import compute_tracking_errors


class RLTracker:
    """PPO RL-based per-clip physics motion tracker."""

    def __init__(
        self,
        device="cuda:0",
        n_envs=N_ENVS,
        total_timesteps=TOTAL_TIMESTEPS,
        early_stop_reward=EARLY_STOP_REWARD,
        verbose=True,
    ):
        self.device = device
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.early_stop_reward = early_stop_reward
        self.verbose = verbose

    def train_and_evaluate(self, joint_q, betas):
        """Train PPO on a single clip and return tracking metrics.

        Args:
            joint_q: (T, 76) reference joint coordinates
            betas:   (10,) SMPL-X shape params

        Returns:
            dict with:
              sim_positions:       (T, 22, 3)
              ref_positions:       (T, 22, 3)
              mpjpe_mm:            float
              per_frame_mpjpe_mm:  (T,)
              per_joint_mpjpe_mm:  (22,)
              max_error_mm:        float
              training_curve:      list of per-update dicts
              elapsed_s:           float
              training_timesteps:  int
              early_stopped:       bool
        """
        t0 = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RL Tracker: T={joint_q.shape[0]}, n_envs={self.n_envs}, "
                  f"device={self.device}")
            print(f"  Target: {self.total_timesteps:,} timesteps")
            print(f"{'='*60}")

        # ── Build env ──
        env = RolloutEnv(
            ref_joint_q=joint_q,
            betas=betas,
            n_envs=self.n_envs,
            device=self.device,
            verbose=self.verbose,
        )
        env.reset()

        # ── Build agent ──
        agent = PPOAgent(
            obs_dim=OBS_DIM_SOLO,
            act_dim=ACT_DIM_SOLO,
            device=self.device,
        )

        # ── Training loop ──
        training_curve, timesteps, early_stopped = self._run_training_loop(
            env, agent
        )

        # ── Evaluate ──
        if self.verbose:
            print(f"\n  Evaluating policy...")

        sim_positions, ref_positions, sim_joint_q = env.evaluate_single_pass(
            lambda obs: agent.act_deterministic(obs)
        )

        errors = compute_tracking_errors(sim_positions, ref_positions)
        elapsed = time.time() - t0

        if self.verbose:
            print(f"\n  {'='*40}")
            print(f"  RL Tracking complete ({elapsed:.0f}s):")
            print(f"    MPJPE:     {errors['mpjpe_mm']:.1f} mm")
            print(f"    Max error: {errors['max_error_mm']:.1f} mm")
            print(f"    Timesteps: {timesteps:,}")
            print(f"    Early stop: {early_stopped}")
            print(f"  {'='*40}\n")

        return {
            'sim_positions':      sim_positions,
            'ref_positions':      ref_positions,
            'sim_joint_q':        sim_joint_q,
            'mpjpe_mm':           errors['mpjpe_mm'],
            'per_frame_mpjpe_mm': errors['per_frame_mpjpe_mm'],
            'per_joint_mpjpe_mm': errors['per_joint_mpjpe_mm'],
            'max_error_mm':       errors['max_error_mm'],
            'training_curve':     training_curve,
            'elapsed_s':          elapsed,
            'training_timesteps': timesteps,
            'early_stopped':      early_stopped,
        }

    def _run_training_loop(self, env, agent):
        """Main PPO training loop with early stopping.

        Returns:
            training_curve: list of per-update metric dicts
            timesteps:      total timesteps collected
            early_stopped:  bool
        """
        timesteps     = 0
        training_curve = []
        reward_history = []
        early_stopped  = False
        update_idx     = 0

        steps_per_update = N_STEPS * self.n_envs

        while timesteps < self.total_timesteps:
            rollout = agent.collect_rollout(env, n_steps=N_STEPS)
            metrics = agent.update(rollout)

            timesteps += steps_per_update
            update_idx += 1

            metrics['timesteps'] = timesteps
            training_curve.append(metrics)

            reward_history.append(metrics['mean_reward'])
            if len(reward_history) > REWARD_WINDOW:
                reward_history.pop(0)

            if self.verbose and update_idx % 20 == 0:
                smooth_rew = np.mean(reward_history)
                print(f"  [{timesteps:>8,}] reward={smooth_rew:.3f}  "
                      f"actor_loss={metrics['actor_loss']:.4f}  "
                      f"entropy={metrics['entropy']:.3f}")

            # Early stopping
            if (len(reward_history) >= REWARD_WINDOW
                    and np.mean(reward_history) >= self.early_stop_reward):
                if self.verbose:
                    print(f"  Early stop: mean reward {np.mean(reward_history):.3f} "
                          f">= {self.early_stop_reward} at {timesteps:,} timesteps")
                early_stopped = True
                break

        return training_curve, timesteps, early_stopped
