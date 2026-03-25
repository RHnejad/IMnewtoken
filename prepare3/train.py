"""
train.py — PPO training loop for Newton-based motion tracking.

Standalone training script that uses NewtonMimicEnv with a simple PPO
implementation. Does NOT require MimicKit's complex agent hierarchy.

Supports:
- Single-character motion tracking
- DeepMimic-style reward (pose + velocity + root + key-body tracking)
- Heading-invariant observations (no global X/Y)
- SolverMuJoCo forward simulation (no differentiable sim)
- Checkpoint saving at regular intervals
- TensorBoard logging

Usage:
    python prepare3/train.py \\
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \\
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \\
        --output-dir output/prepare3/1000_p0 \\
        --max-steps 10000000 \\
        --device cuda:0

Architecture:
    Policy / Value network: shared MLP backbone with separate heads
    Action: (69,) hinge DOF position targets fed through PD controller
    Observation: heading-invariant features (see newton_mimic_env.py)
"""
import os
import sys
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare3.newton_mimic_env import NewtonMimicEnv


# ═══════════════════════════════════════════════════════════════
# Policy Network (Actor-Critic)
# ═══════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    """
    Two-headed MLP: shared backbone → actor (mean) + critic (value).

    The actor outputs action means; a learnable log_std vector parameterizes
    the diagonal Gaussian policy.
    """

    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 256, 128),
                 init_log_std=-1.0):
        """
        Args:
            obs_dim: observation vector dimension
            action_dim: action vector dimension (69 hinge DOFs)
            hidden_dims: MLP hidden layer sizes
            init_log_std: initial log standard deviation for action distribution
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared backbone
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Actor head: action mean
        self.actor_mean = nn.Linear(in_dim, action_dim)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, dtype=torch.float32)
        )

        # Critic head: state value
        self.critic = nn.Linear(in_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor output: small init for exploration
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, obs):
        """
        Args:
            obs: (batch, obs_dim) tensor

        Returns:
            action_mean: (batch, action_dim)
            value: (batch, 1)
        """
        # Sanitize input
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs = torch.clamp(obs, -100.0, 100.0)
        features = self.backbone(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.

        Args:
            obs: (obs_dim,) numpy array
            deterministic: if True, return mean (no noise)

        Returns:
            action: (action_dim,) numpy array
            log_prob: float
            value: float
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            mean, value = self.forward(obs_t)
            std = torch.exp(self.log_std)
            # Protect against NaN from diverged observations
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
            std = torch.clamp(std, min=1e-6, max=1e6)
            std = torch.nan_to_num(std, nan=1e-2, posinf=1e6, neginf=1e-6)

            if deterministic:
                action = mean
                log_prob = torch.zeros(1).to(obs_t.device)
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.squeeze().item(),
        )

    def evaluate_actions(self, obs, actions):
        """
        Evaluate log_prob and value for given obs-action pairs (batched).

        Args:
            obs: (batch, obs_dim) tensor
            actions: (batch, action_dim) tensor

        Returns:
            log_prob: (batch,) tensor
            value: (batch, 1) tensor
            entropy: (batch,) tensor
        """
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        # Clamp mean to prevent NaN from diverged states
        mean = torch.clamp(mean, -1e6, 1e6)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
        std = torch.clamp(std, min=1e-6, max=1e6)
        std = torch.nan_to_num(std, nan=1e-2, posinf=1e6, neginf=1e-6)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy


# ═══════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════
class RolloutBuffer:
    """Fixed-size buffer for PPO rollout data."""

    def __init__(self, buffer_size, obs_dim, action_dim):
        self.buffer_size = buffer_size
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def is_full(self):
        return self.full

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute GAE-Lambda advantages and discounted returns.

        Args:
            last_value: value estimate for state after last stored transition
            gamma: discount factor
            lam: GAE lambda

        Returns:
            advantages: (buffer_size,) array
            returns: (buffer_size,) array
        """
        advantages = np.zeros(self.buffer_size, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t]
                     + gamma * next_value * next_non_terminal
                     - self.values[t])
            advantages[t] = last_gae_lam = (
                delta + gamma * lam * next_non_terminal * last_gae_lam
            )

        returns = advantages + self.values[:self.buffer_size]
        return advantages, returns

    def reset(self):
        self.ptr = 0
        self.full = False


# ═══════════════════════════════════════════════════════════════
# PPO Update
# ═══════════════════════════════════════════════════════════════
def ppo_update(policy, optimizer, buffer, advantages, returns,
               clip_eps=0.2, value_loss_coef=0.5, entropy_coef=0.01,
               num_epochs=4, mini_batch_size=256, max_grad_norm=0.5,
               device="cuda:0"):
    """
    Perform PPO update on the policy.

    Args:
        policy: ActorCritic network
        optimizer: torch optimizer
        buffer: RolloutBuffer with collected data
        advantages: (N,) numpy array
        returns: (N,) numpy array
        Other standard PPO hyperparameters

    Returns:
        update_info: dict with loss statistics
    """
    N = buffer.buffer_size

    # Normalize advantages
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # To tensors
    # Sanitize buffer data before converting to tensors
    buf_obs = np.nan_to_num(buffer.obs[:N], nan=0.0, posinf=100.0, neginf=-100.0)
    buf_obs = np.clip(buf_obs, -100.0, 100.0)
    buf_act = np.nan_to_num(buffer.actions[:N], nan=0.0, posinf=100.0, neginf=-100.0)
    buf_lp = np.nan_to_num(buffer.log_probs[:N], nan=0.0, posinf=100.0, neginf=-100.0)

    obs_t = torch.tensor(buf_obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(buf_act, dtype=torch.float32, device=device)
    old_log_probs_t = torch.tensor(buf_lp, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

    # Sanitize advantages/returns
    adv_t = torch.nan_to_num(adv_t, nan=0.0)
    ret_t = torch.nan_to_num(ret_t, nan=0.0)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_updates = 0

    for _epoch in range(num_epochs):
        indices = np.random.permutation(N)
        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            idx = indices[start:end]

            mb_obs = obs_t[idx]
            mb_act = act_t[idx]
            mb_old_lp = old_log_probs_t[idx]
            mb_adv = adv_t[idx]
            mb_ret = ret_t[idx]

            log_prob, value, entropy = policy.evaluate_actions(mb_obs, mb_act)

            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_prob - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * (value.squeeze(-1) - mb_ret).pow(2).mean()

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = (policy_loss
                    + value_loss_coef * value_loss
                    + entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()

            # Skip update if NaN gradients detected
            has_nan = False
            for p in policy.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan or torch.isnan(loss):
                optimizer.zero_grad()  # discard NaN gradients
                continue

            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_updates += 1

    info = {
        "policy_loss": total_policy_loss / max(total_updates, 1),
        "value_loss": total_value_loss / max(total_updates, 1),
        "entropy": total_entropy / max(total_updates, 1),
    }
    return info


# ═══════════════════════════════════════════════════════════════
# Observation normalization (running mean/var)
# ═══════════════════════════════════════════════════════════════
class RunningMeanStd:
    """Online running mean and standard deviation."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + 1e-8
        )

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(),
                "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════
def train_loop(args):
    """Main training loop."""
    print("=" * 60)
    print("prepare3 — PPO training for Newton motion tracking")
    print("=" * 60)

    # ── Build environment ────────────────────────────────────
    env_config = {
        "motion_file": args.motion,
        "betas_file": args.betas,
        "device": args.device,
        "sim_freq": args.sim_freq,
        "control_freq": args.control_freq,
        "control_mode": args.control_mode,
        "max_episode_length": args.max_ep_length,
        "enable_early_termination": not args.no_early_termination,
        "rand_init": not args.no_rand_init,
        "enable_tar_obs": not args.no_tar_obs,
    }

    print(f"Motion file: {args.motion}")
    print(f"Betas file:  {args.betas}")
    print(f"Device:      {args.device}")
    print(f"Control:     {args.control_mode}")
    print(f"Sim freq:    {args.sim_freq}")
    print(f"Ctrl freq:   {args.control_freq}")
    print()

    env = NewtonMimicEnv(env_config)
    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")
    print(f"Ref motion:      {env.ref_T} frames, {env.ref_duration:.2f}s")
    print()

    # ── Build policy ─────────────────────────────────────────
    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        init_log_std=args.init_log_std,
    ).to(args.device)

    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy parameters: {num_params:,}")

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps // args.steps_per_iter,
        eta_min=args.lr * 0.01,
    )

    # ── Observation normalizer ───────────────────────────────
    obs_normalizer = RunningMeanStd(obs_dim)

    # ── TensorBoard ──────────────────────────────────────────
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        print(f"TensorBoard: {tb_dir}")
    except ImportError:
        print("TensorBoard not available — logging to stdout only")

    # ── Output directory ─────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved: {config_path}")
    print()

    # ── Buffer ───────────────────────────────────────────────
    buffer = RolloutBuffer(args.steps_per_iter, obs_dim, action_dim)

    # ── Training loop ────────────────────────────────────────
    total_steps = 0
    iteration = 0
    best_mean_reward = -float("inf")
    start_time = time.time()

    # Load checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        total_steps = ckpt.get("total_steps", 0)
        iteration = ckpt.get("iteration", 0)
        best_mean_reward = ckpt.get("best_mean_reward", -float("inf"))
        print(f"Resumed from {args.resume} (step {total_steps}, iter {iteration})")

    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_rewards = []
    episode_lengths = []

    print(f"Training for {args.max_steps:,} total steps...")
    print("-" * 60)

    while total_steps < args.max_steps:
        policy.eval()
        buffer.reset()

        # ── Collect rollout ──────────────────────────────────
        for _step in range(args.steps_per_iter):
            # Normalize observation (with NaN protection)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            obs_norm = obs_normalizer.normalize(obs)
            obs_norm = np.clip(obs_norm, -10.0, 10.0)

            action, log_prob, value = policy.get_action(obs_norm)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs_norm, action, reward, value, log_prob, float(done))
            total_steps += 1
            episode_reward += reward
            episode_length += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0.0
                episode_length = 0
                obs, info = env.reset()
            else:
                obs = next_obs

        # ── Update observation normalizer ────────────────────
        obs_normalizer.update(buffer.obs[:buffer.buffer_size])

        # ── Compute advantages ───────────────────────────────
        # Get value of last state for GAE
        with torch.no_grad():
            last_obs_norm = obs_normalizer.normalize(obs)
            last_obs_t = torch.tensor(
                last_obs_norm, dtype=torch.float32, device=args.device
            ).unsqueeze(0)
            _, last_value = policy(last_obs_t)
            last_value = last_value.item()

        advantages, returns = buffer.compute_returns(
            last_value, gamma=args.gamma, lam=args.lam,
        )

        # ── PPO update ───────────────────────────────────────
        policy.train()
        update_info = ppo_update(
            policy, optimizer, buffer, advantages, returns,
            clip_eps=args.clip_eps,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=args.entropy_coef,
            num_epochs=args.num_epochs,
            mini_batch_size=args.mini_batch_size,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
        )

        scheduler.step()
        iteration += 1

        # ── Logging ──────────────────────────────────────────
        mean_reward = (np.mean(episode_rewards) if episode_rewards
                       else 0.0)
        mean_ep_len = (np.mean(episode_lengths) if episode_lengths
                       else 0.0)

        elapsed = time.time() - start_time
        steps_per_sec = total_steps / max(elapsed, 1e-6)

        if iteration % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Iter {iteration:5d}] "
                f"steps={total_steps:>9,}  "
                f"reward={mean_reward:6.3f}  "
                f"ep_len={mean_ep_len:6.1f}  "
                f"p_loss={update_info['policy_loss']:.4f}  "
                f"v_loss={update_info['value_loss']:.4f}  "
                f"ent={update_info['entropy']:.4f}  "
                f"lr={lr:.2e}  "
                f"SPS={steps_per_sec:.0f}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar("reward/mean", mean_reward, total_steps)
                tb_writer.add_scalar("reward/ep_len", mean_ep_len, total_steps)
                tb_writer.add_scalar("loss/policy", update_info["policy_loss"], total_steps)
                tb_writer.add_scalar("loss/value", update_info["value_loss"], total_steps)
                tb_writer.add_scalar("loss/entropy", update_info["entropy"], total_steps)
                tb_writer.add_scalar("train/lr", lr, total_steps)
                tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, total_steps)

            episode_rewards.clear()
            episode_lengths.clear()

        # ── Checkpointing ────────────────────────────────────
        if iteration % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, policy, optimizer, obs_normalizer,
                             total_steps, iteration, mean_reward)
            print(f"  → Checkpoint: {ckpt_path}")

            # Best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_path = os.path.join(args.output_dir, "best_policy.pt")
                _save_checkpoint(best_path, policy, optimizer, obs_normalizer,
                                 total_steps, iteration, mean_reward)
                print(f"  → Best model updated (reward={mean_reward:.4f})")

    # ── Final save ───────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final_policy.pt")
    _save_checkpoint(final_path, policy, optimizer, obs_normalizer,
                     total_steps, iteration, mean_reward)
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"Total steps: {total_steps:,}  |  Iterations: {iteration}")
    print(f"Best reward: {best_mean_reward:.4f}")

    if tb_writer is not None:
        tb_writer.close()

    env.close()


def _save_checkpoint(path, policy, optimizer, obs_normalizer,
                     total_steps, iteration, mean_reward):
    """Save training checkpoint."""
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict(),
        "total_steps": total_steps,
        "iteration": iteration,
        "best_mean_reward": mean_reward,
    }, path)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO training for Newton motion tracking (prepare3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required paths
    parser.add_argument("--motion", required=True,
                        help="Path to .pkl motion file (MimicKit format)")
    parser.add_argument("--betas", required=True,
                        help="Path to .npy betas file (10,)")
    parser.add_argument("--output-dir", default="output/prepare3",
                        help="Output directory for checkpoints and logs")

    # Training
    parser.add_argument("--max-steps", type=int, default=10_000_000,
                        help="Total environment steps")
    parser.add_argument("--steps-per-iter", type=int, default=4096,
                        help="Steps per PPO iteration")
    parser.add_argument("--num-epochs", type=int, default=4,
                        help="PPO epochs per iteration")
    parser.add_argument("--mini-batch-size", type=int, default=256,
                        help="Mini-batch size for PPO update")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm for clipping")

    # Policy
    parser.add_argument("--hidden-dims", type=int, nargs="+",
                        default=[512, 256, 128],
                        help="MLP hidden layer sizes")
    parser.add_argument("--init-log-std", type=float, default=-1.0,
                        help="Initial log std for action distribution")

    # Environment
    parser.add_argument("--device", default="cuda:0",
                        help="Compute device")
    parser.add_argument("--sim-freq", type=int, default=480,
                        help="Physics simulation frequency (Hz)")
    parser.add_argument("--control-freq", type=int, default=30,
                        help="Control/policy frequency (Hz)")
    parser.add_argument("--control-mode", choices=["pd", "torque"],
                        default="pd",
                        help="Action interpretation mode")
    parser.add_argument("--max-ep-length", type=float, default=10.0,
                        help="Max episode length (seconds)")
    parser.add_argument("--no-early-termination", action="store_true",
                        help="Disable fall/divergence early termination")
    parser.add_argument("--no-rand-init", action="store_true",
                        help="Always start from t=0 (no random init)")
    parser.add_argument("--no-tar-obs", action="store_true",
                        help="Exclude target joint angles from observation")

    # Logging & checkpointing
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Log every N iterations")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
