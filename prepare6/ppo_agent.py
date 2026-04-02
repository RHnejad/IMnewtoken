"""
ppo_agent.py — PPO agent (pure PyTorch, no external RL library).

Implements the standard CleanRL-style PPO update:
  - PolicyNet: MLP with fixed log_std (ProtoMotions default: -2.9)
  - ValueNet:  MLP critic
  - PPO update with separate actor/critic optimizers
  - GAE advantage estimation
  - Orthogonal weight initialisation

References:
  - ProtoMotions ppo/config.py: actor_optimizer.lr=2e-5, critic_optimizer.lr=1e-4
  - ProtoMotions ppo/config.py: e_clip=0.2, tau=0.95, entropy_coef=0.005
  - CleanRL ppo_continuous_action.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from prepare6.rl_config import (
    CLIP_EPS, ENTROPY_COEF, VALUE_COEF,
    LR_ACTOR, LR_CRITIC,
    GAMMA, GAE_LAMBDA,
    N_STEPS, N_EPOCHS, MINIBATCH_SIZE,
    POLICY_HIDDEN, VALUE_HIDDEN,
    ACTOR_LOGSTD,
)


def _ortho_init(layer, scale=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=scale)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _make_mlp(in_dim, hidden_dims, out_dim, out_scale=1.0, activation=nn.ELU):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [_ortho_init(nn.Linear(prev, h)), activation()]
        prev = h
    layers += [_ortho_init(nn.Linear(prev, out_dim), scale=out_scale)]
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mu = _make_mlp(obs_dim, POLICY_HIDDEN, act_dim, out_scale=0.01)
        self.log_std = nn.Parameter(
            torch.ones(act_dim) * ACTOR_LOGSTD, requires_grad=False
        )

    def forward(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def get_action(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def evaluate(self, obs, action):
        dist = self.forward(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = _make_mlp(obs_dim, VALUE_HIDDEN, 1, out_scale=1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPOAgent:
    """PPO agent with separate actor/critic optimizers."""

    def __init__(self, obs_dim, act_dim, device):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.policy = PolicyNet(obs_dim, act_dim).to(device)
        self.value  = ValueNet(obs_dim).to(device)

        self.actor_opt  = torch.optim.Adam(self.policy.parameters(), lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.value.parameters(),  lr=LR_CRITIC)

        self._update_count = 0

    # ── Rollout collection ────────────────────────────────────────────────────

    def collect_rollout(self, env, n_steps=N_STEPS):
        """Collect a rollout of n_steps from all envs.

        Returns rollout_buffer dict with keys:
          obs, actions, log_probs, rewards, dones, values
          each shape (n_steps, n_envs, ...) or (n_steps, n_envs)
        """
        n_envs = env.n_envs

        obs_list      = []
        act_list      = []
        lp_list       = []
        rew_list      = []
        done_list     = []
        val_list      = []

        obs = env._get_obs()   # (n_envs, obs_dim)

        for _ in range(n_steps):
            obs_t = torch.FloatTensor(obs).to(self.device)

            with torch.no_grad():
                actions, log_probs = self.policy.get_action(obs_t)
                values = self.value(obs_t)

            acts_np = actions.cpu().numpy()
            next_obs, rewards, dones, _ = env.step(acts_np)

            obs_list.append(obs)
            act_list.append(acts_np)
            lp_list.append(log_probs.cpu().numpy())
            rew_list.append(rewards)
            done_list.append(dones.astype(np.float32))
            val_list.append(values.cpu().numpy())

            obs = next_obs

        # Final value for bootstrapping
        with torch.no_grad():
            last_val = self.value(
                torch.FloatTensor(obs).to(self.device)
            ).cpu().numpy()

        return {
            'obs':      np.stack(obs_list),       # (T, N, obs_dim)
            'actions':  np.stack(act_list),       # (T, N, act_dim)
            'log_probs': np.stack(lp_list),       # (T, N)
            'rewards':  np.stack(rew_list),       # (T, N)
            'dones':    np.stack(done_list),      # (T, N)
            'values':   np.stack(val_list),       # (T, N)
            'last_val': last_val,                 # (N,)
        }

    # ── GAE ───────────────────────────────────────────────────────────────────

    def compute_gae(self, rewards, values, dones, last_val):
        """Compute GAE advantages and returns.

        Args:
            rewards: (T, N)
            values:  (T, N)
            dones:   (T, N)  1.0 if done
            last_val: (N,)

        Returns:
            advantages: (T, N)
            returns:    (T, N)
        """
        T, N = rewards.shape
        advantages = np.zeros_like(rewards)
        gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_val
            else:
                next_val = values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self, rollout):
        """Run N_EPOCHS of minibatch PPO updates.

        Returns:
            metrics dict with mean_reward, actor_loss, critic_loss, entropy
        """
        obs       = rollout['obs']         # (T, N, obs_dim)
        actions   = rollout['actions']     # (T, N, act_dim)
        old_lp    = rollout['log_probs']   # (T, N)
        rewards   = rollout['rewards']     # (T, N)
        dones     = rollout['dones']       # (T, N)
        values    = rollout['values']      # (T, N)
        last_val  = rollout['last_val']    # (N,)

        adv, returns = self.compute_gae(rewards, values, dones, last_val)

        # Flatten (T, N) → (T*N,)
        T, N = rewards.shape
        obs_f     = torch.FloatTensor(obs.reshape(T*N, -1)).to(self.device)
        acts_f    = torch.FloatTensor(actions.reshape(T*N, -1)).to(self.device)
        old_lp_f  = torch.FloatTensor(old_lp.reshape(T*N)).to(self.device)
        adv_f     = torch.FloatTensor(adv.reshape(T*N)).to(self.device)
        ret_f     = torch.FloatTensor(returns.reshape(T*N)).to(self.device)

        # Normalise advantages
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        n_updates = 0

        idx = np.arange(T * N)
        for _ in range(N_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, T * N, MINIBATCH_SIZE):
                mb = idx[start:start + MINIBATCH_SIZE]

                new_lp, entropy = self.policy.evaluate(obs_f[mb], acts_f[mb])
                ratio = (new_lp - old_lp_f[mb]).exp()

                adv_mb = adv_f[mb]
                pg_loss = torch.max(
                    -adv_mb * ratio,
                    -adv_mb * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS),
                ).mean()

                val_pred = self.value(obs_f[mb])
                critic_loss = F.mse_loss(val_pred, ret_f[mb])

                actor_loss  = pg_loss - ENTROPY_COEF * entropy.mean()
                total_loss  = actor_loss + VALUE_COEF * critic_loss

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.value.parameters(),  1.0)
                self.actor_opt.step()
                self.critic_opt.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates += 1

        self._update_count += 1

        return {
            'mean_reward':  float(rewards.mean()),
            'actor_loss':   total_actor_loss  / max(n_updates, 1),
            'critic_loss':  total_critic_loss / max(n_updates, 1),
            'entropy':      total_entropy     / max(n_updates, 1),
        }

    # ── Deterministic action ──────────────────────────────────────────────────

    @torch.no_grad()
    def act_deterministic(self, obs_np):
        """Return mean action (no sampling) for evaluation.

        Args:
            obs_np: (N, obs_dim) numpy array
        Returns:
            actions: (N, act_dim) numpy array
        """
        obs_t = torch.FloatTensor(obs_np).to(self.device)
        dist = self.policy(obs_t)
        return dist.mean.cpu().numpy()

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value':  self.value.state_dict(),
            'actor_opt':  self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'update_count': self._update_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self._update_count = ckpt.get('update_count', 0)
