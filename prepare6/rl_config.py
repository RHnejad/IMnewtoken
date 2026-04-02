"""
rl_config.py — PPO RL tracker hyperparameters.

Extends prepare5/phc_config.py with RL-specific settings.
All gains, body counts, and simulation parameters come from phc_config.
"""
from prepare5.phc_config import *   # noqa: F401,F403

# ═══════════════════════════════════════════════════════════════
# Vectorized environment
# ═══════════════════════════════════════════════════════════════
N_ENVS = 256        # parallel envs in one Newton model (use 4 for smoke test)

# ═══════════════════════════════════════════════════════════════
# PPO hyperparameters (matched to ProtoMotions defaults)
# ═══════════════════════════════════════════════════════════════
N_STEPS        = 32          # rollout steps per PPO update
N_EPOCHS       = 5           # epochs of minibatch updates per rollout
MINIBATCH_SIZE = 512
CLIP_EPS       = 0.2         # e_clip
ENTROPY_COEF   = 0.005
VALUE_COEF     = 0.5
LR_ACTOR       = 2e-5        # ProtoMotions actor_optimizer.lr
LR_CRITIC      = 1e-4        # ProtoMotions critic_optimizer.lr
GAMMA          = 0.95
GAE_LAMBDA     = 0.95        # tau in ProtoMotions
ACTOR_LOGSTD   = -2.9        # ProtoMotions default (fixed, not learnable)

TOTAL_TIMESTEPS   = 500_000  # ~5 min per clip on RTX 3090 with N_ENVS=256
EARLY_STOP_REWARD = 0.85     # stop early if mean reward exceeds this
REWARD_WINDOW     = 20       # number of updates for early-stop smoothing

# ═══════════════════════════════════════════════════════════════
# Observation / action dimensions
# ═══════════════════════════════════════════════════════════════
# 22 SMPL joints × (3 pos + 4 rot + 3 vel + 3 ang_vel) + 1 height + 3 grav
# + 22 × (3 ref_pos + 4 ref_rot) + 2 phase = 22*13 + 4 + 22*7 + 2 = 286+4+154+2 = 446
OBS_DIM_SOLO   = 446
OBS_DIM_PAIRED = 892         # two-person: concatenate both obs vectors
ACT_DIM_SOLO   = 69          # hinge DOFs only (23 bodies × 3, no root)
ACT_DIM_PAIRED = 138

# ═══════════════════════════════════════════════════════════════
# Action parameterization
# ═══════════════════════════════════════════════════════════════
ACTION_SCALE = 0.5       # target = ref_q[7:] + tanh(action) * ACTION_SCALE * pi
RSI_PROB     = 0.9       # Reference State Initialization: random-frame reset prob

# ═══════════════════════════════════════════════════════════════
# Network architecture
# ═══════════════════════════════════════════════════════════════
POLICY_HIDDEN = [512, 256, 128]
VALUE_HIDDEN  = [512, 256, 128]
