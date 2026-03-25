"""
Configuration for Newton-VQ: Physics-Informed VQ-VAE.

Centralises all hyper-parameters so that every module imports from here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import argparse


# ═══════════════════════════════════════════════════════════════
# Physics simulation constants
# ═══════════════════════════════════════════════════════════════
SIM_FREQ = 480           # Hz  (physics substeps)
MOTION_FPS = 30          # Hz  (motion playback)
SIM_SUBSTEPS = SIM_FREQ // MOTION_FPS  # 16

TORQUE_LIMIT = 1000.0    # Nm  (max PD torque)
GRAD_CHECKPOINT_FRAMES = 8  # wp.Tape checkpoint interval

# ═══════════════════════════════════════════════════════════════
# Skeleton constants  (shared with prepare2/pd_utils)
# ═══════════════════════════════════════════════════════════════
DOFS_PER_PERSON = 75
COORDS_PER_PERSON = 76
BODIES_PER_PERSON = 24
N_SMPL_JOINTS = 22
N_JOINT_Q = 76


@dataclass
class PhysicsLossWeights:
    """Weights for the Newtonian loss terms."""
    alpha:  float = 10.0    # FK-MPJPE (primary tracking)
    beta:   float = 0.001   # Torque magnitude
    gamma:  float = 1.0     # Log-space skyhook
    delta:  float = 50.0    # Penetration + foot slide
    epsilon: float = 5.0    # Zero-moment-point stability

    # Sub-weights inside SoftFlow
    pen_weight:      float = 1.0     # Penetration sub-weight
    slide_weight:    float = 1.0     # Foot sliding sub-weight

    # Sigmoid steepness for soft contact detection
    contact_sigmoid_k: float = 100.0
    contact_threshold: float = 0.02  # m  (2 cm)


@dataclass
class TrainingConfig:
    """Training-loop hyper-parameters."""
    # Data
    dataset_name:     str   = "interhuman"
    data_dir:         str   = "data/InterHuman"
    batch_size:       int   = 64       # Reduced from 256 (Newton memory)
    window_size:      int   = 64
    window_stride:    int   = 10
    num_workers:      int   = 4
    cache:            bool  = True

    # Optimiser  (InterMask defaults)
    max_epoch:        int   = 50
    lr:               float = 2e-4
    weight_decay:     float = 0.0
    gamma:            float = 0.1      # LR step decay
    grad_clip:        float = 1.0

    # InterMask kinematic-loss weights (unchanged)
    commit:           float = 0.02
    loss_explicit:    float = 1.0
    loss_vel:         float = 100.0
    loss_bn:          float = 5.0
    loss_geo:         float = 0.01
    loss_fc:          float = 500.0

    # Physics loss schedule
    physics_warmup_epochs: int   = 5    # Low-weight warm-up
    physics_ramp_epochs:   int   = 10   # Linear ramp epochs
    physics_warmup_scale:  float = 0.1  # 10 % weight during warm-up
    physics_cooldown_epoch: int  = 40   # Start physics LR cooldown
    physics_cooldown_gamma: float = 0.1

    # Physics-per-batch frequency (1 = every batch)
    physics_every_n_batches: int = 1

    # Temporal smooth sigma for decoded joint angles
    smooth_sigma: float = 1.0  # Gaussian σ (frames)

    # Torque limit for loss clamping
    torque_limit_loss: float = 1000.0

    # Skyhook log-space + outlier clipping
    skyhook_log_space:     bool  = True
    skyhook_clip_factor:   float = 3.0  # c · median

    # Phase 2-specific
    phase2:              bool  = False
    interaction_force_w: float = 1.0
    dm_loss_w:           float = 1.0
    ro_loss_w:           float = 1.0

    # VQ-VAE architecture (InterMask defaults)
    code_dim:            int   = 512
    nb_code:             int   = 1024
    mu:                  float = 0.99
    down_t:              int   = 2
    stride_t:            int   = 2
    width:               int   = 512
    depth:               int   = 2
    vq_width:            int   = 512   # alias for width
    vq_depth:            int   = 2     # alias for depth
    dilation_growth_rate: int  = 3
    output_emb_width:    int   = 512
    vq_act:              str   = "relu"
    vq_norm:             Optional[str] = None
    num_quantizers:      int   = 1
    shared_codebook:     bool  = False
    quantize_dropout_prob: float = 0.2

    joints_num:          int   = 22

    # Warm-start from InterMask kinematic checkpoint
    pretrained_ckpt:     Optional[str] = None
    intermask_ckpt:      Optional[str] = None  # alias for pretrained_ckpt

    # Device
    gpu_id:              int   = 0
    device:              str   = "cuda:0"

    # Output
    output_dir:          str   = "./outputs/newton_vqvae"
    data_root:           str   = ""  # resolved from data_dir

    # Logging / saving
    name:                str   = "newton_vq_phase1"
    checkpoints_dir:     str   = "./checkpoints"
    log_dir:             str   = ""   # Filled at init
    model_dir:           str   = ""   # Filled at init
    eval_dir:            str   = ""   # Filled at init
    do_eval:             bool  = False

    # Physics
    physics_weights: PhysicsLossWeights = field(default_factory=PhysicsLossWeights)

    def setup_dirs(self):
        import os
        expr = os.path.join(self.checkpoints_dir, self.dataset_name, self.name)
        self.model_dir = os.path.join(expr, "models")
        self.log_dir = os.path.join(expr, "logs")
        self.eval_dir = os.path.join(expr, "eval")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        # Resolve aliases
        if not self.data_root:
            self.data_root = self.data_dir
        if self.intermask_ckpt is None:
            self.intermask_ckpt = self.pretrained_ckpt
        if not self.output_dir or self.output_dir == "./outputs/newton_vqvae":
            self.output_dir = expr


def make_config_from_args() -> TrainingConfig:
    """Parse CLI into a TrainingConfig."""
    p = argparse.ArgumentParser("Newton-VQ Training")

    # Expose the most-tuned knobs; everything else uses dataclass defaults
    p.add_argument("--dataset_name", default="interhuman")
    p.add_argument("--data_dir", default="data/InterHuman")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--name", default="newton_vq_phase1")
    p.add_argument("--pretrained_ckpt", default=None)
    p.add_argument("--phase2", action="store_true")
    p.add_argument("--physics_every_n_batches", type=int, default=1)
    p.add_argument("--do_eval", action="store_true")

    # Physics weights
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--gamma_skyhook", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=50.0)
    p.add_argument("--epsilon", type=float, default=5.0)
    p.add_argument("--smooth_sigma", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--intermask_ckpt", type=str, default=None)
    p.add_argument("--physics_warmup_epochs", type=int, default=5)
    p.add_argument("--physics_ramp_epochs", type=int, default=10)
    p.add_argument("--window_size", type=int, default=64)

    args = p.parse_args()

    cfg = TrainingConfig(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        gpu_id=args.gpu_id,
        device=f"cuda:{args.gpu_id}",
        name=args.name,
        pretrained_ckpt=args.pretrained_ckpt or args.intermask_ckpt,
        intermask_ckpt=args.intermask_ckpt,
        phase2=args.phase2,
        physics_every_n_batches=args.physics_every_n_batches,
        do_eval=args.do_eval,
        smooth_sigma=args.smooth_sigma,
        output_dir=args.output_dir,
        physics_warmup_epochs=args.physics_warmup_epochs,
        physics_ramp_epochs=args.physics_ramp_epochs,
        window_size=args.window_size,
        physics_weights=PhysicsLossWeights(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma_skyhook,
            delta=args.delta,
            epsilon=args.epsilon,
        ),
    )
    cfg.setup_dirs()
    return cfg
