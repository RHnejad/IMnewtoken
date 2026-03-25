"""
model.py — Physics-Informed RVQ-VAE (PhysicsRVQVAE).

Wraps InterMask's RVQVAE encoder/decoder/quantizer and adds
a differentiable physics simulation pass in the forward loop.

Architecture:
    Input motion → [Encoder] → [RVQ Codebook] → [Decoder] → x_hat
                                                              ↓
                                                    [decoder_to_joint_q]
                                                              ↓
                                                    [Newton PD Sim]
                                                              ↓
                                                    physics losses ← gradients
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.config import TrainingConfig, N_SMPL_JOINTS
from newton_vqvae.data_adapter import LightSAGEConv, MotionNormalizer
from newton_vqvae.newton_bridge import (
    DifferentiableNewtonSim,
    decoder_output_to_joint_q,
    gaussian_smooth_1d,
)
from newton_vqvae.physics_losses import PhysicsLoss, PhysicsLossScheduler
from newton_vqvae.skeleton_cache import SkeletonCache

# Import VQ-VAE components — use our local encdec (no torch_geometric)
from newton_vqvae.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
from models.losses import Geometric_Losses


class PhysicsRVQVAE(nn.Module):
    """
    Physics-Informed Residual VQ-VAE.

    Contains:
    - InterMask encoder/decoder/quantizer (kinematic backbone)
    - Newton simulation bridge (differentiable physics)
    - Physics loss module
    - Geometric loss module (from InterMask)

    Training modes:
    - 'kinematic': standard InterMask VQ-VAE training (warmup)
    - 'physics': kinematic + physics losses (main training)
    """

    def __init__(self, config: TrainingConfig, device: str = "cuda:0"):
        super().__init__()
        self.config = config
        self.device_str = device
        self.joints_num = 22
        self.dataset_name = "interhuman"

        # ── InterMask-compatible encoder/decoder/quantizer ──
        self._build_vqvae()

        # ── Physics modules (lazy-initialized on first use) ──
        self.skeleton_cache = SkeletonCache(device=device)
        self.newton_sim = DifferentiableNewtonSim(self.skeleton_cache, device=device)
        self.physics_loss = PhysicsLoss(config)
        self.physics_scheduler = PhysicsLossScheduler(
            warmup_epochs=config.physics_warmup_epochs,
            ramp_epochs=config.physics_ramp_epochs,
        )

        # ── Denormalization ──
        self.normalizer = MotionNormalizer()

        # ── Training state ──
        self.current_epoch = 0
        self.physics_enabled = False

    def _build_vqvae(self):
        """Build encoder, decoder, quantizer matching InterMask architecture."""
        cfg = self.config

        # InterMask-compatible args namespace
        class VQArgs:
            joints_num = 22
            dataset_name = "interhuman"
            num_quantizers = cfg.num_quantizers
            shared_codebook = False
            quantize_dropout_prob = cfg.quantize_dropout_prob
            mu = cfg.mu

        args = VQArgs()

        self.encoder = Encoder(
            args,
            input_emb_width=12,
            output_emb_width=cfg.code_dim,
            down_t=cfg.down_t,
            stride_t=cfg.stride_t,
            width=cfg.vq_width,
            depth=cfg.vq_depth,
            dilation_growth_rate=3,
            activation='relu',
            norm=None,
            filter_s=None,
            stride_s=None,
            gcn=True,
        )

        self.decoder = Decoder(
            args,
            input_emb_width=12,
            output_emb_width=cfg.code_dim,
            down_t=cfg.down_t,
            stride_t=cfg.stride_t,
            width=cfg.vq_width,
            depth=cfg.vq_depth,
            dilation_growth_rate=3,
            activation='relu',
            norm=None,
            spatial_upsample=(2.2, 2),
            gcn=True,
        )

        self.quantizer = ResidualVQ(
            num_quantizers=cfg.num_quantizers,
            shared_codebook=False,
            nb_code=cfg.nb_code,
            code_dim=cfg.code_dim,
            args=args,
            quantize_dropout_prob=cfg.quantize_dropout_prob,
            quantize_dropout_cutoff_index=0,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 262) → (B, 12, 22, T) matching InterMask."""
        pos = x[..., :self.joints_num * 3].reshape(
            x.shape[0], x.shape[1], self.joints_num, 3
        )
        vel = x[..., self.joints_num * 3:self.joints_num * 6].reshape(
            x.shape[0], x.shape[1], self.joints_num, 3
        )
        rot = x[..., self.joints_num * 6:self.joints_num * 6 + (self.joints_num - 1) * 6]
        rot = rot.reshape(x.shape[0], x.shape[1], self.joints_num - 1, 6)
        rot = torch.cat([
            torch.zeros(rot.shape[0], rot.shape[1], 1, 6, device=x.device),
            rot,
        ], dim=2)

        joints = torch.cat([pos, vel, rot], dim=-1)  # (B, T, 22, 12)
        return joints.permute(0, 3, 2, 1).float()  # (B, 12, 22, T)

    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 12, 22, T) → (B, T, 262) matching InterMask."""
        x = x.permute(0, 3, 2, 1).float()  # (B, T, 22, 12)
        pos = x[:, :, :, :3].reshape(x.shape[0], x.shape[1], -1)
        vel = x[:, :, :, 3:6].reshape(x.shape[0], x.shape[1], -1)
        rot = x[:, :, 1:, 6:12].reshape(x.shape[0], x.shape[1], -1)
        fc = torch.zeros(x.shape[0], x.shape[1], 4, device=x.device)
        return torch.cat([pos, vel, rot, fc], dim=-1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode motion to codebook indices."""
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        enc_shape = x_enc.shape
        x_enc = x_enc if len(enc_shape) == 3 else x_enc.reshape(
            enc_shape[0], enc_shape[1], -1
        )
        code_idx, all_codes = self.quantizer.quantize(x_enc, return_latent=True)
        return code_idx, all_codes

    def decode(self, x_quantized: torch.Tensor, enc_shape) -> torch.Tensor:
        """Decode from quantized representation."""
        x_quantized = x_quantized.reshape(enc_shape)
        x_out = self.decoder(x_quantized)
        return self.postprocess(x_out)

    def forward(
        self,
        x: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
        run_physics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: (B, T, 262) normalized motion
            betas: (B, 10) SMPL-X body shape params (needed for physics)
            run_physics: whether to run physics simulation step

        Returns:
            dict with:
                'x_hat': (B, T, 262) reconstructed motion
                'commit_loss': commitment loss
                'perplexity': codebook usage
                'physics': optional physics results dict
        """
        B, T, D = x.shape

        # ── Encode ──
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        enc_shape = x_enc.shape
        x_enc_flat = x_enc if len(enc_shape) == 3 else x_enc.reshape(
            enc_shape[0], enc_shape[1], -1
        )

        # ── Quantize ──
        x_quant, code_idx, commit_loss, perplexity = self.quantizer(
            x_enc_flat, sample_codebook_temp=0.5
        )
        x_quant = x_quant.reshape(enc_shape)

        # ── Decode ──
        x_out = self.decoder(x_quant)
        x_hat = self.postprocess(x_out)

        result = {
            'x_hat': x_hat,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

        # ── Physics pass (optional) ──
        if run_physics and betas is not None:
            physics_result = self._physics_forward(x_hat, betas)
            result['physics'] = physics_result

        return result

    def _physics_forward(
        self,
        x_hat: torch.Tensor,
        betas: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run decoded motion through Newton physics simulation.

        Args:
            x_hat: (B, T, 262) decoded (still normalized) motion
            betas: (B, 10) body shape params

        Returns:
            dict with sim_positions, pd_torques, root_forces, body_positions,
            target_positions (FK-derived)
        """
        B, T, _ = x_hat.shape
        device = x_hat.device

        # Denormalize to get real-world positions
        x_denorm = self.normalizer.backward(x_hat)

        # Extract target FK positions from decoded motion
        target_pos = x_denorm[..., :self.joints_num * 3].reshape(
            B, T, self.joints_num, 3
        )

        # Process each sample in batch
        all_sim_pos = []
        all_torques = []
        all_root_forces = []
        all_body_pos = []

        for b in range(B):
            # Get body offset from betas (numpy for skeleton_cache)
            betas_np = betas[b].detach().cpu().numpy()
            offset_np = self.skeleton_cache.get_body_offset(betas_np)
            offset_t = torch.from_numpy(offset_np).float().to(device)

            # Convert decoder output to Newton joint_q
            joint_q = decoder_output_to_joint_q(
                x_denorm[b:b + 1], betas[b:b + 1], offset_t.unsqueeze(0)
            )  # (1, T, 76)

            # Optional temporal smoothing
            smooth_sigma = getattr(self.config, 'smooth_sigma', 0.0)
            if smooth_sigma > 0:
                joint_q_smooth = gaussian_smooth_1d(
                    joint_q[0], sigma=smooth_sigma
                )
                joint_q = joint_q_smooth.unsqueeze(0)

            # Get Newton model for this body shape
            model = self.skeleton_cache.get_model(betas_np)

            # Run simulation
            sim_result = self.newton_sim.simulate_single(
                model, joint_q[0], n_frames=T
            )

            all_sim_pos.append(sim_result['sim_positions'])
            all_torques.append(sim_result['pd_torques'])
            all_root_forces.append(sim_result['root_forces'])
            all_body_pos.append(sim_result['body_positions'])

        return {
            'sim_positions': torch.stack(all_sim_pos, dim=0),
            'pd_torques': torch.stack(all_torques, dim=0),
            'root_forces': torch.stack(all_root_forces, dim=0),
            'body_positions': torch.stack(all_body_pos, dim=0),
            'target_positions': target_pos,
        }

    def compute_losses(
        self,
        x: torch.Tensor,
        result: Dict[str, torch.Tensor],
        geo_losses: Geometric_Losses,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined kinematic + physics losses.

        Args:
            x: (B, T, 262) ground truth motion
            result: output from forward()
            geo_losses: InterMask's Geometric_Losses instance

        Returns:
            total_loss: scalar
            loss_log: dict for logging
        """
        x_hat = result['x_hat']
        commit_loss = result['commit_loss']

        # ── Kinematic losses (from InterMask) ──
        loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, _, _ = (
            geo_losses.forward(x, x_hat)
        )

        kinematic_loss = (
            loss_rec
            + 0.02 * commit_loss
            + 1.0 * loss_explicit
            + 100.0 * loss_vel
            + 5.0 * loss_bn
            + 0.01 * loss_geo
            + 500.0 * loss_fc
        )

        loss_log = {
            'loss_rec': loss_rec.item(),
            'loss_explicit': loss_explicit.item(),
            'loss_vel': loss_vel.item(),
            'loss_bn': loss_bn.item(),
            'loss_geo': loss_geo.item(),
            'loss_fc': loss_fc.item(),
            'loss_commit': commit_loss.item(),
            'loss_kinematic': kinematic_loss.item(),
        }

        # ── Physics losses ──
        physics_weight = self.physics_scheduler.get_weight(self.current_epoch)
        loss_log['physics_weight'] = physics_weight

        if 'physics' in result and physics_weight > 0:
            physics_loss, physics_log = self.physics_loss(
                result['physics'],
                result['physics']['target_positions'],
            )
            total_loss = kinematic_loss + physics_weight * physics_loss

            loss_log.update({k: v.item() for k, v in physics_log.items()})
            loss_log['loss_physics_weighted'] = (physics_weight * physics_loss).item()
        else:
            total_loss = kinematic_loss

        loss_log['loss_total'] = total_loss.item()
        return total_loss, loss_log

    def load_intermask_checkpoint(self, ckpt_path: str, strict: bool = False):
        """
        Load weights from a pre-trained InterMask VQ-VAE checkpoint.

        Maps InterMask's RVQVAE state_dict to this model's encoder/decoder/quantizer.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # InterMask saves under 'vq_model' key in the checkpoint
        if 'vq_model' in ckpt:
            state_dict = ckpt['vq_model']
        elif 'net' in ckpt:
            state_dict = ckpt['net']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Filter to encoder/decoder/quantizer keys
        mapped = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if DDP
            k = k.replace('module.', '')
            if k.startswith(('encoder.', 'decoder.', 'quantizer.')):
                mapped[k] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)

        print(f"[PhysicsRVQVAE] Loaded InterMask checkpoint: {ckpt_path}")
        print(f"  Matched: {len(mapped)} params")
        if missing:
            print(f"  Missing: {len(missing)} ({missing[:5]}...)")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)} ({unexpected[:5]}...)")

    def set_epoch(self, epoch: int):
        """Update current epoch for physics loss scheduling."""
        self.current_epoch = epoch
        self.physics_enabled = self.physics_scheduler.get_weight(epoch) > 0
