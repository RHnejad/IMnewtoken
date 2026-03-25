"""
train.py — Training loop for Physics-Informed RVQ-VAE.

Usage:
    conda activate mimickit
    python -m newton_vqvae.train \
        --data_root ../interhuman_data \
        --output_dir ./outputs/physics_vqvae \
        --intermask_ckpt ./checkpoints/interhuman/finest.tar \
        --batch_size 32 \
        --max_epoch 50

Two-phase training:
    Phase 1 (per-character):
        Kinematic warmup (5 epochs) → ramp physics losses → full training
    Phase 2 (pair interaction):
        Both characters in same Newton sim (future extension)
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict, defaultdict
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.config import TrainingConfig, make_config_from_args
from newton_vqvae.data_adapter import InterHumanPhysicsDataset
from newton_vqvae.model import PhysicsRVQVAE
from models.losses import Geometric_Losses


def def_value():
    return 0.0


class PhysicsVQTrainer:
    """Training loop for PhysicsRVQVAE."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        # ── Model ──
        self.model = PhysicsRVQVAE(config, device=config.device)
        self.model.to(self.device)

        # ── Load InterMask pre-trained weights (warm start) ──
        ckpt_path = config.intermask_ckpt or config.pretrained_ckpt
        if ckpt_path and os.path.exists(ckpt_path):
            self.model.load_intermask_checkpoint(ckpt_path)
            print(f"[Train] Loaded InterMask checkpoint: {ckpt_path}")

        # ── Geometric losses (InterMask) ──
        self.geo_losses = Geometric_Losses(
            'l1_smooth',
            joints_num=22,
            dataset_name='interhuman',
            device=self.device,
        )

        # ── Optimizer ──
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )

        # ── Directories ──
        out = config.output_dir or config.model_dir or './outputs/newton_vqvae'
        os.makedirs(out, exist_ok=True)
        self.model_dir = pjoin(out, 'models') if not config.model_dir else config.model_dir
        self.log_dir = pjoin(out, 'logs') if not config.log_dir else config.log_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = SummaryWriter(self.log_dir)

    def build_dataloaders(self):
        """Build train/val dataloaders."""
        train_dataset = InterHumanPhysicsDataset(
            data_root=self.config.data_root,
            mode='train',
            window_size=self.config.window_size,
            window_stride=self.config.window_stride,
        )
        val_dataset = InterHumanPhysicsDataset(
            data_root=self.config.data_root,
            mode='val',
            window_size=self.config.window_size,
            window_stride=self.config.window_size,  # No overlap for val
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return train_loader, val_loader

    def forward_step(self, batch, run_physics: bool = False):
        """
        Single forward step.

        Args:
            batch: dict with 'motion', 'betas', 'clip_id'
            run_physics: whether to run physics sim

        Returns:
            total_loss, loss_log dict
        """
        motion = batch['motion'].to(self.device).float()
        betas = batch.get('betas')
        if betas is not None:
            betas = betas.to(self.device).float()

        # Forward pass
        result = self.model(
            motion,
            betas=betas,
            run_physics=run_physics and betas is not None,
        )

        # Compute losses
        total_loss, loss_log = self.model.compute_losses(
            motion, result, self.geo_losses
        )

        return total_loss, loss_log

    def save(self, file_name: str, epoch: int, total_it: int):
        """Save checkpoint."""
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'total_it': total_it,
            'config': {
                'lr': self.config.lr,
                'batch_size': self.config.batch_size,
                'code_dim': self.config.code_dim,
                'nb_code': self.config.nb_code,
            },
        }
        torch.save(state, file_name)

    def resume(self, model_dir: str):
        """Resume from checkpoint."""
        ckpt = torch.load(model_dir, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt['epoch'], ckpt['total_it']

    def train(self):
        """Main training loop."""
        train_loader, val_loader = self.build_dataloaders()

        total_iters = self.config.max_epoch * len(train_loader)
        print(f"\n{'=' * 60}")
        print(f"Physics-Informed VQ-VAE Training")
        print(f"{'=' * 60}")
        print(f"Total Epochs: {self.config.max_epoch}")
        print(f"Total Iters: {total_iters}")
        print(f"Train batches/epoch: {len(train_loader)}")
        print(f"Val batches/epoch: {len(val_loader)}")
        print(f"Physics warmup: {self.config.physics_warmup_epochs} epochs")
        print(f"Physics ramp: {getattr(self.config, 'physics_ramp_epochs', 10)} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.lr}")
        print(f"{'=' * 60}\n")

        # LR scheduler
        warm_up_iter = len(train_loader) // 4
        milestones = [int(total_iters * 0.7), int(total_iters * 0.85)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1
        )

        log_every = max(len(train_loader) // 10, 1)
        save_every = max(len(train_loader) // 2, 1)

        epoch = 0
        it = 0
        start_time = time.time()
        logs = defaultdict(def_value, OrderedDict())
        min_val_loss = np.inf

        while epoch < self.config.max_epoch:
            epoch += 1
            self.model.set_epoch(epoch)
            self.model.train()

            # Determine if physics should run this epoch
            run_physics = self.model.physics_enabled
            if run_physics:
                print(f"[Epoch {epoch}] Physics ENABLED "
                      f"(weight={self.model.physics_scheduler.get_weight(epoch):.3f})")
            else:
                print(f"[Epoch {epoch}] Kinematic-only (physics warmup)")

            for i, batch in enumerate(train_loader):
                it += 1

                # LR warmup
                if it < warm_up_iter:
                    current_lr = self.config.lr * (it + 1) / (warm_up_iter + 1)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = current_lr

                # Forward
                # Physics sim is expensive — only run on subset of batches
                do_physics = run_physics and (i % self.config.physics_every_n_batches == 0) if hasattr(self.config, 'physics_every_n_batches') else False

                loss, loss_log = self.forward_step(batch, run_physics=do_physics)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                if it >= warm_up_iter:
                    scheduler.step()

                # Accumulate logs
                for k, v in loss_log.items():
                    logs[k] += v
                logs['lr'] += self.optimizer.param_groups[0]['lr']

                # Print
                if it % log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar(f'Train/{tag}', value / log_every, it)
                        mean_loss[tag] = value / log_every
                    logs = defaultdict(def_value, OrderedDict())

                    elapsed = time.time() - start_time
                    print(f"  [Ep {epoch}, It {it}/{total_iters}] "
                          f"loss={mean_loss.get('loss_total', 0):.4f} "
                          f"rec={mean_loss.get('loss_rec', 0):.4f} "
                          f"phys={mean_loss.get('loss_physics_weighted', 0):.4f} "
                          f"lr={mean_loss.get('lr', 0):.6f} "
                          f"({elapsed:.0f}s)")

                # Save latest
                if it % save_every == 0:
                    self.save(pjoin(self.model_dir, 'latest.tar'), epoch, it)

            # End of epoch — save
            self.save(pjoin(self.model_dir, 'latest.tar'), epoch, it)

            # ── Validation ──
            print(f"  Validation (epoch {epoch})...")
            self.model.eval()
            val_losses = defaultdict(list)

            with torch.no_grad():
                for batch in val_loader:
                    loss, loss_log = self.forward_step(batch, run_physics=False)
                    for k, v in loss_log.items():
                        val_losses[k].append(v)

            val_mean = {k: np.mean(v) for k, v in val_losses.items()}
            for k, v in val_mean.items():
                self.logger.add_scalar(f'Val/{k}', v, epoch)

            print(f"  Val loss={val_mean.get('loss_total', 0):.4f} "
                  f"rec={val_mean.get('loss_rec', 0):.4f}")

            if val_mean.get('loss_total', np.inf) < min_val_loss:
                min_val_loss = val_mean['loss_total']
                self.save(pjoin(self.model_dir, 'best.tar'), epoch, it)
                print("  >> Best validation model saved!")

        print(f"\nTraining complete. Models saved to {self.model_dir}")
        self.logger.close()


def main():
    config = make_config_from_args()
    trainer = PhysicsVQTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
