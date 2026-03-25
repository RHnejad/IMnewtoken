"""
evaluate.py — Physics-quality evaluation for trained PhysicsRVQVAE.

Metrics computed:
    1. FK-MPJPE: Forward kinematics mean per-joint position error (mm)
    2. Torque RMS: Root-mean-square joint torques (Nm)
    3. Skyhook Force: Mean root support force magnitude (N)
    4. Penetration Rate: % of frames with feet below ground
    5. Foot Sliding: Mean horizontal foot speed during contact (m/s)
    6. ZMP Violation: Mean ZMP distance outside support polygon (m)
    7. Reconstruction L1: Standard VQ-VAE reconstruction loss
    8. Codebook Perplexity: Effective codebook usage

Usage:
    python -m newton_vqvae.evaluate \
        --checkpoint ./outputs/physics_vqvae/models/best.tar \
        --data_root ../interhuman_data \
        --output_file ./outputs/physics_vqvae/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from newton_vqvae.config import TrainingConfig
from newton_vqvae.data_adapter import InterHumanPhysicsDataset
from newton_vqvae.model import PhysicsRVQVAE
from newton_vqvae.physics_losses import (
    loss_fk_mpjpe, loss_torque, loss_skyhook, loss_softflow, loss_zmp,
)


FOOT_INDICES = (7, 8, 10, 11)


def compute_physics_metrics(
    sim_result: dict,
    target_positions: torch.Tensor,
    ground_height: float = 0.0,
    contact_threshold: float = 0.02,
) -> dict:
    """
    Compute all physics quality metrics from a simulation result.

    Args:
        sim_result: output from DifferentiableNewtonSim
        target_positions: (T, J, 3) FK target positions
        ground_height: ground Y coordinate
        contact_threshold: foot contact height threshold

    Returns:
        dict of metric name → scalar value
    """
    sim_pos = sim_result['sim_positions']
    torques = sim_result['pd_torques']
    root_f = sim_result['root_forces']
    body_pos = sim_result['body_positions']

    metrics = {}

    # 1. FK-MPJPE (millimeters)
    diff = sim_pos - target_positions
    mpjpe = diff.norm(dim=-1).mean().item() * 1000.0  # meters → mm
    metrics['fk_mpjpe_mm'] = mpjpe

    # 2. Torque RMS (Nm), excluding root DOFs
    body_torques = torques[..., 6:]
    torque_rms = body_torques.pow(2).mean().sqrt().item()
    metrics['torque_rms_nm'] = torque_rms

    # 3. Skyhook force magnitude (N)
    root_trans_force = root_f[..., :3].norm(dim=-1).mean().item()
    root_rot_torque = root_f[..., 3:6].norm(dim=-1).mean().item()
    metrics['skyhook_trans_n'] = root_trans_force
    metrics['skyhook_rot_nm'] = root_rot_torque
    metrics['skyhook_total'] = root_trans_force + root_rot_torque

    # 4. Penetration rate
    if body_pos.dim() == 3:  # (T, B, 3)
        feet_h = body_pos[:, FOOT_INDICES, 1]  # (T, 4)
    else:
        feet_h = body_pos[..., FOOT_INDICES, 1]
    penetration_mask = (feet_h < ground_height - 0.005)  # 5mm tolerance
    penetration_rate = penetration_mask.float().mean().item()
    metrics['penetration_rate'] = penetration_rate
    if penetration_mask.any():
        pen_depth = (ground_height - feet_h[penetration_mask]).mean().item() * 1000
        metrics['penetration_depth_mm'] = pen_depth
    else:
        metrics['penetration_depth_mm'] = 0.0

    # 5. Foot sliding
    feet_pos = body_pos[:, FOOT_INDICES, :]  # (T, 4, 3)
    if feet_pos.shape[0] > 1:
        feet_vel = feet_pos[1:] - feet_pos[:-1]
        horiz_speed = feet_vel[..., [0, 2]].norm(dim=-1)  # (T-1, 4)
        feet_h_t = feet_h[:-1]
        contact_mask = feet_h_t < (ground_height + contact_threshold)
        if contact_mask.any():
            foot_sliding = horiz_speed[contact_mask].mean().item()
        else:
            foot_sliding = 0.0
    else:
        foot_sliding = 0.0
    metrics['foot_sliding_m_per_frame'] = foot_sliding

    # 6. ZMP violation
    com_xz = body_pos[:, 0, [0, 2]]  # (T, 2) root XZ
    feet_xz = body_pos[:, FOOT_INDICES, :][:, :, [0, 2]]  # (T, 4, 2)
    bb_min = feet_xz.min(dim=1).values - 0.05
    bb_max = feet_xz.max(dim=1).values + 0.05
    below = torch.relu(bb_min - com_xz)
    above = torch.relu(com_xz - bb_max)
    zmp_dist = (below + above).norm(dim=-1).mean().item()
    metrics['zmp_violation_m'] = zmp_dist

    return metrics


@torch.no_grad()
def evaluate_model(
    checkpoint_path: str,
    data_root: str,
    output_file: str = None,
    max_clips: int = 100,
    device: str = "cuda:0",
):
    """
    Evaluate a trained PhysicsRVQVAE model.

    Args:
        checkpoint_path: path to model checkpoint
        data_root: path to InterHuman data
        output_file: where to save JSON results
        max_clips: max number of clips to evaluate
        device: torch device
    """
    print(f"\n{'=' * 60}")
    print(f"Physics VQ-VAE Evaluation")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = TrainingConfig()
    if 'config' in ckpt:
        for k, v in ckpt['config'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Build model
    model = PhysicsRVQVAE(config, device=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()

    # Dataset
    dataset = InterHumanPhysicsDataset(
        data_root=data_root,
        split='test',
        window_size=config.window_size,
        window_stride=config.window_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Test clips: {len(dataset)}")
    print(f"Max clips to evaluate: {max_clips}")

    # Collect metrics
    all_metrics = defaultdict(list)
    reconstruction_losses = []

    for idx, batch in enumerate(loader):
        if idx >= max_clips:
            break

        motion = batch['motion'].to(device).float()
        betas = batch.get('betas')
        if betas is not None:
            betas = betas.to(device).float()

        # Forward pass
        result = model(motion, betas=betas, run_physics=(betas is not None))

        # Reconstruction loss
        x_hat = result['x_hat']
        rec_loss = torch.nn.functional.l1_loss(
            x_hat[..., :-4], motion[..., :-4]
        ).item()
        reconstruction_losses.append(rec_loss)

        # Perplexity
        all_metrics['perplexity'].append(result['perplexity'].item())

        # Physics metrics
        if 'physics' in result:
            physics = result['physics']
            target_pos = physics['target_positions']

            for b in range(physics['sim_positions'].shape[0]):
                sim_result_b = {
                    'sim_positions': physics['sim_positions'][b],
                    'pd_torques': physics['pd_torques'][b],
                    'root_forces': physics['root_forces'][b],
                    'body_positions': physics['body_positions'][b],
                }
                metrics = compute_physics_metrics(
                    sim_result_b, target_pos[b]
                )
                for k, v in metrics.items():
                    all_metrics[k].append(v)

        if (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{min(max_clips, len(dataset))} clips")

    # Aggregate
    results = {
        'num_clips': len(reconstruction_losses),
        'reconstruction_l1': float(np.mean(reconstruction_losses)),
    }
    for k, v in all_metrics.items():
        results[f'{k}_mean'] = float(np.mean(v))
        results[f'{k}_std'] = float(np.std(v))
        results[f'{k}_median'] = float(np.median(v))

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results ({results['num_clips']} clips)")
    print(f"{'=' * 60}")
    print(f"  Reconstruction L1:    {results['reconstruction_l1']:.6f}")
    if 'fk_mpjpe_mm_mean' in results:
        print(f"  FK-MPJPE (mm):        {results['fk_mpjpe_mm_mean']:.2f} "
              f"± {results['fk_mpjpe_mm_std']:.2f}")
        print(f"  Torque RMS (Nm):      {results['torque_rms_nm_mean']:.2f} "
              f"± {results['torque_rms_nm_std']:.2f}")
        print(f"  Skyhook (N):          {results['skyhook_total_mean']:.2f} "
              f"± {results['skyhook_total_std']:.2f}")
        print(f"  Penetration rate:     {results['penetration_rate_mean']:.4f}")
        print(f"  Penetration (mm):     {results['penetration_depth_mm_mean']:.2f}")
        print(f"  Foot sliding (m/f):   {results['foot_sliding_m_per_frame_mean']:.6f}")
        print(f"  ZMP violation (m):    {results['zmp_violation_m_mean']:.4f}")
    if 'perplexity_mean' in results:
        print(f"  Codebook perplexity:  {results['perplexity_mean']:.1f}")
    print(f"{'=' * 60}\n")

    # Save
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Physics VQ-VAE')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--max_clips', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_file=args.output_file,
        max_clips=args.max_clips,
        device=args.device,
    )


if __name__ == '__main__':
    main()
