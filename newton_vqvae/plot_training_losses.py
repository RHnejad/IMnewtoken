"""
plot_training_losses.py — Plot all loss terms from TensorBoard logs.

Reads TensorBoard event files and plots:
- Total loss, reconstruction loss, kinematic loss
- All physics sub-losses (FK-MPJPE, torque, skyhook, SoftFlow, ZMP, contact budget)
- Physics weight schedule
- Learning rate

Usage:
    conda activate mimickit
    python newton_vqvae/plot_training_losses.py --logdir outputs/overfit_test/logs
    python newton_vqvae/plot_training_losses.py --logdir outputs/newton_vqvae_full/logs

Options:
    --logdir: TensorBoard log directory
    --output: Output image path (default: <logdir>/training_losses.png)
    --show: Show plot interactively
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

# Ensure imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def read_tensorboard_logs(logdir: str) -> dict:
    """Read all scalar events from TensorBoard log files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Run: pip install tensorboard")
        sys.exit(1)

    ea = EventAccumulator(logdir)
    ea.Reload()

    data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}

    return data


def read_from_stdout_log(logdir: str) -> dict:
    """Fallback: parse training stdout output for loss values.

    This works even without TensorBoard event files, by scanning for lines like:
      [Ep 1, It 358/179250] loss=0.8299 rec=0.3186 phys=0.0000 lr=0.000040 (124s)
    """
    data = defaultdict(lambda: {'steps': [], 'values': []})

    log_file = None
    # Try finding a log file
    for name in ['train.log', 'stdout.log', 'output.log']:
        path = os.path.join(logdir, name)
        if os.path.exists(path):
            log_file = path
            break

    if log_file is None:
        return dict(data)

    import re
    pattern = re.compile(
        r'\[Ep (\d+), It (\d+)/\d+\] '
        r'loss=([\d.]+) rec=([\d.]+) phys=([\d.]+) lr=([\d.]+)'
    )

    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                it = int(m.group(2))
                data['Train/loss_total']['steps'].append(it)
                data['Train/loss_total']['values'].append(float(m.group(3)))
                data['Train/loss_rec']['steps'].append(it)
                data['Train/loss_rec']['values'].append(float(m.group(4)))
                data['Train/loss_physics_weighted']['steps'].append(it)
                data['Train/loss_physics_weighted']['values'].append(float(m.group(5)))
                data['Train/lr']['steps'].append(it)
                data['Train/lr']['values'].append(float(m.group(6)))

    # Convert to numpy
    for k in data:
        data[k]['steps'] = np.array(data[k]['steps'])
        data[k]['values'] = np.array(data[k]['values'])

    return dict(data)


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def plot_losses(data: dict, output_path: str, show: bool = False):
    """Create multi-panel loss plot."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Categorize available tags
    train_tags = {k: v for k, v in data.items() if k.startswith('Train/')}
    val_tags = {k: v for k, v in data.items() if k.startswith('Val/')}

    # Define plot groups
    groups = [
        {
            'title': 'Total & Reconstruction Loss',
            'tags': ['Train/loss_total', 'Train/loss_rec', 'Train/loss_kinematic'],
            'val_tags': ['Val/loss_total', 'Val/loss_rec'],
            'colors': ['#2196F3', '#FF9800', '#4CAF50'],
            'val_colors': ['#1565C0', '#E65100'],
        },
        {
            'title': 'Kinematic Sub-Losses',
            'tags': ['Train/loss_explicit', 'Train/loss_vel', 'Train/loss_bn',
                     'Train/loss_geo', 'Train/loss_fc', 'Train/loss_commit'],
            'colors': ['#E91E63', '#9C27B0', '#3F51B5', '#00BCD4', '#795548', '#607D8B'],
        },
        {
            'title': 'Physics Losses',
            'tags': ['Train/l_fk_mpjpe', 'Train/l_torque', 'Train/l_skyhook',
                     'Train/l_softflow', 'Train/l_zmp', 'Train/l_contact_budget'],
            'colors': ['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3'],
        },
        {
            'title': 'Physics Weight & Weighted Loss',
            'tags': ['Train/physics_weight', 'Train/loss_physics_weighted', 'Train/l_physics_total'],
            'colors': ['#4CAF50', '#FF5722', '#FF9800'],
        },
        {
            'title': 'Learning Rate',
            'tags': ['Train/lr'],
            'colors': ['#607D8B'],
        },
    ]

    # Filter to groups that have data
    active_groups = []
    for g in groups:
        available_tags = [t for t in g['tags'] if t in data and len(data[t]['values']) > 0]
        if available_tags:
            g['available_tags'] = available_tags
            active_groups.append(g)

    if not active_groups:
        print("WARNING: No loss data found to plot.")
        print(f"  Available tags: {list(data.keys())}")
        return

    n_panels = len(active_groups)
    fig = plt.figure(figsize=(14, 4 * n_panels))
    gs = gridspec.GridSpec(n_panels, 1, hspace=0.35)

    for idx, group in enumerate(active_groups):
        ax = fig.add_subplot(gs[idx])
        ax.set_title(group['title'], fontsize=13, fontweight='bold')

        for i, tag in enumerate(group['available_tags']):
            d = data[tag]
            label = tag.split('/')[-1]
            color = group['colors'][i % len(group['colors'])]

            # Plot raw data (faded) + smoothed
            if len(d['values']) > 5:
                ax.plot(d['steps'], d['values'], alpha=0.15, color=color, linewidth=0.5)
                ax.plot(d['steps'], smooth(d['values'], 0.9), color=color,
                        linewidth=1.5, label=label)
            else:
                ax.plot(d['steps'], d['values'], color=color, linewidth=1.5,
                        label=label, marker='o', markersize=3)

        # Overlay validation curves if available
        if 'val_tags' in group:
            for i, tag in enumerate(group.get('val_tags', [])):
                if tag in data and len(data[tag]['values']) > 0:
                    d = data[tag]
                    label = tag.split('/')[-1] + ' (val)'
                    color = group.get('val_colors', group['colors'])[i % len(group['colors'])]
                    ax.plot(d['steps'], d['values'], color=color, linewidth=2,
                            linestyle='--', label=label, marker='s', markersize=4)

        ax.set_xlabel('Step' if idx == n_panels - 1 else '')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Use log scale for physics losses if range is large
        if 'Physics Losses' in group['title']:
            values_concat = np.concatenate([
                data[t]['values'] for t in group['available_tags']
            ])
            if values_concat.max() > 0 and values_concat.max() / max(values_concat.min(), 1e-10) > 100:
                ax.set_yscale('log')

    fig.suptitle('Physics-Informed VQ-VAE Training Progress', fontsize=15,
                 fontweight='bold', y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser("Plot Training Losses")
    parser.add_argument("--logdir", type=str, required=True,
                        help="TensorBoard log directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--show", action="store_true",
                        help="Show plot interactively")
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.logdir.rstrip('/')),
            'training_losses.png'
        )

    print(f"Reading logs from: {args.logdir}")
    data = read_tensorboard_logs(args.logdir)

    if not data:
        print("No TensorBoard events found. Trying stdout log parse...")
        data = read_from_stdout_log(args.logdir)

    if not data:
        print("ERROR: No training data found.")
        sys.exit(1)

    print(f"Found {len(data)} metric tags:")
    for tag, d in sorted(data.items()):
        print(f"  {tag}: {len(d['values'])} points")

    plot_losses(data, args.output, show=args.show)


if __name__ == '__main__':
    main()
