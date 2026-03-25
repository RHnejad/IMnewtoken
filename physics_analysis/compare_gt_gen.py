#!/usr/bin/env python
"""
Side-by-side comparison of GT vs Generated physics analysis results.
Creates combined PNG images and prints summary table.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_comparison(clip_id, gt_dir, gen_dir, out_dir):
    """Create side-by-side comparison for one clip."""
    os.makedirs(out_dir, exist_ok=True)

    for prefix, label in [("newton_analysis", "Force Analysis"), 
                           ("skeleton_keyframes", "Skeleton Keyframes")]:
        gt_path = os.path.join(gt_dir, f"{prefix}_clip_{clip_id}_m12.png")
        gen_path = os.path.join(gen_dir, f"{prefix}_clip_{clip_id}_m12.png")

        if not os.path.exists(gt_path):
            print(f"  WARNING: GT not found: {gt_path}")
            continue
        if not os.path.exists(gen_path):
            print(f"  WARNING: Gen not found: {gen_path}")
            continue

        gt_img = mpimg.imread(gt_path)
        gen_img = mpimg.imread(gen_path)

        # Create side-by-side figure
        fig, axes = plt.subplots(1, 2, figsize=(28, max(gt_img.shape[0], gen_img.shape[0]) / 100))
        
        axes[0].imshow(gt_img)
        axes[0].set_title(f"GT — Clip {clip_id}", fontsize=16, fontweight='bold', color='green')
        axes[0].axis('off')

        axes[1].imshow(gen_img)
        axes[1].set_title(f"Generated — Clip {clip_id}", fontsize=16, fontweight='bold', color='red')
        axes[1].axis('off')

        fig.suptitle(f"{label} — Clip {clip_id}: GT vs Generated", fontsize=20, fontweight='bold')
        fig.tight_layout()
        
        out_path = os.path.join(out_dir, f"comparison_{prefix}_clip_{clip_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def print_summary_table():
    """Print comparison metrics (Method 2: Inverse Dynamics root residual forces).
    
    Data from position-based IK retarget pipeline (corrected coordinate frames).
    GT source: data/retargeted_v2/gt_from_positions/
    Gen source: data/retargeted_v2/gen_from_positions/
    """
    print("\n" + "="*80)
    print("PHYSICS ANALYSIS SUMMARY: GT vs Generated (Position-based IK)")
    print("="*80)
    
    # Metrics from corrected position-based IK retarget (Method 2: Inverse Dynamics)
    data = {
        '4659': {
            'GT': {'frames': 36, 'p1_mean': 834464.5, 'p1_max': 3999532.0, 
                   'p2_mean': 227346.5, 'p2_max': 360012.6},
            'Gen': {'frames': 32, 'p1_mean': 106148.7, 'p1_max': 186328.3,
                    'p2_mean': 120580.4, 'p2_max': 312098.9},
        },
        '3678': {
            'GT': {'frames': 2580, 'p1_mean': 306516.6, 'p1_max': 13433694.0,
                   'p2_mean': 368049.4, 'p2_max': 24096532.0},
            'Gen': {'frames': 296, 'p1_mean': 745351.2, 'p1_max': 51336532.0,
                    'p2_mean': 237437.0, 'p2_max': 2448834.2},
        },
    }

    for clip_id, clip_data in data.items():
        gt = clip_data['GT']
        gen = clip_data['Gen']
        print(f"\n{'─'*60}")
        print(f"Clip {clip_id}")
        print(f"{'─'*60}")
        print(f"{'Metric':<30} {'GT':>15} {'Generated':>15} {'Ratio (Gen/GT)':>15}")
        print(f"{'─'*75}")
        print(f"{'Frames':<30} {gt['frames']:>15} {gen['frames']:>15} {gen['frames']/gt['frames']:>15.2f}x")
        for person_label, mean_key, max_key in [('P1', 'p1_mean', 'p1_max'), ('P2', 'p2_mean', 'p2_max')]:
            gt_mean, gen_mean = gt[mean_key], gen[mean_key]
            gt_max, gen_max = gt[max_key], gen[max_key]
            ratio_mean = gen_mean / gt_mean if gt_mean > 0 else float('inf')
            ratio_max = gen_max / gt_max if gt_max > 0 else float('inf')
            print(f"{f'{person_label} residual (mean N)':<30} {gt_mean:>15.1f} {gen_mean:>15.1f} {ratio_mean:>15.2f}x")
            print(f"{f'{person_label} residual (max N)':<30} {gt_max:>15.1f} {gen_max:>15.1f} {ratio_max:>15.2f}x")

    print(f"\n{'='*80}")
    print("OBSERVATIONS:")
    print("  - Both GT and Gen have very high residual forces (100k-800k N range)")
    print("    This is expected: position-based IK → finite-diff velocities/accelerations")
    print("    introduces large numerical artifacts at 30 fps")
    print("  - Clip 4659: Gen actually has LOWER residuals than GT (0.13x for P1)")
    print("    Likely because Gen is smoother (VQ quantization smooths out jitter)")
    print("  - Clip 3678: Gen has ~2.4x higher P1 mean, ~0.6x lower P2 mean than GT")
    print("    Mixed results — Gen is not uniformly worse")
    print("  - Generated sequences are shorter (VQ-VAE token limit)")
    print(f"{'='*80}\n")


def main():
    gt_dir = os.path.join(PROJECT_ROOT, "physics_analysis", "gt_results")
    gen_dir = os.path.join(PROJECT_ROOT, "physics_analysis", "gen_results")
    out_dir = os.path.join(PROJECT_ROOT, "physics_analysis", "comparison_results")

    clips = ["4659", "3678"]

    for clip_id in clips:
        print(f"\nCreating comparison for clip {clip_id}...")
        create_comparison(clip_id, gt_dir, gen_dir, out_dir)

    print_summary_table()


if __name__ == "__main__":
    main()
