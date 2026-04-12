#!/usr/bin/env python3
"""Compare GT vs generated tracking MPJPE from ProtoMotions evaluation.

Reads per-motion evaluation JSON files produced by run_evaluation.py
and computes comparison metrics + plots.

Usage:
    python prepare7/compare_results.py \
        --gt-json prepare7/output/eval_gt.json \
        --gen-json prepare7/output/eval_generated.json \
        --output-dir prepare7/output/comparison
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("compare_results")


def load_eval_json(path):
    with open(path) as f:
        return json.load(f)


def extract_per_clip_mpjpe(eval_data):
    """Extract per-clip MPJPE (gt_error_mean) from evaluation results.

    Returns dict: clip_id → {person1: mpjpe, person2: mpjpe}
    """
    per_clip = defaultdict(dict)
    for motion_name, metrics in eval_data["per_motion"].items():
        # motion_name format: "{clip_id}_person{1|2}"
        parts = motion_name.rsplit("_person", 1)
        if len(parts) == 2:
            clip_id, person_num = parts[0], f"person{parts[1]}"
        else:
            clip_id, person_num = motion_name, "unknown"

        mpjpe = metrics.get("gt_error_mean")
        if mpjpe is not None:
            per_clip[clip_id][person_num] = {
                "mpjpe": mpjpe,
                "failed": metrics.get("failed", False),
                "gt_error_min": metrics.get("gt_error_min"),
                "gt_error_max": metrics.get("gt_error_max"),
                "gr_error_mean": metrics.get("gr_error_mean"),
                "max_joint_error_mean": metrics.get("max_joint_error_mean"),
            }
    return per_clip


def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare GT vs generated tracking MPJPE")
    parser.add_argument("--gt-json", type=str, required=True, help="GT evaluation JSON")
    parser.add_argument("--gen-json", type=str, required=True, help="Generated evaluation JSON")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prepare7/output/comparison",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Logging setup ---
    log_path = os.path.join(args.output_dir, "comparison.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"GT JSON:  {args.gt_json}")
    logger.info(f"Gen JSON: {args.gen_json}")
    logger.info(f"Output:   {args.output_dir}")

    gt_data = load_eval_json(args.gt_json)
    gen_data = load_eval_json(args.gen_json)

    gt_clips = extract_per_clip_mpjpe(gt_data)
    gen_clips = extract_per_clip_mpjpe(gen_data)

    # Collect all MPJPE values
    gt_mpjpes = []
    gen_mpjpes = []
    paired_diffs = []  # gen - gt for matched clips
    paired_gt = []
    paired_gen = []

    # Per-clip comparison (match by clip_id + person)
    per_clip_comparison = {}
    for clip_id in sorted(set(gt_clips.keys()) | set(gen_clips.keys())):
        clip_entry = {}
        for person in ["person1", "person2"]:
            gt_val = gt_clips.get(clip_id, {}).get(person, {}).get("mpjpe")
            gen_val = gen_clips.get(clip_id, {}).get(person, {}).get("mpjpe")

            if gt_val is not None:
                gt_mpjpes.append(gt_val)
            if gen_val is not None:
                gen_mpjpes.append(gen_val)
            if gt_val is not None and gen_val is not None:
                paired_diffs.append(gen_val - gt_val)
                paired_gt.append(gt_val)
                paired_gen.append(gen_val)
                clip_entry[person] = {
                    "gt_mpjpe": gt_val,
                    "gen_mpjpe": gen_val,
                    "delta": gen_val - gt_val,
                }
        if clip_entry:
            per_clip_comparison[clip_id] = clip_entry

    # Aggregate stats
    results = {
        "gt_aggregate": gt_data.get("aggregate", {}),
        "gen_aggregate": gen_data.get("aggregate", {}),
        "gt_mpjpe_stats": compute_stats(gt_mpjpes) if gt_mpjpes else None,
        "gen_mpjpe_stats": compute_stats(gen_mpjpes) if gen_mpjpes else None,
        "delta_stats": compute_stats(paired_diffs) if paired_diffs else None,
        "num_gt_motions": gt_data.get("num_motions", 0),
        "num_gen_motions": gen_data.get("num_motions", 0),
        "num_paired": len(paired_diffs),
        "gt_success_rate": gt_data.get("success_rate"),
        "gen_success_rate": gen_data.get("success_rate"),
    }

    # Save comparison JSON
    comparison_path = os.path.join(args.output_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save per-clip comparison
    per_clip_path = os.path.join(args.output_dir, "per_clip_comparison.json")
    with open(per_clip_path, "w") as f:
        json.dump(per_clip_comparison, f, indent=2)

    # Print summary
    logger.info("=" * 70)
    logger.info("PP-Motion Physics Plausibility Comparison: GT vs Generated")
    logger.info("=" * 70)

    if results["gt_mpjpe_stats"]:
        s = results["gt_mpjpe_stats"]
        logger.info(f"GT Tracking MPJPE (m):")
        logger.info(f"  Mean: {s['mean']:.4f}  Median: {s['median']:.4f}  "
                     f"Std: {s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]  N={s['count']}")

    if results["gen_mpjpe_stats"]:
        s = results["gen_mpjpe_stats"]
        logger.info(f"Generated Tracking MPJPE (m):")
        logger.info(f"  Mean: {s['mean']:.4f}  Median: {s['median']:.4f}  "
                     f"Std: {s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]  N={s['count']}")

    if results["delta_stats"]:
        s = results["delta_stats"]
        logger.info(f"Delta (Gen - GT) MPJPE (m):")
        logger.info(f"  Mean: {s['mean']:.4f}  Median: {s['median']:.4f}  "
                     f"Std: {s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]  N={s['count']}")

    logger.info(f"Success rates: GT={results['gt_success_rate']}, Gen={results['gen_success_rate']}")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"Log saved to {log_path}")

    # Generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Histogram of MPJPE distributions
        ax = axes[0]
        bins = np.linspace(0, max(max(gt_mpjpes, default=0.5), max(gen_mpjpes, default=0.5)) * 1.1, 50)
        if gt_mpjpes:
            ax.hist(gt_mpjpes, bins=bins, alpha=0.6, label=f"GT (mean={np.mean(gt_mpjpes):.4f})")
        if gen_mpjpes:
            ax.hist(gen_mpjpes, bins=bins, alpha=0.6, label=f"Gen (mean={np.mean(gen_mpjpes):.4f})")
        ax.set_xlabel("Tracking MPJPE (m)")
        ax.set_ylabel("Count")
        ax.set_title("MPJPE Distribution")
        ax.legend()

        # 2. Delta histogram
        ax = axes[1]
        if paired_diffs:
            ax.hist(paired_diffs, bins=50, alpha=0.7, color="green")
            ax.axvline(x=0, color="red", linestyle="--", label="No difference")
            ax.axvline(x=np.mean(paired_diffs), color="blue", linestyle="--",
                       label=f"Mean={np.mean(paired_diffs):.4f}")
            ax.set_xlabel("Delta MPJPE: Gen - GT (m)")
            ax.set_ylabel("Count")
            ax.set_title("Per-Clip MPJPE Difference")
            ax.legend()

        # 3. Scatter: GT vs Gen MPJPE
        ax = axes[2]
        if paired_gt and paired_gen:
            ax.scatter(paired_gt, paired_gen, alpha=0.3, s=10)
            lim = max(max(paired_gt), max(paired_gen)) * 1.1
            ax.plot([0, lim], [0, lim], "r--", label="y=x")
            ax.set_xlabel("GT MPJPE (m)")
            ax.set_ylabel("Generated MPJPE (m)")
            ax.set_title("GT vs Generated MPJPE")
            ax.legend()
            ax.set_aspect("equal")

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "comparison_plots.png")
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Plots saved to {plot_path}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
