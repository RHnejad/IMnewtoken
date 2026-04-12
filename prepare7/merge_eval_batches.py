#!/usr/bin/env python3
"""Merge multiple batch eval JSON files into one combined result."""
import json
import logging
import sys
import os
import numpy as np

logger = logging.getLogger("merge_eval_batches")


def merge_batch_jsons(json_paths, output_path):
    all_per_motion = {}
    total_motions = 0

    for path in sorted(json_paths):
        logger.info(f"Loading batch: {path}")
        with open(path) as f:
            data = json.load(f)
        batch_n = len(data.get("per_motion", {}))
        batch_sr = data.get("success_rate", "N/A")
        logger.info(f"  Motions: {batch_n}, success_rate: {batch_sr}")
        all_per_motion.update(data["per_motion"])
        total_motions += data["num_motions"]

    # Recompute aggregate from per-motion data
    num_motions = len(all_per_motion)
    num_failed = sum(1 for v in all_per_motion.values() if v.get("failed", True))
    success_rate = 1.0 - num_failed / num_motions if num_motions > 0 else 0.0

    # Aggregate each metric
    aggregate = {
        "eval/success_rate": success_rate,
        "eval/gt_error/failure_rate": num_failed / num_motions if num_motions > 0 else 1.0,
    }

    for metric in ["gt_error", "gr_error", "max_joint_error"]:
        means = [v[f"{metric}_mean"] for v in all_per_motion.values()
                 if v.get(f"{metric}_mean") is not None]
        mins = [v[f"{metric}_min"] for v in all_per_motion.values()
                if v.get(f"{metric}_min") is not None]
        maxs = [v[f"{metric}_max"] for v in all_per_motion.values()
                if v.get(f"{metric}_max") is not None]
        if means:
            aggregate[f"eval/{metric}/mean"] = float(np.mean(means))
            aggregate[f"eval/{metric}/min"] = float(np.min(mins))
            aggregate[f"eval/{metric}/max"] = float(np.max(maxs))

    results = {
        "aggregate": aggregate,
        "per_motion": all_per_motion,
        "success_rate": success_rate,
        "num_motions": num_motions,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Batches merged: {len(json_paths)}")
    logger.info(f"  Total motions:  {num_motions}")
    logger.info(f"  Failed:         {num_failed}")
    logger.info(f"  Success rate:   {success_rate:.4f}")
    for k, v in sorted(aggregate.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.6f}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_eval_batches.py <output.json> <batch1.json> [batch2.json ...]")
        sys.exit(1)

    output_path = sys.argv[1]
    log_path = output_path.replace(".json", "_merge.log") if output_path.endswith(".json") else output_path + "_merge.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    merge_batch_jsons(sys.argv[2:], output_path)
