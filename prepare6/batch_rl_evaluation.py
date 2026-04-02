"""
batch_rl_evaluation.py — Run RL tracker on GT and generated test sets.

Evaluates physics plausibility by comparing tracking MPJPE:
  - GT motions should track better (lower MPJPE)
  - Generated motions should track worse (higher MPJPE)
  - The gap = physics plausibility metric

Usage:
    # Full test set (1098 clips, both GT and generated)
    python prepare6/batch_rl_evaluation.py --n-clips 0

    # Quick test (20 clips)
    python prepare6/batch_rl_evaluation.py --n-clips 20

    # GT only
    python prepare6/batch_rl_evaluation.py --n-clips 50 --source gt

    # Resume interrupted run
    python prepare6/batch_rl_evaluation.py --n-clips 50 --resume
"""
import os
import sys
import json
import time
import argparse
import traceback
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_test_ids(n_clips=0, seed=42):
    """Load test set clip IDs."""
    split_path = os.path.join(
        PROJECT_ROOT, "data", "text_generated_dataset",
        "interhuman", "split", "test.txt"
    )
    with open(split_path) as f:
        ids = [int(line.strip()) for line in f if line.strip()]

    if n_clips > 0 and n_clips < len(ids):
        rng = np.random.RandomState(seed)
        ids = sorted(rng.choice(ids, size=n_clips, replace=False).tolist())

    return ids


def has_generated_data(clip_id):
    """Check if a generated clip exists."""
    gen_path = os.path.join(
        PROJECT_ROOT, "data", "generated", "interhuman", f"{clip_id}.pkl"
    )
    return os.path.isfile(gen_path)


def run_clip(clip_id, source, person, device, n_envs, total_timesteps):
    """Run RL tracker on a single clip+source+person.

    Returns metrics dict or None on failure.
    """
    from prepare5.run_phc_tracker import load_clip, retarget_person
    from prepare6.rl_tracker import RLTracker

    try:
        persons, text = load_clip(clip_id, source)
    except Exception as e:
        print(f"  [SKIP] clip {clip_id} {source}: load failed: {e}")
        return None

    try:
        joint_q, betas = retarget_person(persons[person], source, device=device)
    except Exception as e:
        print(f"  [SKIP] clip {clip_id} {source} p{person}: retarget failed: {e}")
        return None

    if joint_q.shape[0] < 10:
        print(f"  [SKIP] clip {clip_id}: too short ({joint_q.shape[0]} frames)")
        return None

    tracker = RLTracker(
        device=device,
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        verbose=False,
    )
    try:
        result = tracker.train_and_evaluate(joint_q, betas)
    except Exception as e:
        print(f"  [FAIL] clip {clip_id} {source} p{person}: {e}")
        traceback.print_exc()
        return None

    return {
        "clip_id": clip_id,
        "source": source,
        "person": person,
        "mpjpe_mm": result["mpjpe_mm"],
        "max_error_mm": result["max_error_mm"],
        "elapsed_s": result["elapsed_s"],
        "training_timesteps": result["training_timesteps"],
        "early_stopped": result["early_stopped"],
        "n_frames": joint_q.shape[0],
        "final_reward": result["training_curve"][-1]["mean_reward"]
            if result["training_curve"] else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch RL tracker evaluation")
    parser.add_argument("--n-clips", type=int, default=20,
                        help="Number of test clips (0=all)")
    parser.add_argument("--source", choices=["gt", "generated", "both"],
                        default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200000,
                        help="Timesteps per clip (200k is faster; 500k for best)")
    parser.add_argument("--output-dir", default="output/rl_batch_eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--person", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.json")

    # Load existing results if resuming
    existing = {}
    if args.resume and os.path.isfile(results_path):
        with open(results_path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            key = f"{r['clip_id']}_{r['source']}_p{r['person']}"
            existing[key] = r
        print(f"Resuming: {len(existing)} clips already done")

    clip_ids = load_test_ids(args.n_clips, seed=args.seed)
    sources = ["gt", "generated"] if args.source == "both" else [args.source]

    print(f"\nBatch RL Evaluation")
    print(f"  Clips: {len(clip_ids)}")
    print(f"  Sources: {sources}")
    print(f"  N_ENVS: {args.n_envs}, timesteps: {args.total_timesteps}")
    print(f"  Output: {out_dir}")
    print(f"  {'='*50}")

    all_results = list(existing.values())
    t0_total = time.time()

    for i, clip_id in enumerate(clip_ids):
        for source in sources:
            if source == "generated" and not has_generated_data(clip_id):
                continue

            key = f"{clip_id}_{source}_p{args.person}"
            if key in existing:
                continue

            print(f"\n[{i+1}/{len(clip_ids)}] Clip {clip_id} {source} "
                  f"p{args.person}...", flush=True)

            result = run_clip(
                clip_id, source, args.person,
                args.device, args.n_envs, args.total_timesteps,
            )
            if result is not None:
                all_results.append(result)
                existing[key] = result
                print(f"  MPJPE: {result['mpjpe_mm']:.1f}mm  "
                      f"reward: {result['final_reward']:.3f}  "
                      f"({result['elapsed_s']:.0f}s)")

            # Save after each clip for resume-ability
            _save_results(all_results, results_path, args)

    elapsed_total = time.time() - t0_total
    _save_results(all_results, results_path, args)
    _print_summary(all_results, elapsed_total)


def _save_results(results, path, args):
    """Save results JSON with summary stats."""
    gt_results  = [r for r in results if r["source"] == "gt"]
    gen_results = [r for r in results if r["source"] == "generated"]

    summary = {
        "n_clips_gt":  len(gt_results),
        "n_clips_gen": len(gen_results),
    }
    if gt_results:
        mpjpes = [r["mpjpe_mm"] for r in gt_results]
        summary["gt_mpjpe_mean"]   = float(np.mean(mpjpes))
        summary["gt_mpjpe_median"] = float(np.median(mpjpes))
        summary["gt_mpjpe_std"]    = float(np.std(mpjpes))
    if gen_results:
        mpjpes = [r["mpjpe_mm"] for r in gen_results]
        summary["gen_mpjpe_mean"]   = float(np.mean(mpjpes))
        summary["gen_mpjpe_median"] = float(np.median(mpjpes))
        summary["gen_mpjpe_std"]    = float(np.std(mpjpes))
    if gt_results and gen_results:
        summary["gap_mm"] = summary["gen_mpjpe_mean"] - summary["gt_mpjpe_mean"]

    data = {
        "config": {
            "n_envs": args.n_envs,
            "total_timesteps": args.total_timesteps,
            "person": args.person,
        },
        "summary": summary,
        "results": results,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _print_summary(results, elapsed):
    gt  = [r for r in results if r["source"] == "gt"]
    gen = [r for r in results if r["source"] == "generated"]

    print(f"\n{'='*60}")
    print(f"BATCH EVALUATION COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    if gt:
        mpjpes = [r["mpjpe_mm"] for r in gt]
        print(f"\n  GT ({len(gt)} clips):")
        print(f"    MPJPE: {np.mean(mpjpes):.1f} ± {np.std(mpjpes):.1f} mm "
              f"(median {np.median(mpjpes):.1f})")

    if gen:
        mpjpes = [r["mpjpe_mm"] for r in gen]
        print(f"\n  Generated ({len(gen)} clips):")
        print(f"    MPJPE: {np.mean(mpjpes):.1f} ± {np.std(mpjpes):.1f} mm "
              f"(median {np.median(mpjpes):.1f})")

    if gt and gen:
        gap = np.mean([r["mpjpe_mm"] for r in gen]) - np.mean([r["mpjpe_mm"] for r in gt])
        print(f"\n  Gap (Gen - GT): {gap:+.1f} mm")
        print(f"  {'VALID' if gap > 0 else 'INVALID'}: "
              f"{'generated tracks worse → metric works' if gap > 0 else 'unexpected!'}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
