"""
run_rl_tracker.py — CLI entry point for the PPO RL physics tracker.

Drop-in replacement for prepare5/run_phc_tracker.py.

Usage:
    # Track GT clip (solo)
    python prepare6/run_rl_tracker.py --clip-id 1129 --source gt

    # Track generated clip
    python prepare6/run_rl_tracker.py --clip-id 500 --source generated

    # Smaller/faster run for smoke test
    python prepare6/run_rl_tracker.py --clip-id 1129 --source gt \
        --n-envs 4 --total-timesteps 10000

    # Compare against PHC baseline
    python prepare6/run_rl_tracker.py --clip-id 1129 --source gt --compare-phc

    # Specify GPU
    python prepare6/run_rl_tracker.py --clip-id 1129 --source gt --device cuda:1
"""
import os
import sys
import argparse
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="PPO RL Physics Tracker")
    parser.add_argument("--clip-id",         type=int,   default=1129)
    parser.add_argument("--source",          choices=["gt", "generated"], default="gt")
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--n-envs",          type=int,   default=None,
                        help="Override N_ENVS (default from rl_config.py)")
    parser.add_argument("--total-timesteps", type=int,   default=None,
                        help="Override TOTAL_TIMESTEPS")
    parser.add_argument("--output-dir",      default="output/rl_tracker")
    parser.add_argument("--compare-phc",     action="store_true",
                        help="Also run PHC tracker and compare")
    parser.add_argument("--person",          type=int,   default=0, choices=[0, 1],
                        help="Which person to track (0 or 1)")
    parser.add_argument("--no-verbose",      action="store_true")
    args = parser.parse_args()

    from prepare5.run_phc_tracker import load_clip, retarget_person
    from prepare6.rl_tracker import RLTracker
    from prepare6.rl_config import N_ENVS, TOTAL_TIMESTEPS

    # ── Load clip ──
    print(f"\nLoading clip {args.clip_id} ({args.source})...")
    persons, text = load_clip(args.clip_id, args.source)
    print(f"  Text: {text}")

    # ── Retarget ──
    person_data = persons[args.person]
    print(f"\nRetargeting person {args.person}...")
    joint_q, betas = retarget_person(person_data, args.source, device=args.device)
    print(f"  joint_q: {joint_q.shape}, betas: {betas.shape}")

    # ── Output dir ──
    tag = f"clip_{args.clip_id}_{args.source}_p{args.person}"
    out_dir = os.path.join(PROJECT_ROOT, args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    n_envs          = args.n_envs          or N_ENVS
    total_timesteps = args.total_timesteps or TOTAL_TIMESTEPS

    # ── RL Tracker ──
    t0 = time.time()
    tracker = RLTracker(
        device=args.device,
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        verbose=not args.no_verbose,
    )
    result = tracker.train_and_evaluate(joint_q, betas)

    # ── Save policy ──
    # (policy is inside tracker's env/agent; re-run to save)
    # For now, save metrics and arrays

    # ── Save metrics ──
    metrics = {
        "clip_id":           args.clip_id,
        "source":            args.source,
        "person":            args.person,
        "mpjpe_mm":          result['mpjpe_mm'],
        "max_error_mm":      result['max_error_mm'],
        "elapsed_s":         result['elapsed_s'],
        "training_timesteps": result['training_timesteps'],
        "early_stopped":     result['early_stopped'],
        "n_envs":            n_envs,
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Save arrays ──
    npz_path = os.path.join(out_dir, "rl_result.npz")
    np.savez(
        npz_path,
        sim_positions=result['sim_positions'],
        ref_positions=result['ref_positions'],
        sim_joint_q=result['sim_joint_q'],
        ref_joint_q=joint_q,
        per_frame_mpjpe_mm=result['per_frame_mpjpe_mm'],
        per_joint_mpjpe_mm=result['per_joint_mpjpe_mm'],
    )
    print(f"Arrays saved: {npz_path}")

    # ── Save training curve ──
    curve_path = os.path.join(out_dir, "training_curve.json")
    with open(curve_path, 'w') as f:
        json.dump(result['training_curve'], f)

    # ── Plot tracking ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    T = result['per_frame_mpjpe_mm'].shape[0]
    axes[0].plot(result['per_frame_mpjpe_mm'], label='RL tracker', color='blue')
    axes[0].axhline(result['mpjpe_mm'], color='blue', linestyle='--', alpha=0.6,
                    label=f'mean={result["mpjpe_mm"]:.1f}mm')
    axes[0].set_ylabel("MPJPE (mm)")
    axes[0].set_title(f"RL Tracker — Clip {args.clip_id} {args.source.upper()} "
                      f"Person {args.person}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rewards = [u['mean_reward'] for u in result['training_curve']]
    axes[1].plot(rewards, color='green')
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("Mean reward")
    axes[1].set_title("Training curve")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "rl_tracking.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Plot saved: {plot_path}")

    # ── Compare with PHC ──
    if args.compare_phc:
        print("\n" + "="*60)
        print("Running PHC baseline for comparison...")
        from prepare5.phc_tracker import PHCTracker

        phc = PHCTracker(device=args.device, verbose=not args.no_verbose)
        phc_result = phc.track(joint_q, betas)

        print(f"\n  {'Method':<12} {'MPJPE':>10} {'Max Err':>10}")
        print(f"  {'-'*34}")
        print(f"  {'RL (PPO)':<12} {result['mpjpe_mm']:>9.1f}mm "
              f"{result['max_error_mm']:>9.1f}mm")
        print(f"  {'PHC (PD)':<12} {phc_result['mpjpe_mm']:>9.1f}mm "
              f"{phc_result['max_error_mm']:>9.1f}mm")
        improvement = phc_result['mpjpe_mm'] - result['mpjpe_mm']
        print(f"\n  RL improvement: {improvement:+.1f}mm "
              f"({'better' if improvement > 0 else 'worse'})")

        # Save comparison
        comparison = {
            "rl_mpjpe_mm":  result['mpjpe_mm'],
            "phc_mpjpe_mm": phc_result['mpjpe_mm'],
            "improvement_mm": improvement,
        }
        comp_path = os.path.join(out_dir, "comparison_phc.json")
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved: {comp_path}")

    print(f"\nAll tasks completed. Output: {out_dir}")


if __name__ == "__main__":
    main()
