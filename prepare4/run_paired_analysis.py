"""
run_paired_analysis.py — CLI for paired-vs-solo torque interaction analysis.

Compares torques when both persons are simulated together vs alone,
revealing physical (im)plausibility of interaction motions.

Usage:
    # Single clip analysis (GT)
    python prepare4/run_paired_analysis.py --clip-id 1129 --source gt

    # Multiple clips
    python prepare4/run_paired_analysis.py --clip-ids 1129 1147 1187 --source gt

    # Batch GT (200 clips)
    python prepare4/run_paired_analysis.py --batch --source gt --n-clips 200

    # Batch Generated
    python prepare4/run_paired_analysis.py --batch --source generated --n-clips 200

    # Compare GT vs Generated (after batch runs)
    python prepare4/run_paired_analysis.py --compare \
        --gt-dir data/paired_eval_gt --gen-dir data/paired_eval_gen
"""
import os
import sys
import json
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FPS = 30
DT = 1.0 / FPS


def analyze_single_clip(clip_id, source, device="cuda:0",
                         output_dir=None, data_dir=None):
    """Run full paired-vs-solo analysis on one clip with plots + metrics."""
    from prepare4.paired_simulation import compute_paired_vs_solo
    from prepare4.interaction_metrics import compute_all_metrics, format_metrics_summary
    from prepare4.plot_paired_torques import generate_all_clip_plots
    from prepare4.run_full_analysis import (
        load_gt_persons, load_gen_persons, load_positions_zup, load_motion_text,
    )
    from prepare4.retarget import rotation_retarget, ik_retarget

    print(f"\n{'='*60}")
    print(f" Analyzing clip {clip_id} ({source})")
    print(f"{'='*60}")

    text = load_motion_text(clip_id)
    print(f"  Text: {text}")

    # Load persons
    if source == "generated":
        persons, _ = load_gen_persons(clip_id)
    else:
        persons = load_gt_persons(clip_id)

    if persons is None or len(persons) < 2:
        print(f"  ERROR: Could not load clip {clip_id}")
        return None

    # Retarget both persons
    jqs = []
    betas_list = []
    for i, p in enumerate(persons):
        betas = p["betas"]
        betas_list.append(betas)

        if source == "generated":
            if "positions_zup" not in p:
                print(f"  ERROR: person {i+1} has no positions_zup")
                return None
            jq, _ = ik_retarget(
                p["positions_zup"], betas,
                ik_iters=50, device=device, sequential=True,
            )
        else:
            jq = rotation_retarget(
                p["root_orient"], p["pose_body"], p["trans"], betas,
            )
            jq = jq[::2]  # 60fps → 30fps

        jqs.append(jq)
        print(f"  Person {i+1}: {jq.shape[0]} frames retargeted")

    T = min(jqs[0].shape[0], jqs[1].shape[0])
    if T < 11:
        print(f"  ERROR: clip too short ({T} frames)")
        return None

    jq_A, jq_B = jqs[0][:T], jqs[1][:T]
    betas_A, betas_B = betas_list[0], betas_list[1]

    # Run 3 scenarios
    t0 = time.time()
    result = compute_paired_vs_solo(
        jq_A, jq_B, betas_A, betas_B,
        dt=DT, device=device, verbose=True,
    )
    sim_time = time.time() - t0
    print(f"\n  Simulation time: {sim_time:.1f}s")

    # Load positions for CTC
    pos_A, pos_B = load_positions_zup(clip_id, source)
    if pos_A is not None and pos_B is not None:
        pos_A, pos_B = pos_A[:T], pos_B[:T]

    # Compute metrics
    metrics = compute_all_metrics(result, pos_A, pos_B)
    summary = format_metrics_summary(metrics, clip_id=clip_id, text=text)
    print(f"\n{summary}")

    # Save
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "paired_analysis",
                                  f"clip_{clip_id}_{source}")
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics text
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(summary)

    # Save raw data
    np.savez_compressed(
        os.path.join(output_dir, "paired_vs_solo.npz"),
        torques_paired_A=result['torques_paired_A'],
        torques_paired_B=result['torques_paired_B'],
        torques_solo_A=result['torques_solo_A'],
        torques_solo_B=result['torques_solo_B'],
        sim_jq_paired_A=result['sim_jq_paired_A'],
        sim_jq_paired_B=result['sim_jq_paired_B'],
        sim_jq_solo_A=result['sim_jq_solo_A'],
        sim_jq_solo_B=result['sim_jq_solo_B'],
    )

    # Generate plots
    generate_all_clip_plots(result, clip_id, text, save_dir=output_dir)

    print(f"\n  Results saved to {output_dir}/")
    return result, metrics


def run_batch(source, n_clips, device, output_dir, data_dir, resume, seed):
    """Run batch evaluation."""
    from prepare4.batch_paired_evaluation import main as batch_main

    sys.argv = [
        "batch_paired_evaluation.py",
        "--source", source,
        "--n-clips", str(n_clips),
        "--device", device,
        "--seed", str(seed),
    ]
    if output_dir:
        sys.argv.extend(["--output-dir", output_dir])
    if data_dir:
        sys.argv.extend(["--data-dir", data_dir])
    if resume:
        sys.argv.append("--resume")

    batch_main()


def run_comparison(gt_dir, gen_dir, output_dir=None):
    """Compare GT vs Generated results from batch runs."""
    from prepare4.plot_paired_torques import (
        plot_gt_vs_gen_comparison, plot_sii_distribution,
        plot_per_group_torque_delta,
    )

    gt_path = os.path.join(gt_dir, "paired_eval_results.json")
    gen_path = os.path.join(gen_dir, "paired_eval_results.json")

    if not os.path.exists(gt_path):
        print(f"ERROR: GT results not found at {gt_path}")
        return
    if not os.path.exists(gen_path):
        print(f"ERROR: Generated results not found at {gen_path}")
        return

    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(gen_path) as f:
        gen_data = json.load(f)

    gt_agg = gt_data['aggregated']
    gen_agg = gen_data['aggregated']

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "paired_comparison")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" GT vs Generated Comparison")
    print(f"{'='*60}")
    print(f"  GT:  {gt_agg.get('n_clips', '?')} clips, "
          f"{gt_agg.get('total_frames', '?')} frames")
    print(f"  Gen: {gen_agg.get('n_clips', '?')} clips, "
          f"{gen_agg.get('total_frames', '?')} frames")

    # Print comparison table
    keys_to_compare = [
        ('sii_A', 'SII (A)', True),
        ('sii_B', 'SII (B)', True),
        ('n3lv_mean', 'N3LV mean', False),
        ('bps_paired_A', 'BPS paired (A)', True),
        ('bps_solo_A', 'BPS solo (A)', True),
        ('torque_delta_A_mean', 'TD hinge (A) Nm', False),
        ('root_force_delta_trans_A_mean', 'Root ΔF (A) N', False),
        ('tau_paired_A_hinge_mean', 'τ paired (A) Nm', False),
        ('tau_solo_A_hinge_mean', 'τ solo (A) Nm', False),
    ]

    print(f"\n  {'Metric':35s} {'GT mean':>10s} {'Gen mean':>10s} {'Ratio':>8s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
    for key, label, is_pct in keys_to_compare:
        gt_val = gt_agg.get(key, {}).get('mean', 0)
        gen_val = gen_agg.get(key, {}).get('mean', 0)
        ratio = gen_val / max(gt_val, 1e-8)
        if is_pct:
            print(f"  {label:35s} {gt_val:9.1%} {gen_val:9.1%} {ratio:7.2f}x")
        else:
            print(f"  {label:35s} {gt_val:10.2f} {gen_val:10.2f} {ratio:7.2f}x")

    # Aggregate per-group torque deltas from per-clip data
    groups = ["L_Leg", "R_Leg", "Spine/Torso", "L_Arm", "R_Arm"]
    for data, agg in [(gt_data, gt_agg), (gen_data, gen_agg)]:
        for g in groups:
            key = f'td_A_{g}'
            vals = [c['metrics'][key] for c in data.get('per_clip', [])
                    if key in c.get('metrics', {})]
            if vals:
                agg[key] = {'mean': float(np.mean(vals)),
                            'std': float(np.std(vals))}

    # Print per-group breakdown
    print(f"\n  {'Body Group':20s} {'GT TD (Nm)':>12s} {'Gen TD (Nm)':>12s} {'Ratio':>8s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8}")
    for g in groups:
        key = f'td_A_{g}'
        gt_v = gt_agg.get(key, {}).get('mean', 0)
        gen_v = gen_agg.get(key, {}).get('mean', 0)
        ratio = gen_v / max(gt_v, 1e-8)
        print(f"  {g:20s} {gt_v:12.2f} {gen_v:12.2f} {ratio:7.1f}x")

    # Generate comparison plots
    plot_gt_vs_gen_comparison(
        gt_agg, gen_agg,
        save_path=os.path.join(output_dir, "gt_vs_gen_metrics.png"))

    plot_sii_distribution(
        gt_data.get('per_clip', []), gen_data.get('per_clip', []),
        save_path=os.path.join(output_dir, "sii_distribution.png"))

    plot_per_group_torque_delta(
        gt_agg, gen_agg,
        save_path=os.path.join(output_dir, "per_group_torque_delta.png"))

    # Torque delta distribution (histogram)
    gt_td = [c['metrics']['torque_delta_A_mean']
             for c in gt_data.get('per_clip', [])
             if 'torque_delta_A_mean' in c.get('metrics', {})]
    gen_td = [c['metrics']['torque_delta_A_mean']
              for c in gen_data.get('per_clip', [])
              if 'torque_delta_A_mean' in c.get('metrics', {})]

    if gt_td and gen_td:
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, max(np.percentile(gen_td, 99), 50), 40)
        ax.hist(gt_td, bins=bins, alpha=0.6, color='#2196F3',
                label=f'GT (n={len(gt_td)}, mean={np.mean(gt_td):.1f})',
                density=True)
        ax.hist(gen_td, bins=bins, alpha=0.6, color='#FF5722',
                label=f'Gen (n={len(gen_td)}, mean={np.mean(gen_td):.1f})',
                density=True)
        ax.axvline(np.mean(gt_td), color='#1565C0', linestyle='--', linewidth=2)
        ax.axvline(np.mean(gen_td), color='#BF360C', linestyle='--', linewidth=2)
        ax.set_xlabel("Mean Hinge Torque Delta (Nm)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Paired-Solo Torque Delta (Person A)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "torque_delta_distribution.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Torque delta distribution saved.")

    # BPS distribution
    gt_bps = [c['metrics']['bps_paired_A']
              for c in gt_data.get('per_clip', [])
              if 'bps_paired_A' in c.get('metrics', {})]
    gen_bps = [c['metrics']['bps_paired_A']
               for c in gen_data.get('per_clip', [])
               if 'bps_paired_A' in c.get('metrics', {})]

    if gt_bps and gen_bps:
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, 1, 31)
        ax.hist(gt_bps, bins=bins, alpha=0.6, color='#2196F3',
                label=f'GT (n={len(gt_bps)}, mean={np.mean(gt_bps):.1%})',
                density=True)
        ax.hist(gen_bps, bins=bins, alpha=0.6, color='#FF5722',
                label=f'Gen (n={len(gen_bps)}, mean={np.mean(gen_bps):.1%})',
                density=True)
        ax.set_xlabel("Biomechanical Plausibility Score (% frames violating)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of BPS (Paired Simulation, Person A)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "bps_distribution.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  BPS distribution saved.")

    # Paired vs solo torque comparison (scatter)
    gt_paired = [c['metrics']['tau_paired_A_hinge_mean']
                 for c in gt_data.get('per_clip', [])
                 if 'tau_paired_A_hinge_mean' in c.get('metrics', {})]
    gt_solo = [c['metrics']['tau_solo_A_hinge_mean']
               for c in gt_data.get('per_clip', [])
               if 'tau_solo_A_hinge_mean' in c.get('metrics', {})]
    gen_paired = [c['metrics']['tau_paired_A_hinge_mean']
                  for c in gen_data.get('per_clip', [])
                  if 'tau_paired_A_hinge_mean' in c.get('metrics', {})]
    gen_solo = [c['metrics']['tau_solo_A_hinge_mean']
                for c in gen_data.get('per_clip', [])
                if 'tau_solo_A_hinge_mean' in c.get('metrics', {})]

    if gt_paired and gen_paired:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(gt_solo, gt_paired, alpha=0.5, color='#2196F3',
                   s=30, label=f'GT (n={len(gt_paired)})')
        ax.scatter(gen_solo, gen_paired, alpha=0.5, color='#FF5722',
                   s=30, label=f'Gen (n={len(gen_paired)})')
        lim = max(max(gt_paired + gen_paired), max(gt_solo + gen_solo)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x (no interaction effect)')
        ax.set_xlabel("Solo Mean Hinge |τ| (Nm)")
        ax.set_ylabel("Paired Mean Hinge |τ| (Nm)")
        ax.set_title("Paired vs Solo Torques per Clip (Person A)\n"
                      "Points above y=x → interaction increases torque demand")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.savefig(os.path.join(output_dir, "paired_vs_solo_scatter.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Paired vs solo scatter saved.")

    # Save comparison data
    comparison = {
        'gt': gt_agg,
        'gen': gen_agg,
        'gt_n_clips': gt_agg.get('n_clips', 0),
        'gen_n_clips': gen_agg.get('n_clips', 0),
    }
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Comparison saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Paired vs Solo interaction torque analysis")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--clip-id", type=str,
                      help="Analyze a single clip")
    mode.add_argument("--clip-ids", nargs="+", type=str,
                      help="Analyze multiple specific clips")
    mode.add_argument("--batch", action="store_true",
                      help="Run batch evaluation")
    mode.add_argument("--compare", action="store_true",
                      help="Compare GT vs Generated batch results")

    parser.add_argument("--source", choices=["gt", "generated"],
                        default="gt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-clips", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-dir", default=None,
                        help="GT results dir (for --compare)")
    parser.add_argument("--gen-dir", default=None,
                        help="Generated results dir (for --compare)")

    args = parser.parse_args()

    if args.clip_id:
        analyze_single_clip(args.clip_id, args.source, args.device,
                            args.output_dir, args.data_dir)

    elif args.clip_ids:
        for cid in args.clip_ids:
            analyze_single_clip(cid, args.source, args.device,
                                args.output_dir, args.data_dir)

    elif args.batch:
        run_batch(args.source, args.n_clips, args.device,
                  args.output_dir, args.data_dir, args.resume, args.seed)

    elif args.compare:
        gt_dir = args.gt_dir or os.path.join(
            PROJECT_ROOT, "data", "paired_eval_gt")
        gen_dir = args.gen_dir or os.path.join(
            PROJECT_ROOT, "data", "paired_eval_gen")
        run_comparison(gt_dir, gen_dir, args.output_dir)


if __name__ == "__main__":
    main()
