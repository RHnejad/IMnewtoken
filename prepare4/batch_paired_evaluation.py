"""
batch_paired_evaluation.py — Batch paired-vs-solo torque evaluation.

For each clip, runs 3 simulations (paired + 2 solo) and computes
interaction plausibility metrics.  Supports resume for long runs.

Usage:
    # GT dataset, 200 clips
    python prepare4/batch_paired_evaluation.py --source gt --n-clips 200

    # Generated dataset
    python prepare4/batch_paired_evaluation.py --source generated --n-clips 200

    # Resume interrupted run
    python prepare4/batch_paired_evaluation.py --source gt --resume

    # Specific output directory
    python prepare4/batch_paired_evaluation.py --output-dir data/paired_eval_gt
"""
import os
import sys
import glob
import json
import time
import argparse
import pickle
import warnings
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", message="Custom attribute")

FPS = 30
DT = 1.0 / FPS


def list_gt_clips(data_dir, n_clips=0, seed=42):
    """List GT clip paths from InterHuman dataset."""
    smplx_dir = os.path.join(data_dir, "smplx_322",
                             "person1")
    if not os.path.isdir(smplx_dir):
        smplx_dir = os.path.join(data_dir, "motions_processed", "person1")
    pattern = os.path.join(smplx_dir, "*.npy")
    files = sorted(glob.glob(pattern))

    clip_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if n_clips > 0 and n_clips < len(clip_ids):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(clip_ids), size=n_clips, replace=False)
        clip_ids = [clip_ids[i] for i in sorted(idx)]

    return clip_ids, data_dir


def list_generated_clips(data_dir, n_clips=0, seed=42):
    """List generated clip paths."""
    pattern = os.path.join(data_dir, "*.pkl")
    files = sorted(glob.glob(pattern))
    clip_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

    if n_clips > 0 and n_clips < len(clip_ids):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(clip_ids), size=n_clips, replace=False)
        clip_ids = [clip_ids[i] for i in sorted(idx)]

    return clip_ids, data_dir


def load_clip_data(clip_id, source, data_dir):
    """Load both persons' data for one clip.

    Returns:
        persons: list of 2 dicts with SMPL-X params, or None
    """
    from prepare4.retarget import load_interhuman_pkl

    if source == "generated":
        path = os.path.join(data_dir, f"{clip_id}.pkl")
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            raw = pickle.load(f)
        persons = []
        for pkey in ["person1", "person2"]:
            if pkey not in raw:
                return None
            p = raw[pkey]
            d = {k: p[k].astype(np.float64) for k in
                 ["root_orient", "pose_body", "trans", "betas"]}
            if "positions_zup" in p:
                d["positions_zup"] = p["positions_zup"].astype(np.float64)
            persons.append(d)
        return persons
    else:
        return load_interhuman_pkl(data_dir, clip_id)


def retarget_person(person_data, source, device="cuda:0"):
    """Retarget one person to Newton joint_q.

    Returns:
        joint_q: (T, 76) or None
        betas: (10,) shape params
    """
    from prepare4.retarget import rotation_retarget, ik_retarget

    betas = person_data["betas"]

    if source == "generated":
        if "positions_zup" not in person_data:
            return None, betas
        joint_q, _ = ik_retarget(
            person_data["positions_zup"], betas,
            ik_iters=50, device=device, sequential=True,
        )
    else:
        joint_q = rotation_retarget(
            person_data["root_orient"],
            person_data["pose_body"],
            person_data["trans"],
            betas,
        )
        joint_q = joint_q[::2]  # 60fps → 30fps

    return joint_q, betas


def process_clip_paired(clip_id, source, data_dir, device="cuda:0"):
    """Process one clip: run all 3 scenarios + compute metrics.

    Returns:
        result_dict: dict with torques, metrics, and metadata, or None
    """
    from prepare4.paired_simulation import compute_paired_vs_solo
    from prepare4.interaction_metrics import compute_all_metrics
    from prepare4.run_full_analysis import load_positions_zup

    persons = load_clip_data(clip_id, source, data_dir)
    if persons is None or len(persons) < 2:
        return None

    # Retarget both persons
    jq_A, betas_A = retarget_person(persons[0], source, device=device)
    jq_B, betas_B = retarget_person(persons[1], source, device=device)

    if jq_A is None or jq_B is None:
        return None

    T = min(jq_A.shape[0], jq_B.shape[0])
    if T < 11:
        return None

    jq_A = jq_A[:T]
    jq_B = jq_B[:T]

    # Run 3 scenarios
    try:
        result = compute_paired_vs_solo(
            jq_A, jq_B, betas_A, betas_B,
            dt=DT, device=device, verbose=False,
        )
    except Exception as e:
        print(f"    ERROR in simulation for clip {clip_id}: {e}")
        return None

    # Load positions for CTC metric
    pos_A, pos_B = load_positions_zup(clip_id, source)
    if pos_A is not None and pos_B is not None:
        pos_A = pos_A[:T]
        pos_B = pos_B[:T]

    # Compute metrics
    try:
        metrics = compute_all_metrics(result, pos_A, pos_B)
    except Exception as e:
        print(f"    ERROR computing metrics for clip {clip_id}: {e}")
        return None

    return {
        'clip_id': clip_id,
        'source': source,
        'n_frames': T,
        'metrics': metrics,
        # Store torque summaries (not full arrays for memory)
        'torques_paired_A_hinge_mean': float(np.abs(result['torques_paired_A'][:, 6:]).mean()),
        'torques_solo_A_hinge_mean': float(np.abs(result['torques_solo_A'][:, 6:]).mean()),
        'torques_paired_B_hinge_mean': float(np.abs(result['torques_paired_B'][:, 6:]).mean()),
        'torques_solo_B_hinge_mean': float(np.abs(result['torques_solo_B'][:, 6:]).mean()),
    }


def batch_evaluate(clip_ids, source, data_dir, device="cuda:0",
                   save_dir=None, resume=False):
    """Process multiple clips with resume support.

    Args:
        clip_ids: list of clip ID strings
        source: 'gt' or 'generated'
        data_dir: root data directory
        device: CUDA device
        save_dir: directory for intermediate/final results
        resume: skip already-processed clips

    Returns:
        all_results: list of per-clip result dicts
    """
    all_results = []
    done_clips = set()

    if resume and save_dir:
        intermediate_path = os.path.join(save_dir, "intermediate_results.json")
        if os.path.exists(intermediate_path):
            with open(intermediate_path) as f:
                saved = json.load(f)
            all_results = saved.get("results", [])
            done_clips = {r['clip_id'] for r in all_results}
            print(f"Resumed: {len(done_clips)} clips already processed")

    n_errors = 0
    t_start = time.time()

    for i, clip_id in enumerate(clip_ids):
        if clip_id in done_clips:
            continue

        elapsed = time.time() - t_start
        done_count = i + 1
        rate = done_count / max(elapsed, 1)
        remaining_count = len(clip_ids) - done_count
        eta = remaining_count / max(rate, 0.001)
        print(f"[{done_count}/{len(clip_ids)}] clip={clip_id}  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

        result = process_clip_paired(clip_id, source, data_dir, device=device)
        if result is None:
            n_errors += 1
            continue

        all_results.append(result)
        done_clips.add(clip_id)

        # Print per-clip summary
        m = result['metrics']
        print(f"    SII: A={m['sii_A']:.0%} B={m['sii_B']:.0%}  "
              f"N3LV={m['n3lv_mean']:.3f}  "
              f"TD_hinge: A={m['torque_delta_A_mean']:.1f} B={m['torque_delta_B_mean']:.1f} Nm")

        # Save intermediate every 20 clips
        if save_dir and len(all_results) % 20 == 0:
            _save_intermediate(all_results, save_dir)

    total_time = time.time() - t_start
    print(f"\nDone: {len(clip_ids)} clips, {n_errors} errors, "
          f"{len(all_results)} successful, {total_time:.0f}s elapsed")

    return all_results


def _save_intermediate(results, save_dir):
    """Save intermediate results for resume."""
    os.makedirs(save_dir, exist_ok=True)
    # Serialize: convert non-serializable metric values
    serializable = []
    for r in results:
        sr = dict(r)
        sr['metrics'] = _serialize_metrics(r['metrics'])
        serializable.append(sr)
    path = os.path.join(save_dir, "intermediate_results.json")
    with open(path, "w") as f:
        json.dump({"results": serializable}, f, indent=2)


def _serialize_metrics(metrics):
    """Convert metrics dict to JSON-serializable form."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {str(kk): float(vv) if isinstance(vv, (np.floating, np.integer))
                       else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def compute_aggregated_statistics(all_results):
    """Aggregate metrics across all clips.

    Returns:
        agg: dict with mean/std/median/P90/P99 for each metric
    """
    if not all_results:
        return {}

    # Collect scalar metrics
    scalar_keys = [
        'torque_delta_A_mean', 'torque_delta_B_mean',
        'torque_delta_A_max', 'torque_delta_B_max',
        'root_force_delta_trans_A_mean', 'root_force_delta_trans_B_mean',
        'n3lv_mean', 'n3lv_median', 'n3lv_p90',
        'sii_A', 'sii_B',
        'solo_root_force_A_mean', 'solo_root_force_B_mean',
        'bps_paired_A', 'bps_paired_B',
        'bps_solo_A', 'bps_solo_B',
        'tau_paired_A_hinge_mean', 'tau_paired_B_hinge_mean',
        'tau_solo_A_hinge_mean', 'tau_solo_B_hinge_mean',
    ]

    agg = {}
    for key in scalar_keys:
        values = []
        for r in all_results:
            m = r['metrics']
            if key in m:
                v = m[key]
                if isinstance(v, (int, float, np.floating, np.integer)):
                    values.append(float(v))
        if values:
            arr = np.array(values)
            agg[key] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'median': float(np.median(arr)),
                'p90': float(np.percentile(arr, 90)),
                'p99': float(np.percentile(arr, 99)),
                'min': float(arr.min()),
                'max': float(arr.max()),
            }

    # CTC (only clips that have it)
    ctc_values = [float(r['metrics']['ctc']) for r in all_results
                  if 'ctc' in r['metrics']]
    if ctc_values:
        arr = np.array(ctc_values)
        agg['ctc'] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'median': float(np.median(arr)),
        }

    agg['n_clips'] = len(all_results)
    agg['total_frames'] = sum(r['n_frames'] for r in all_results)

    return agg


def print_aggregated(agg, source=""):
    """Pretty-print aggregated statistics."""
    print(f"\n{'='*70}")
    print(f" PAIRED vs SOLO EVALUATION — {source.upper()}")
    print(f" {agg.get('n_clips', 0)} clips, "
          f"{agg.get('total_frames', 0):,} total frames")
    print(f"{'='*70}")

    def _fmt(key, label, unit=""):
        if key not in agg:
            return
        s = agg[key]
        print(f"  {label:40s}: "
              f"mean={s['mean']:8.2f}  std={s['std']:6.2f}  "
              f"med={s['median']:7.2f}  P90={s['p90']:7.2f}  {unit}")

    print("\n  --- Torque Delta (paired - solo) ---")
    _fmt('torque_delta_A_mean', 'Hinge delta mean (A)', 'Nm')
    _fmt('torque_delta_B_mean', 'Hinge delta mean (B)', 'Nm')

    print("\n  --- Root Force Delta ---")
    _fmt('root_force_delta_trans_A_mean', 'Root trans delta (A)', 'N')
    _fmt('root_force_delta_trans_B_mean', 'Root trans delta (B)', 'N')

    print("\n  --- Newton 3rd Law Violation ---")
    _fmt('n3lv_mean', 'N3LV (mean per clip)')

    print("\n  --- Solo Impossibility Index ---")
    _fmt('sii_A', 'SII (A)')
    _fmt('sii_B', 'SII (B)')

    print("\n  --- Biomechanical Plausibility ---")
    _fmt('bps_paired_A', 'BPS paired (A)')
    _fmt('bps_solo_A', 'BPS solo (A)')

    print("\n  --- Torque Magnitudes ---")
    _fmt('tau_paired_A_hinge_mean', 'Paired hinge mean (A)', 'Nm')
    _fmt('tau_solo_A_hinge_mean', 'Solo hinge mean (A)', 'Nm')

    if 'ctc' in agg:
        print(f"\n  --- Contact-Torque Correlation ---")
        print(f"  {'CTC':40s}: mean={agg['ctc']['mean']:.3f}  "
              f"std={agg['ctc']['std']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch paired-vs-solo torque evaluation")
    parser.add_argument("--source", choices=["gt", "generated"],
                        default="gt")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: auto-detect)")
    parser.add_argument("--n-clips", type=int, default=100,
                        help="Number of clips to sample (0 = all)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/paired_eval_{source})")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved intermediate results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif args.source == "generated":
        data_dir = os.path.join(PROJECT_ROOT, "data", "generated", "interhuman")
    else:
        data_dir = os.path.join(PROJECT_ROOT, "data", "InterHuman")

    # Output directory
    if args.output_dir:
        save_dir = args.output_dir
    else:
        save_dir = os.path.join(PROJECT_ROOT, "data",
                                f"paired_eval_{args.source}")

    os.makedirs(save_dir, exist_ok=True)

    # List clips
    if args.source == "generated":
        clip_ids, data_dir = list_generated_clips(
            data_dir, n_clips=args.n_clips, seed=args.seed)
    else:
        clip_ids, data_dir = list_gt_clips(
            data_dir, n_clips=args.n_clips, seed=args.seed)

    print(f"Source: {args.source}")
    print(f"Data dir: {data_dir}")
    print(f"Clips: {len(clip_ids)}")
    print(f"Output: {save_dir}")

    # Run evaluation
    results = batch_evaluate(
        clip_ids, args.source, data_dir,
        device=args.device, save_dir=save_dir, resume=args.resume,
    )

    # Aggregate and save
    agg = compute_aggregated_statistics(results)
    print_aggregated(agg, source=args.source)

    # Save final results
    final_results = []
    for r in results:
        sr = dict(r)
        sr['metrics'] = _serialize_metrics(r['metrics'])
        final_results.append(sr)

    with open(os.path.join(save_dir, "paired_eval_results.json"), "w") as f:
        json.dump({
            "source": args.source,
            "n_clips": len(results),
            "aggregated": agg,
            "per_clip": final_results,
        }, f, indent=2)

    print(f"\nResults saved to {save_dir}/paired_eval_results.json")


if __name__ == "__main__":
    main()
