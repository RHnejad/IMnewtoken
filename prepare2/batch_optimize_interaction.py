"""
Batch ΔT optimization for interaction torques across the entire dataset.

Wraps optimize_interaction.py's InteractionOptimizer for headless batch runs
with multi-GPU parallelization and date-stamped output directories.

Pipeline per clip:
  1. Ensure solo torques exist (compute via inverse dynamics if needed)
  2. Run ΔT optimization (SolverFeatherstone + wp.Tape backprop)
  3. Save delta + full torques to date-stamped output directory

Why is optimization slow?
  Each iteration involves:
    - Forward pass: window_size × sim_substeps physics steps through wp.Tape
      (default: 10 frames × 24 substeps = 240 physics evaluations)
    - Backward pass: same 240 steps in reverse for gradient computation
    - Per clip total: epochs × n_windows iterations
      (e.g., 50 epochs × 20 windows = 1000 iterations → 480,000 physics evals)
  Best practices applied here:
    - Multi-GPU: distribute clips round-robin across GPUs (spawn workers)
    - Early stopping: stop a clip when loss stops improving
    - Resume: skip clips with existing outputs

Output: data/batch_optimize_interaction/{YYYY-MM-DD_HHMMSS}/
  Saved files per clip:
    {clip_id}_person{p}_delta_torques.npy   — learned ΔT
    {clip_id}_person{p}_torques_full.npy    — solo + ΔT
    progress.json                           — tracking file

Usage:
    # All test clips, both GPUs
    python prepare2/batch_optimize_interaction.py \\
        --dataset interhuman --split test --gpus 0 1

    # Single GPU, fewer epochs (faster)
    python prepare2/batch_optimize_interaction.py \\
        --dataset interhuman --split test --gpus 1 --epochs 30

    # Custom learning rate and window size
    python prepare2/batch_optimize_interaction.py \\
        --dataset interhuman --split test --gpus 0 1 \\
        --lr 0.5 --window 20 --epochs 30

    # Test with a few clips first
    python prepare2/batch_optimize_interaction.py \\
        --dataset interhuman --split test --gpus 1 --max-clips 5

    # Resume interrupted run (reuse output dir)
    python prepare2/batch_optimize_interaction.py \\
        --dataset interhuman --split test --gpus 0 1 \\
        --output-dir data/batch_optimize_interaction/2026-02-24_010000
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from datetime import datetime

import numpy as np

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════
#  Worker functions (run in spawned processes, separate CUDA ctx)
# ═══════════════════════════════════════════════════════════════

def _worker_init(gpu_id):
    """Initialize warp + newton in a spawned worker process."""
    import warnings
    warnings.filterwarnings("ignore", message="Custom attribute")
    import warp as wp
    wp.config.verbose = False
    wp.init()
    # Store GPU id for this worker
    _worker_init._gpu = f"cuda:{gpu_id}"


def _worker_optimize_clip(args_tuple):
    """
    Optimize a single clip in a worker process.

    Returns:
        (clip_id, success, elapsed_sec, final_loss, n_iters, message)
    """
    (clip_id, data_dir, output_dir, fps, sim_freq, downsample, lr, reg_lambda,
     window, epochs, early_stop_patience) = args_tuple

    device = _worker_init._gpu

    import warnings
    warnings.filterwarnings("ignore", message="Custom attribute")

    try:
        import warp as wp
        import newton
        from prepare2.optimize_interaction import InteractionOptimizer
        import argparse as _argparse

        # Check if output already exists
        out_check = os.path.join(
            output_dir, f"{clip_id}_person0_delta_torques.npy"
        )
        if os.path.exists(out_check):
            return (clip_id, True, 0.0, 0.0, 0, "skipped (exists)")

        # Ensure solo torques exist — check both naming conventions
        for p_idx in [0, 1]:
            solo_path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_torques_solo.npy"
            )
            batch_path = os.path.join(
                data_dir, f"{clip_id}_person{p_idx}_batch_sim_solo_torques.npy"
            )
            if not os.path.exists(solo_path):
                if os.path.exists(batch_path):
                    # Create symlink from batch name → solo name
                    os.symlink(os.path.abspath(batch_path), solo_path)
                else:
                    # Compute inverse dynamics torques
                    from prepare2.retarget import get_or_create_xml
                    from prepare2.compute_torques import inverse_dynamics

                    jq_path = os.path.join(
                        data_dir, f"{clip_id}_person{p_idx}_joint_q.npy"
                    )
                    betas_path = os.path.join(
                        data_dir, f"{clip_id}_person{p_idx}_betas.npy"
                    )
                    if not os.path.exists(jq_path) or not os.path.exists(betas_path):
                        return (clip_id, False, 0.0, 0.0, 0,
                                f"missing input: joint_q or betas for person{p_idx}")

                    joint_q = np.load(jq_path).astype(np.float32)
                    betas = np.load(betas_path)
                    xml_path = get_or_create_xml(betas)

                    builder = newton.ModelBuilder(
                        up_axis=newton.Axis.Z, gravity=-9.81
                    )
                    builder.add_mjcf(xml_path, enable_self_collisions=False)
                    builder.add_ground_plane()
                    model_solo = builder.finalize(device=device)

                    tau, _, _ = inverse_dynamics(
                        model_solo, joint_q, fps, device=device
                    )
                    np.save(solo_path, tau)
                    print(f"  [{clip_id}] Computed torques → "
                          f"person{p_idx}_torques_solo.npy")

        # Build optimizer args
        opt_args = _argparse.Namespace(
            clip=clip_id,
            data_dir=data_dir,
            mode="optimize",
            fps=fps,
            sim_freq=sim_freq,
            downsample=downsample,
            lr=lr,
            reg_lambda=reg_lambda,
            window=window,
            device=device,
        )

        # Stub viewer
        class _StubViewer:
            def set_model(self, m): pass
            def set_camera(self, *a, **kw): pass
            def begin_frame(self, t): pass
            def log_state(self, s): pass
            def log_scalar(self, k, v): pass
            def end_frame(self): pass
            def close(self): pass
            def register_ui_callback(self, *a, **kw): pass

        opt = InteractionOptimizer(_StubViewer(), opt_args)

        max_iters = epochs * opt.n_windows
        t0 = time.time()

        # Early stopping state
        best_epoch_loss = float('inf')
        patience_counter = 0
        last_epoch = -1

        for iteration in range(max_iters):
            opt.step()

            # Check early stopping at epoch boundaries
            current_epoch = opt.epoch
            if current_epoch != last_epoch and current_epoch > 0:
                # Average loss over last epoch
                n_win = opt.n_windows
                if len(opt.loss_history) >= n_win:
                    epoch_loss = np.mean(opt.loss_history[-n_win:])
                    if epoch_loss < best_epoch_loss * 0.999:
                        best_epoch_loss = epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if (early_stop_patience > 0
                            and patience_counter >= early_stop_patience):
                        print(f"  [{clip_id}] Early stop at epoch "
                              f"{current_epoch} (patience={early_stop_patience},"
                              f" loss={epoch_loss:.6f})")
                        break
                last_epoch = current_epoch

        elapsed = time.time() - t0
        final_loss = opt.loss_history[-1] if opt.loss_history else 0.0

        # Save results to output_dir (NOT data_dir)
        opt.save_results(output_dir)

        return (clip_id, True, elapsed, final_loss,
                opt.train_iter, f"done in {elapsed:.1f}s")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (clip_id, False, 0.0, 0.0, 0, str(e))


# ═══════════════════════════════════════════════════════════════
#  Clip discovery
# ═══════════════════════════════════════════════════════════════

def list_clips_with_data(data_dir, split_file=None):
    """List clips that have joint_q + betas for both persons."""
    clips_p0 = set()
    clips_p1 = set()
    for f in os.listdir(data_dir):
        if f.endswith("_person0_joint_q.npy"):
            clips_p0.add(f.replace("_person0_joint_q.npy", ""))
        elif f.endswith("_person1_joint_q.npy"):
            clips_p1.add(f.replace("_person1_joint_q.npy", ""))

    all_clips = clips_p0 & clips_p1

    if split_file and os.path.exists(split_file):
        with open(split_file) as f:
            split_ids = {line.strip() for line in f if line.strip()}
        all_clips = all_clips & split_ids

    return sorted(all_clips)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch ΔT optimization for interaction torques "
                    "(multi-GPU, date-stamped output)"
    )
    parser.add_argument("--dataset", type=str, default="interhuman",
                        choices=["interhuman", "interx"])
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Only process clips in this split")
    parser.add_argument("--clip", type=str, default=None,
                        help="Single clip ID (for testing)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPU IDs to use (e.g., --gpus 0 1)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max optimization epochs per clip")
    parser.add_argument("--lr", type=float, default=1.0,
                        help="Learning rate")
    parser.add_argument("--reg-lambda", type=float, default=0.01,
                        help="L2 regularization weight on ΔT")
    parser.add_argument("--window", type=int, default=10,
                        help="Frames per optimization window")
    parser.add_argument("--fps", type=int, default=30,
                        help="Data playback FPS (default 30 = InterMask eval rate)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample loaded data by this factor "
                             "(2 = 60->30fps to match InterMask)")
    parser.add_argument("--sim-freq", type=int, default=120,
                        help="Simulation frequency in Hz (default 120 for batch). "
                             "Lower = faster but less accurate. "
                             "480 = full accuracy, 120 = 4x faster.")
    parser.add_argument("--early-stop", type=int, default=5,
                        help="Stop if loss doesn't improve for N epochs "
                             "(0 = disabled)")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Process at most N clips (for testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Resume into existing output dir "
                             "(default: create new date-stamped dir)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    # ── Data directory ───────────────────────────────────
    data_dir = os.path.join(
        PROJECT_ROOT, "data", "retargeted_v2", args.dataset
    )
    if not os.path.isdir(data_dir):
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # ── Output directory (date-stamped) ──────────────────
    if args.output_dir:
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(
            PROJECT_ROOT, "data", "batch_optimize_interaction",
            f"{args.dataset}_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # ── Get clips ────────────────────────────────────────
    if args.clip:
        clips = [args.clip]
    else:
        split_file = None
        if args.split:
            gt_root = os.path.join(PROJECT_ROOT, "data", "InterHuman")
            split_file = os.path.join(gt_root, "split", f"{args.split}.txt")
            if not os.path.exists(split_file):
                print(f"Split file not found: {split_file}")
                sys.exit(1)
        clips = list_clips_with_data(data_dir, split_file)

    if args.max_clips:
        clips = clips[:args.max_clips]

    # ── Skip already processed (resume) ──────────────────
    if not args.force:
        remaining = []
        skipped = 0
        for c in clips:
            out_path = os.path.join(
                output_dir, f"{c}_person0_delta_torques.npy"
            )
            if os.path.exists(out_path):
                skipped += 1
            else:
                remaining.append(c)
        if skipped:
            print(f"Resuming: {skipped} clips already done, "
                  f"{len(remaining)} remaining")
        clips = remaining

    if not clips:
        print("Nothing to process!")
        sys.exit(0)

    # ── Save run config ──────────────────────────────────
    config = {
        "script": "batch_optimize_interaction.py",
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "gpus": args.gpus,
        "epochs": args.epochs,
        "lr": args.lr,
        "reg_lambda": args.reg_lambda,
        "window": args.window,
        "sim_freq": args.sim_freq,
        "fps": args.fps,
        "downsample": args.downsample,
        "early_stop": args.early_stop,
        "n_clips": len(clips),
        "data_dir": data_dir,
        "output_dir": output_dir,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Batch ΔT Optimization")
    print(f"{'='*60}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Split:      {args.split or 'all'}")
    print(f"  Clips:      {len(clips)}")
    print(f"  GPUs:       {args.gpus}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Window:     {args.window}")
    print(f"  Downsample: {args.downsample}x (data {args.fps * args.downsample}→{args.fps} fps)")
    print(f"  Sim freq:   {args.sim_freq} Hz ({args.sim_freq // args.fps} substeps/frame)")
    print(f"  Early stop: {args.early_stop} epochs")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Build task list ──────────────────────────────────
    tasks = [
        (clip_id, data_dir, output_dir, args.fps, args.sim_freq,
         args.downsample, args.lr, args.reg_lambda, args.window,
         args.epochs, args.early_stop)
        for clip_id in clips
    ]

    # ── Distribute across GPUs ───────────────────────────
    n_gpus = len(args.gpus)
    t_start = time.time()
    total_done = 0
    total_fail = 0
    total_skip = 0

    if n_gpus == 1:
        # Single GPU: simple sequential in one process
        # (avoid spawn overhead)
        gpu_id = args.gpus[0]
        _worker_init(gpu_id)

        for i, task in enumerate(tasks):
            clip_id = task[0]
            elapsed = time.time() - t_start
            done = total_done + total_fail
            rate = (done / elapsed * 3600) if elapsed > 0 and done > 0 else 0
            eta_h = ((len(tasks) - i) / rate) if rate > 0 else 0

            print(f"\n[{i+1}/{len(tasks)}] Clip {clip_id} "
                  f"(done={total_done}, fail={total_fail}, "
                  f"rate={rate:.0f}/h, ETA={eta_h:.1f}h)")

            result = _worker_optimize_clip(task)
            clip_id, success, elapsed_s, loss, n_iters, msg = result

            if "skipped" in msg:
                total_skip += 1
                print(f"  SKIP: {msg}")
            elif success:
                total_done += 1
                print(f"  OK: {msg} (loss={loss:.6f}, iters={n_iters})")
            else:
                total_fail += 1
                print(f"  FAIL: {msg}")

            # Update progress file
            _save_progress(output_dir, total_done, total_fail,
                           total_skip, len(tasks), t_start)
    else:
        # Multi-GPU: one pool per GPU, distribute clips round-robin
        # Each pool has 1 worker (1 clip at a time per GPU)
        gpu_task_lists = [[] for _ in range(n_gpus)]
        for i, task in enumerate(tasks):
            gpu_task_lists[i % n_gpus].append(task)

        print(f"Distributing {len(tasks)} clips across {n_gpus} GPUs:")
        for g, gpu_id in enumerate(args.gpus):
            print(f"  cuda:{gpu_id}: {len(gpu_task_lists[g])} clips")

        # Use spawn context for separate CUDA contexts
        ctx = mp.get_context("spawn")

        # Create one pool per GPU (each with 1 worker)
        pools = []
        async_results = []
        for g, gpu_id in enumerate(args.gpus):
            pool = ctx.Pool(
                1,
                initializer=_worker_init,
                initargs=(gpu_id,),
            )
            # Submit all tasks for this GPU
            result = pool.map_async(
                _worker_optimize_clip, gpu_task_lists[g]
            )
            pools.append(pool)
            async_results.append(result)

        # Collect results
        for g, (pool, result) in enumerate(zip(pools, async_results)):
            gpu_id = args.gpus[g]
            try:
                results = result.get()  # blocks until all done
                for clip_id, success, elapsed_s, loss, n_iters, msg in results:
                    if "skipped" in msg:
                        total_skip += 1
                    elif success:
                        total_done += 1
                        print(f"  [cuda:{gpu_id}] {clip_id}: {msg} "
                              f"(loss={loss:.6f})")
                    else:
                        total_fail += 1
                        print(f"  [cuda:{gpu_id}] {clip_id}: FAIL — {msg}")
            except Exception as e:
                print(f"  [cuda:{gpu_id}] Pool error: {e}")
                total_fail += len(gpu_task_lists[g])
            finally:
                pool.close()
                pool.join()

    # ── Summary ──────────────────────────────────────────
    elapsed = time.time() - t_start
    _save_progress(output_dir, total_done, total_fail,
                   total_skip, len(tasks), t_start)

    print(f"\n{'='*60}")
    print(f"BATCH OPTIMIZATION COMPLETE")
    print(f"  Processed: {total_done}")
    print(f"  Skipped:   {total_skip}")
    print(f"  Failed:    {total_fail}")
    print(f"  Time:      {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}")


def _save_progress(output_dir, done, fail, skip, total, t_start):
    """Save progress tracking file."""
    elapsed = time.time() - t_start
    progress = {
        "done": done,
        "failed": fail,
        "skipped": skip,
        "total": total,
        "elapsed_sec": elapsed,
        "updated": datetime.now().isoformat(),
    }
    path = os.path.join(output_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


if __name__ == "__main__":
    main()
