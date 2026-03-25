"""
Batch compute solo torques + simulate to get positions for FID evaluation.

For each clip in the dataset:
  1. Compute inverse dynamics torques (if not already saved)
  2. Simulate both persons with solo torques (SolverMuJoCo + 10% PD)
  3. Extract 22-joint positions at each frame
  4. Save as {clip_id}_two_person_sim_positions.npy (T, 2, 22, 3)

This generates the data needed for eval_tests/eval_fid.py Case 3.

Usage:
    # Process all test-split clips
    python prepare2/batch_sim_solo.py --dataset interhuman --split test

    # Process all clips
    python prepare2/batch_sim_solo.py --dataset interhuman

    # Single clip
    python prepare2/batch_sim_solo.py --clip 1000

    # Resume (skips clips with existing sim_positions)
    python prepare2/batch_sim_solo.py --dataset interhuman --split test --resume

    # Skip torque computation (assume already computed)
    python prepare2/batch_sim_solo.py --dataset interhuman --split test --skip-torques
"""
import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import traceback
from datetime import datetime

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare2.retarget import get_or_create_xml, SMPL_TO_NEWTON, N_SMPL_JOINTS
from prepare2.compute_torques import inverse_dynamics
from prepare2.pd_utils import (
    DOFS_PER_PERSON, COORDS_PER_PERSON, BODIES_PER_PERSON,
    BODY_GAINS, DEFAULT_SIM_FREQ, DEFAULT_TORQUE_LIMIT,
    build_model, setup_model_properties, create_mujoco_solver,
    build_pd_gains, compute_all_pd_torques_np,
    extract_positions_from_state, init_state,
    downsample_trajectory,
    create_contact_sensors, update_contact_sensors,
)


# extract_positions_from_state imported from pd_utils


def simulate_clip(clip_id, data_dir, fps=30, gain_scale=1.0,
                  device="cuda:0", skip_torques=False, force=False,
                  downsample=2, output_dir=None, torque_mode="pd",
                  pd_scale=1.0):
    """
    Simulate a single clip and return positions.

    Torque modes:
      - "pd":    Full PD tracking only (no precomputed torques). Fast and
                 stable. Matches simulate_torques.py default behavior.
      - "id+pd": Precomputed inverse dynamics torques + PD correction.
                 I.D. torques are computed without ground contacts, so
                 tracking is poor (characters fly). Use for research only.

    Args:
        data_dir: directory with input files (joint_q, betas)
        output_dir: directory for output files (if None, uses data_dir)
        torque_mode: "pd" or "id+pd"
        pd_scale: PD gain multiplier (1.0 for full PD, 0.1 for 10% correction)

    Returns:
        positions: (T, 2, 22, 3) or None on error
    """
    save_dir = output_dir or data_dir

    # ── Check if output exists ──────────────────────────
    out_path = os.path.join(save_dir, f"{clip_id}_two_person_sim_positions.npy")
    if os.path.exists(out_path) and not force:
        return "skip"

    # ── Check input files ───────────────────────────────
    jq_paths = []
    betas_paths = []
    torque_paths = []
    for p in [0, 1]:
        jq_p = os.path.join(data_dir, f"{clip_id}_person{p}_joint_q.npy")
        beta_p = os.path.join(data_dir, f"{clip_id}_person{p}_betas.npy")
        tau_p = os.path.join(save_dir, f"{clip_id}_person{p}_batch_sim_solo_torques.npy")
        if not os.path.exists(jq_p) or not os.path.exists(beta_p):
            return None  # missing input
        jq_paths.append(jq_p)
        betas_paths.append(beta_p)
        torque_paths.append(tau_p)

    # ── Load data ──────────────────────────────────────
    all_jq = [np.load(p).astype(np.float32) for p in jq_paths]
    all_betas = [np.load(p) for p in betas_paths]

    # Downsample from data FPS to target FPS (e.g. 60→30)
    if downsample > 1:
        all_jq = [downsample_trajectory(jq, downsample) for jq in all_jq]

    T = min(jq.shape[0] for jq in all_jq)

    # ── Compute or load torques (only for id+pd mode) ──
    all_torques = []
    if torque_mode == "id+pd":
        for p in [0, 1]:
            if os.path.exists(torque_paths[p]) and not force:
                tau = np.load(torque_paths[p]).astype(np.float32)
                all_torques.append(tau)
            elif skip_torques:
                return None  # no torques and told not to compute
            else:
                # Compute inverse dynamics for this person
                xml_path = get_or_create_xml(all_betas[p])
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
                builder.add_mjcf(xml_path, enable_self_collisions=False)
                builder.add_ground_plane()
                model_solo = builder.finalize(device=device)

                tau, _, _ = inverse_dynamics(
                    model_solo, all_jq[p], fps, device=device
                )
                # Save torques
                np.save(torque_paths[p], tau)
                print(f"  Saved torques → {os.path.basename(torque_paths[p])}")
                all_torques.append(tau)

    # ── Build two-person Newton model ──────────────────
    model, _ = build_model(all_betas, device=device, with_ground=True)

    n_dof = model.joint_dof_count
    n_coords = model.joint_coord_count
    n_persons = 2

    # ── Configure model (passive springs, armature) ─────
    setup_model_properties(model, n_persons, device=device)

    # ── PD gains ────────────────────────────────────────
    kp, kd = build_pd_gains(model, n_persons, gain_scale=gain_scale)

    # ── Solver ──────────────────────────────────────────
    sim_freq = DEFAULT_SIM_FREQ
    sim_substeps = sim_freq // fps
    sim_dt = 1.0 / sim_freq
    torque_limit = DEFAULT_TORQUE_LIMIT

    solver = create_mujoco_solver(model, n_persons)

    # ── Contact sensors (feet + hands) ──────────────────
    sensor_dict = create_contact_sensors(
        model, solver, n_persons, verbose=False
    )

    state_0, state_1, control = init_state(
        model, all_jq, n_persons, device=device
    )

    # ── Simulate ────────────────────────────────────────
    positions = np.zeros((T, n_persons, N_SMPL_JOINTS, 3), dtype=np.float32)
    # Contact force recordings (per frame)
    foot_forces_all = []
    hand_forces_all = []

    use_id = torque_mode == "id+pd" and len(all_torques) == n_persons
    precomputed = all_torques if use_id else None

    for t in range(T):
        # Compute PD torques once per control frame
        cq = state_0.joint_q.numpy()
        cqd = state_0.joint_qd.numpy()

        tau = compute_all_pd_torques_np(
            cq, cqd, all_jq, t, kp, kd, n_persons,
            pd_scale=pd_scale, torque_limit=torque_limit,
            precomputed_torques=precomputed,
            precomputed_pd_scale=pd_scale if use_id else 1.0,
        )
        tau_wp = wp.array(tau, dtype=wp.float32, device=device)

        # Run substeps with same torques
        for _ in range(sim_substeps):
            control.joint_f = tau_wp
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        # Read contact sensors (once per control frame)
        cf = update_contact_sensors(solver, state_0, sensor_dict)
        if cf is not None:
            foot_forces_all.append(cf['foot_forces'].copy())
            if cf['hand_forces'] is not None:
                hand_forces_all.append(cf['hand_forces'].copy())

        # Extract positions
        positions[t] = extract_positions_from_state(state_0, n_persons)

    # ── Save ────────────────────────────────────────────
    np.save(out_path, positions)

    # Save contact force data alongside positions
    if foot_forces_all:
        foot_path = os.path.join(
            save_dir, f"{clip_id}_foot_contact_forces.npy"
        )
        np.save(foot_path, np.array(foot_forces_all))
    if hand_forces_all:
        hand_path = os.path.join(
            save_dir, f"{clip_id}_hand_contact_forces.npy"
        )
        np.save(hand_path, np.array(hand_forces_all))

    return positions


def plot_clip_forces(clip_id, save_dir, fps=30):
    """
    Generate a matplotlib figure showing contact forces over time.

    Reads the saved _foot_contact_forces.npy and _hand_contact_forces.npy
    files and produces a PNG plot.

    Args:
        clip_id: clip identifier
        save_dir: directory containing saved force data
        fps: motion FPS for x-axis time labels
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping force plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Contact Forces — Clip {clip_id}", fontsize=14)

    # ── Foot GRF ────────────────────────────────────────
    foot_path = os.path.join(save_dir, f"{clip_id}_foot_contact_forces.npy")
    if os.path.exists(foot_path):
        foot_f = np.load(foot_path)  # (T, n_groups, n_counterparts, 3)
        T = foot_f.shape[0]
        t_sec = np.arange(T) / fps

        # Sum vertical GRF per person
        # Column 0 = total (include_total=True), use [g, 0, 2] for Fz
        n_groups = foot_f.shape[1]
        half = n_groups // 2
        for p in range(2):
            start, end = p * half, (p + 1) * half
            # Sum Fz across foot groups for this person
            person_fz = np.array([
                sum(abs(float(foot_f[t, g, 0, 2]))
                    for g in range(start, min(end, n_groups)))
                for t in range(T)
            ])
            axes[0].plot(t_sec, person_fz, label=f"Person {p}", alpha=0.8)

        axes[0].set_ylabel("GRF Fz (N)")
        axes[0].set_title("Ground Reaction Forces (vertical)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # ── Hand contacts ───────────────────────────────────
    hand_path = os.path.join(save_dir, f"{clip_id}_hand_contact_forces.npy")
    if os.path.exists(hand_path):
        hand_f = np.load(hand_path)  # (T, n_groups, n_counterparts, 3)
        T = hand_f.shape[0]
        t_sec = np.arange(T) / fps

        for g in range(hand_f.shape[1]):
            mag = np.array([
                float(np.linalg.norm(hand_f[t, g, 0]))
                for t in range(T)
            ])
            person = g // 2
            side = "L" if g % 2 == 0 else "R"
            axes[1].plot(t_sec, mag,
                         label=f"P{person} {side}_Hand", alpha=0.8)

        axes[1].set_ylabel("Force (N)")
        axes[1].set_title("Hand Contact Forces")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{clip_id}_contact_forces.png")
    plt.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"  Saved force plot → {os.path.basename(plot_path)}")


def list_clips(data_dir, split_file=None):
    """List clip IDs, optionally filtered by split file."""
    # Get all clips with joint_q for both persons
    clips_p0 = set()
    clips_p1 = set()
    for f in os.listdir(data_dir):
        if f.endswith("_person0_joint_q.npy"):
            clips_p0.add(f.replace("_person0_joint_q.npy", ""))
        elif f.endswith("_person1_joint_q.npy"):
            clips_p1.add(f.replace("_person1_joint_q.npy", ""))

    all_clips = clips_p0 & clips_p1  # both persons must exist

    if split_file and os.path.exists(split_file):
        with open(split_file) as f:
            split_ids = {line.strip() for line in f if line.strip()}
        all_clips = all_clips & split_ids

    return sorted(all_clips)


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute solo torques + simulate to get positions"
    )
    parser.add_argument("--clip", type=str, default=None,
                        help="Single clip ID")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["interhuman", "interx"],
                        help="Process dataset")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Only process clips in this split")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman")
    parser.add_argument("--fps", type=int, default=30,
                        help="Data playback FPS (default 30 = InterMask eval rate)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample loaded data by this factor "
                             "(2 = 60→30fps to match InterMask)")
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--torque-mode", type=str, default="pd",
                        choices=["pd", "id+pd"],
                        help="Torque mode: 'pd' = full PD tracking only "
                             "(stable, no I.D. needed); 'id+pd' = inverse "
                             "dynamics + PD correction (experimental)")
    parser.add_argument("--pd-scale", type=float, default=1.0,
                        help="PD gain multiplier (1.0 = full PD tracking, "
                             "0.1 = 10%% correction on top of I.D. torques)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results "
                             "(default: date-stamped under data/batch_sim_solo/)")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--resume", action="store_true",
                        help="Skip clips with existing output")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")
    parser.add_argument("--skip-torques", action="store_true",
                        help="Skip clips without pre-computed torques")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Process at most N clips (for testing)")
    parser.add_argument("--plot-forces", action="store_true",
                        help="Generate matplotlib plots of contact forces "
                             "for each clip (saved as PNG)")
    args = parser.parse_args()

    if args.dataset:
        args.data_dir = f"data/retargeted_v2/{args.dataset}"

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.isdir(data_dir):
        if os.path.isdir(args.data_dir):
            data_dir = args.data_dir
        else:
            print(f"ERROR: data directory not found: {data_dir}")
            sys.exit(1)

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
        clips = list_clips(data_dir, split_file)

    if args.max_clips:
        clips = clips[:args.max_clips]

    # ── Output directory (date-stamped) ──────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dataset_name = args.dataset or "custom"
        output_dir = os.path.join(
            PROJECT_ROOT, "data", "batch_sim_solo",
            f"{dataset_name}_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = {
        "script": "batch_sim_solo.py",
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "gpu": args.gpu,
        "fps": args.fps,
        "downsample": args.downsample,
        "gain_scale": args.gain_scale,
        "torque_mode": args.torque_mode,
        "pd_scale": args.pd_scale,
        "n_clips": len(clips),
        "data_dir": data_dir,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Batch Solo Simulation")
    print(f"{'='*60}")
    print(f"  Dataset:    {args.dataset or 'custom'}")
    print(f"  Split:      {args.split or 'all'}")
    print(f"  Clips:      {len(clips)}")
    print(f"  FPS:        {args.fps}")
    print(f"  Downsample: {args.downsample}x (data {args.fps * args.downsample}→{args.fps} fps)")
    print(f"  Torque:     {args.torque_mode} (pd_scale={args.pd_scale})")
    print(f"  GPU:        {args.gpu}")
    print(f"  Input:      {data_dir}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Process ──────────────────────────────────────────
    processed = 0
    skipped = 0
    errors = 0
    missing = 0
    t_start = time.time()

    for i, clip_id in enumerate(clips):
        elapsed = time.time() - t_start
        rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
        eta_h = ((len(clips) - i) / rate) if rate > 0 else 0

        print(f"\n[{i+1}/{len(clips)}] Clip {clip_id} "
              f"(done={processed}, skip={skipped}, err={errors}, "
              f"rate={rate:.0f}/h, ETA={eta_h:.1f}h)")

        try:
            result = simulate_clip(
                clip_id, data_dir,
                fps=args.fps, gain_scale=args.gain_scale,
                device=args.gpu, skip_torques=args.skip_torques,
                force=args.force, downsample=args.downsample,
                output_dir=output_dir,
                torque_mode=args.torque_mode,
                pd_scale=args.pd_scale,
            )
            if result is None:
                missing += 1
                print(f"  SKIP: missing data")
            elif isinstance(result, str) and result == "skip":
                skipped += 1
                print(f"  SKIP: output exists")
            else:
                processed += 1
                print(f"  DONE: saved {result.shape}")
                if args.plot_forces:
                    plot_clip_forces(clip_id, output_dir, fps=args.fps)
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start

    # Save progress
    progress = {
        "done": processed,
        "failed": errors,
        "skipped": skipped,
        "missing": missing,
        "total": len(clips),
        "elapsed_sec": elapsed,
        "updated": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BATCH SIMULATION COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Missing:   {missing}")
    print(f"  Errors:    {errors}")
    print(f"  Time:      {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
