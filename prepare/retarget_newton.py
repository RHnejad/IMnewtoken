"""
retarget_newton.py — Batch GPU motion retargeting via Newton IK.

For each clip: solve ALL frames in one batched GPU IK call (n_problems=T),
then FK to extract body positions with consistent bone lengths.

Verified: MPJPE ~1.7 cm on InterHuman clips.

Config:
  - 22 position objectives, no joint limits → _lm_solve_tiled_75_66 (fits RTX 3090)
  - AUTODIFF Jacobian mode
  - n_problems=T (all frames solved in parallel on GPU)

Usage:
    python prepare/retarget_newton.py \
        --input_dir data/extracted_positions/interhuman \
        --output_dir data/retargeted/test \
        --clip 1000_person0

    python prepare/retarget_newton.py \
        --input_dir data/extracted_positions/interhuman \
        --output_dir data/retargeted/interhuman
"""
import os
import time
import argparse
import warnings
import numpy as np
import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.ik as ik

# ═══════════════════════════════════════════════════════════════
# SMPL joint → Newton body index (verified via discover_ik.py)
# ═══════════════════════════════════════════════════════════════
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}
N_SMPL_JOINTS = 22


def build_model(device="cuda:0"):
    """Build Newton model for IK + FK."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf("prepare/assets/smpl.xml", enable_self_collisions=False)
    return builder.finalize(device=device)


def build_ik_solver(model, T, ref_pos, device="cuda:0"):
    """
    Build batched IK solver for T frames.

    Args:
        model: Newton model
        T: number of frames (= n_problems)
        ref_pos: (T, 22, 3) reference positions (used to set initial targets)
        device: CUDA device

    Returns:
        solver, objectives list
    """
    objectives = []
    for smpl_j in range(N_SMPL_JOINTS):
        # Build target array for all T frames at once
        targets = []
        for t in range(T):
            p = ref_pos[t, smpl_j]
            targets.append(wp.vec3(float(p[0]), float(p[1]), float(p[2])))

        obj = ik.IKObjectivePosition(
            link_index=SMPL_TO_NEWTON[smpl_j],
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array(targets, dtype=wp.vec3, device=device),
            weight=1.0,
        )
        objectives.append(obj)

    solver = ik.IKSolver(
        model=model,
        n_problems=T,
        objectives=objectives,
        lambda_initial=0.01,
        jacobian_mode=ik.IKJacobianType.AUTODIFF,
    )
    return solver, objectives


def extract_positions(body_q_np):
    """Extract (22, 3) from body_q numpy (24, 7)."""
    pos = np.zeros((N_SMPL_JOINTS, 3), dtype=np.float32)
    for j in range(N_SMPL_JOINTS):
        pos[j] = body_q_np[SMPL_TO_NEWTON[j], :3]
    return pos


def retarget_clip(model, ref_pos, ik_iters=50, device="cuda:0"):
    """
    Retarget one clip: (T, 22, 3) → (T, 22, 3) with consistent bone lengths.

    1. Build batched IK solver (n_problems=T)
    2. Solve all frames in one GPU call
    3. FK per frame to extract body positions
    """
    T = ref_pos.shape[0]
    n_coords = model.joint_coord_count  # 76

    # ── Build solver for this clip ───────────────────────────
    solver, objectives = build_ik_solver(model, T, ref_pos, device)

    # ── Initialize joint angles: pelvis from reference, rest zero ──
    jq_init = np.zeros((T, n_coords), dtype=np.float32)
    for t in range(T):
        jq_init[t, 0:3] = ref_pos[t, 0]       # pelvis xyz
        jq_init[t, 3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quat
    jq = wp.array(jq_init, dtype=wp.float32, device=device)

    # ── Solve all frames at once ─────────────────────────────
    solver.step(jq, jq, iterations=ik_iters)
    wp.synchronize()

    # ── FK: joint angles → body positions per frame ──────────
    jq_np = jq.numpy()  # (T, 76)
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)

    all_positions = np.zeros((T, N_SMPL_JOINTS, 3), dtype=np.float32)
    for t in range(T):
        state.joint_q = wp.array(jq_np[t], dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        all_positions[t] = extract_positions(state.body_q.numpy())

    # ── Report error ─────────────────────────────────────────
    per_frame_err = np.sqrt(((all_positions - ref_pos) ** 2).sum(-1).mean(-1))
    mean_err = per_frame_err.mean()
    max_err = per_frame_err.max()
    print(f"    MPJPE: mean={mean_err*100:.2f}cm, max={max_err*100:.2f}cm")

    return all_positions


def main():
    parser = argparse.ArgumentParser(
        description="Batch GPU motion retargeting: Newton IK + FK"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clip", default=None, help="Single clip (no .npy)")
    parser.add_argument("--ik_iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building model on {args.device}...")
    model = build_model(args.device)

    # Collect clips
    if args.clip:
        files = [f"{args.clip}.npy"]
    else:
        files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.npy'))

    print(f"Retargeting {len(files)} clips (ik_iters={args.ik_iters})\n")

    t_start = time.time()
    done, skipped = 0, 0

    for i, fname in enumerate(files):
        out_path = os.path.join(args.output_dir, fname)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        ref = np.load(os.path.join(args.input_dir, fname))  # (T, 22, 3)
        print(f"  [{i+1}/{len(files)}] {fname}: {ref.shape[0]} frames", end="  ")

        t0 = time.time()
        result = retarget_clip(model, ref, args.ik_iters, args.device)
        clip_time = time.time() - t0

        np.save(out_path, result)
        done += 1
        print(f"({clip_time:.1f}s)")

        # Progress every 200 clips
        if done % 200 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed
            remaining = (len(files) - skipped - done) / rate if rate > 0 else 0
            print(f"  --- {done}/{len(files)-skipped} done, "
                  f"{rate:.1f} clips/s, ~{remaining/60:.0f} min remaining ---")

    elapsed = time.time() - t_start
    print(f"\nDone. {done} processed, {skipped} skipped, {elapsed:.1f}s total.")
    if done > 0:
        print(f"Average: {elapsed/done:.2f}s per clip")
    print(f"Output in {args.output_dir}")


if __name__ == "__main__":
    main()