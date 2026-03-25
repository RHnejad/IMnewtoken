"""
Visualize extracted/retargeted joint positions on the SMPL model in Newton.

Loads (T, 22, 3) position data, runs quick IK to recover joint angles,
then plays back the motion in Newton's OpenGL viewer.

Usage:
    # View extracted positions (default)
    python prepare/visualize_newton.py --clip 1000 --person 0

    # View retargeted positions
    python prepare/visualize_newton.py --clip 1000 --person 0 \
        --data-dir data/retargeted/test

    # Both persons side-by-side (loads two SMPL models)
    python prepare/visualize_newton.py --clip 1000

    # Headless (no viewer window, just IK sanity check)
    python prepare/visualize_newton.py --clip 1000 --headless

    # Adjust playback speed
    python prepare/visualize_newton.py --clip 1000 --fps 30
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np

import warp as wp

wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples
import newton.ik as ik

# ═══════════════════════════════════════════════════════════════
# SMPL joint → Newton body index
# ═══════════════════════════════════════════════════════════════
SMPL_TO_NEWTON = {
    0: 0, 1: 1, 2: 5, 3: 9, 4: 2, 5: 6, 6: 10, 7: 3,
    8: 7, 9: 11, 10: 4, 11: 8, 12: 12, 13: 14, 14: 19,
    15: 13, 16: 15, 17: 20, 18: 16, 19: 21, 20: 17, 21: 22,
}
N_SMPL = 22
PERSON_OFFSETS = [
    wp.vec3(0.0, 0.0, 0.0),    # person 0: no offset
    wp.vec3(0.0, 2.0, 0.0),    # person 1: offset in Y (so they don't overlap)
]


def solve_joint_q(model, ref_pos, ik_iters=30, device="cuda:0"):
    """
    Given (T, 22, 3) positions, recover joint_q (T, 76) via batched IK.
    Since positions already come from FK, this converges very fast.
    """
    T = ref_pos.shape[0]
    n_coords = model.joint_coord_count  # 76

    # Build IK objectives
    objectives = []
    for j in range(N_SMPL):
        targets = [
            wp.vec3(float(ref_pos[t, j, 0]),
                    float(ref_pos[t, j, 1]),
                    float(ref_pos[t, j, 2]))
            for t in range(T)
        ]
        obj = ik.IKObjectivePosition(
            link_index=SMPL_TO_NEWTON[j],
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

    # Initialize: pelvis from ref, rest zero
    jq_init = np.zeros((T, n_coords), dtype=np.float32)
    for t in range(T):
        jq_init[t, 0:3] = ref_pos[t, 0]
        jq_init[t, 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quat (xyzw)
    jq = wp.array(jq_init, dtype=wp.float32, device=device)

    solver.step(jq, jq, iterations=ik_iters)
    wp.synchronize()

    jq_np = jq.numpy()  # (T, 76)

    # Quick error check
    state = model.state()
    jqd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=device)
    errs = []
    for t in range(T):
        state.joint_q = wp.array(jq_np[t], dtype=wp.float32, device=device)
        newton.eval_fk(model, state.joint_q, jqd, state)
        bq = state.body_q.numpy()
        fk_pos = np.array([bq[SMPL_TO_NEWTON[j], :3] for j in range(N_SMPL)])
        errs.append(np.sqrt(((fk_pos - ref_pos[t]) ** 2).sum(-1)).mean())
    mean_err = np.mean(errs) * 100
    print(f"  IK quality: MPJPE = {mean_err:.2f} cm")

    return jq_np


class MotionVisualizer:
    """Newton viewer that plays back retargeted motion on SMPL model(s)."""

    def __init__(self, viewer, args=None):
        self.fps = args.fps if args else 20
        self.sim_time = 0.0
        self._wall_start = None
        self.viewer = viewer
        self.device = args.device if args else "cuda:0"
        self.loop = True

        data_dir = args.data_dir if args else "data/extracted_positions/interhuman"
        clip_id = args.clip if args else "1000"

        # ── Load position data ───────────────────────────────
        self.persons_pos = []
        self.person_labels = []
        if args and args.person is not None:
            personas = [args.person]
        else:
            personas = [0, 1]

        # ── Try loading pre-computed joint_q first, fall back to positions ──
        self.all_joint_q = []  # list of (T, 76) per person
        need_ik = False

        for p_idx in personas:
            # Try joint_q file first (from debug_pipeline.py)
            jq_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
            pos_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}.npy")

            if os.path.exists(jq_path):
                jq = np.load(jq_path)
                self.all_joint_q.append(jq)
                self.person_labels.append(f"person{p_idx}")
                print(f"Loaded joint_q {jq_path}: {jq.shape}")
            elif os.path.exists(pos_path):
                pos = np.load(pos_path)
                self.persons_pos.append(pos)
                self.person_labels.append(f"person{p_idx}")
                need_ik = True
                print(f"Loaded positions {pos_path}: {pos.shape}")
            else:
                print(f"File not found: {pos_path}")

        if not self.person_labels:
            raise FileNotFoundError(f"No data found for clip {clip_id} in {data_dir}")

        # ── Build model with N persons ───────────────────────
        self.n_persons = len(self.person_labels)
        print(f"Clip {clip_id}: {self.n_persons} person(s), FPS={self.fps}")

        print("Building Newton model...")
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for i in range(self.n_persons):
            builder.add_mjcf("prepare/assets/smpl.xml", enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        self.state = self.model.state()
        self.jqd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32,
                            device=self.device)

        # ── Solve IK if we loaded positions instead of joint_q ──
        if need_ik and self.persons_pos:
            print("Solving IK for joint angles...")
            single_builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
            single_builder.add_mjcf("prepare/assets/smpl.xml", enable_self_collisions=False)
            single_model = single_builder.finalize(device=self.device)

            for i, pos in enumerate(self.persons_pos):
                print(f"  IK for {self.person_labels[len(self.all_joint_q)]} ({pos.shape[0]} frames)...")
                jq = solve_joint_q(single_model, pos,
                                   ik_iters=args.ik_iters if args else 30,
                                   device=self.device)
                self.all_joint_q.append(jq)

        # ── Determine frame count ───────────────────────────
        self.T = min(jq.shape[0] for jq in self.all_joint_q)

        # ── Combine joint_q for multi-person model ───────────
        # Multi-person model has n_persons * 76 joint coords
        self.combined_n_coords = self.model.joint_coord_count
        n_per = self.all_joint_q[0].shape[1]
        print(f"Combined model: {self.combined_n_coords} joint coords "
              f"({self.n_persons} x {n_per})")

        # Set initial frame
        self._set_frame(0)
        self.frame = 0

        # Set viewer model
        self.viewer.set_model(self.model)
        print(f"\nReady! Playing {self.T} frames. Close viewer to exit.")

    def _set_frame(self, t):
        """Set model state to frame t."""
        combined_q = np.zeros(self.combined_n_coords, dtype=np.float32)
        n_per = self.all_joint_q[0].shape[1]  # 76

        for i, jq in enumerate(self.all_joint_q):
            frame = min(t, jq.shape[0] - 1)
            combined_q[i * n_per:(i + 1) * n_per] = jq[frame]

        self.state.joint_q = wp.array(combined_q, dtype=wp.float32,
                                      device=self.device)
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def step(self):
        """Advance to next frame."""
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        self.sim_time = now - self._wall_start
        self.frame = int(self.sim_time * self.fps) % self.T
        self._set_frame(self.frame)

    def render(self):
        """Render current frame in viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, default="1000",
                        help="Clip ID (e.g. 1000)")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Person index (omit for both)")
    parser.add_argument("--data-dir", type=str,
                        default="data/extracted_positions/interhuman",
                        help="Directory with .npy files")
    parser.add_argument("--fps", type=int, default=20,
                        help="Playback FPS")
    parser.add_argument("--ik-iters", type=int, default=30,
                        help="IK iterations (30 is enough for retargeted data)")
    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"
    example = MotionVisualizer(viewer, args)
    newton.examples.run(example, args)
