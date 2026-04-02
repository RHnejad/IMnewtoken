"""
visualize_newton_comparison.py — Newton GL viewer for 4-way comparison.

Shows 4 humanoids in the Newton viewer:
  1. GT reference (far left)
  2. GT after PPO (center-left)
  3. Generated reference (center-right)
  4. Generated after PPO (far right)

Usage:
    # Run RL tracker on both GT and generated, then visualize
    python prepare6/visualize_newton_comparison.py --clip-id 161 --run

    # Load from saved npz (skip training)
    python prepare6/visualize_newton_comparison.py --clip-id 161 \
        --gt-npz output/rl_comparison/clip_161/gt_result.npz \
        --gen-npz output/rl_comparison/clip_161/gen_result.npz

    # Adjust playback speed
    python prepare6/visualize_newton_comparison.py --clip-id 161 --run --fps 15
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare4.gen_xml import get_or_create_xml
from prepare5.phc_config import COORDS_PER_PERSON


class FourWayVisualizer:
    """Newton GL viewer showing GT ref | GT sim | Gen ref | Gen sim."""

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.sim_time = 0.0
        self._wall_start = None
        self.viewer = viewer
        self.device = args.device
        self._clip_id = args.clip_id
        self._x_spacing = 2.5  # meters between humanoids

        # Load or compute data
        self._load_data(args)

        # Build 4-humanoid Newton model
        self._build_viewer_model()

        # Set initial frame
        self.frame = 0
        self._set_frame(0)
        self.viewer.set_model(self.model)
        self._setup_camera()

        print(f"\nReady! Playing {self.T} frames ({self.T / self.fps:.1f}s).")
        print(f"  Layout (left → right):")
        print(f"    1. GT Reference (input)")
        print(f"    2. GT after PPO (physics-corrected)")
        print(f"    3. Generated Reference")
        print(f"    4. Generated after PPO (physics-corrected)")

    def _load_data(self, args):
        """Load ref/sim joint_q from npz files or by running RL tracker."""
        from prepare5.run_phc_tracker import load_clip, retarget_person

        # Get betas for model construction
        persons_gt, self._text = load_clip(args.clip_id, "gt")
        self._betas = persons_gt[args.person]["betas"]
        if isinstance(self._betas, np.ndarray) and self._betas.ndim == 2:
            self._betas = self._betas[0]

        if args.run:
            self._run_both(args)
        else:
            self._load_from_npz(args)

    def _run_both(self, args):
        """Run RL tracker on both GT and generated."""
        from prepare5.run_phc_tracker import load_clip, retarget_person
        from prepare6.rl_tracker import RLTracker

        out_dir = os.path.join(PROJECT_ROOT, "output", "rl_comparison",
                                f"clip_{args.clip_id}")
        os.makedirs(out_dir, exist_ok=True)

        tracker = RLTracker(
            device=args.device, n_envs=args.n_envs,
            total_timesteps=args.total_timesteps, verbose=True,
        )

        # GT
        print(f"\n--- Training RL tracker on GT ---")
        persons_gt, _ = load_clip(args.clip_id, "gt")
        gt_jq, gt_betas = retarget_person(persons_gt[args.person], "gt",
                                            device=args.device)
        gt_result = tracker.train_and_evaluate(gt_jq, gt_betas)
        self._gt_ref_jq = gt_jq.astype(np.float32)
        self._gt_sim_jq = gt_result['sim_joint_q']
        self._gt_mpjpe = gt_result['mpjpe_mm']

        np.savez(os.path.join(out_dir, "gt_result.npz"),
                 sim_positions=gt_result['sim_positions'],
                 ref_positions=gt_result['ref_positions'],
                 sim_joint_q=gt_result['sim_joint_q'],
                 ref_joint_q=gt_jq,
                 per_frame_mpjpe_mm=gt_result['per_frame_mpjpe_mm'])

        # Generated
        print(f"\n--- Training RL tracker on Generated ---")
        persons_gen, _ = load_clip(args.clip_id, "generated")
        gen_jq, gen_betas = retarget_person(persons_gen[args.person], "generated",
                                             device=args.device)
        gen_result = tracker.train_and_evaluate(gen_jq, gen_betas)
        self._gen_ref_jq = gen_jq.astype(np.float32)
        self._gen_sim_jq = gen_result['sim_joint_q']
        self._gen_mpjpe = gen_result['mpjpe_mm']

        np.savez(os.path.join(out_dir, "gen_result.npz"),
                 sim_positions=gen_result['sim_positions'],
                 ref_positions=gen_result['ref_positions'],
                 sim_joint_q=gen_result['sim_joint_q'],
                 ref_joint_q=gen_jq,
                 per_frame_mpjpe_mm=gen_result['per_frame_mpjpe_mm'])

        self.T = min(self._gt_ref_jq.shape[0], self._gen_ref_jq.shape[0])

    def _load_from_npz(self, args):
        """Load from saved npz files."""
        from prepare5.run_phc_tracker import load_clip, retarget_person

        gt_path  = args.gt_npz
        gen_path = args.gen_npz

        # Auto-detect paths if not provided
        if not gt_path:
            gt_path = os.path.join(PROJECT_ROOT, "output", "rl_comparison",
                                    f"clip_{args.clip_id}", "gt_result.npz")
        if not gen_path:
            gen_path = os.path.join(PROJECT_ROOT, "output", "rl_comparison",
                                     f"clip_{args.clip_id}", "gen_result.npz")

        gt_data  = np.load(gt_path)
        gen_data = np.load(gen_path)

        # sim_joint_q may or may not exist in older npz files
        if 'sim_joint_q' in gt_data and 'ref_joint_q' in gt_data:
            self._gt_ref_jq = gt_data['ref_joint_q']
            self._gt_sim_jq = gt_data['sim_joint_q']
        else:
            # Fallback: re-run retarget for ref, and we can't get sim_jq
            # without re-running tracker. Raise helpful error.
            raise ValueError(
                f"npz at {gt_path} missing sim_joint_q/ref_joint_q. "
                f"Re-run with --run to generate them."
            )

        if 'sim_joint_q' in gen_data and 'ref_joint_q' in gen_data:
            self._gen_ref_jq = gen_data['ref_joint_q']
            self._gen_sim_jq = gen_data['sim_joint_q']
        else:
            raise ValueError(
                f"npz at {gen_path} missing sim_joint_q/ref_joint_q. "
                f"Re-run with --run to generate them."
            )

        # Compute MPJPE from positions
        self._gt_mpjpe = np.linalg.norm(
            gt_data['sim_positions'] - gt_data['ref_positions'], axis=-1
        ).mean() * 1000
        self._gen_mpjpe = np.linalg.norm(
            gen_data['sim_positions'] - gen_data['ref_positions'], axis=-1
        ).mean() * 1000

        self.T = min(self._gt_ref_jq.shape[0], self._gen_ref_jq.shape[0])

    def _build_viewer_model(self):
        """Build Newton model with 4 humanoids for visualization."""
        xml_path = get_or_create_xml(self._betas, foot_geom="sphere")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for _ in range(4):
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        self.state = self.model.state()
        self.jqd = wp.zeros(
            self.model.joint_dof_count, dtype=wp.float32, device=self.device
        )

        print(f"Viewer model: 4 humanoids ({self.model.joint_coord_count} coords)")

    def _set_frame(self, t):
        """Set all 4 humanoids to frame t with X offsets."""
        combined_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        C = COORDS_PER_PERSON  # 76

        # Order: GT ref, GT sim, Gen ref, Gen sim
        jq_list = [
            self._gt_ref_jq,
            self._gt_sim_jq,
            self._gen_ref_jq,
            self._gen_sim_jq,
        ]

        for i, jq_arr in enumerate(jq_list):
            frame = min(t, jq_arr.shape[0] - 1)
            jq = jq_arr[frame].astype(np.float32).copy()
            jq[0] += i * self._x_spacing   # X offset
            combined_q[i * C:(i + 1) * C] = jq

        self.state.joint_q = wp.array(
            combined_q, dtype=wp.float32, device=self.device
        )
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def _setup_camera(self):
        """Position camera to see all 4 humanoids."""
        center_x = 1.5 * self._x_spacing   # center of 4 humanoids
        cam_pos = wp.vec3(center_x, -8.0, 2.5)
        yaw = 90.0
        pitch = -10.0
        self.viewer.set_camera(cam_pos, pitch, yaw)

    def step(self):
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        self.sim_time = now - self._wall_start
        self.frame = int(self.sim_time * self.fps) % self.T
        self._set_frame(self.frame)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def gui(self, imgui):
        """Side panel with clip info and metrics."""
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
            "[ RL TRACKING — 4-WAY COMPARISON ]"
        )
        imgui.separator()
        imgui.text(f"Clip:    {self._clip_id}")
        imgui.text(f"FPS:     {self.fps}")
        imgui.text(f"Frames:  {self.T}")

        pct = int(100 * self.frame / max(self.T - 1, 1))
        imgui.text(f"Frame:   {self.frame} / {self.T - 1}  ({pct}%)")

        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Tracking Metrics:"
        )
        imgui.text(f"GT MPJPE:  {self._gt_mpjpe:.1f} mm")
        imgui.text(f"Gen MPJPE: {self._gen_mpjpe:.1f} mm")
        gap = self._gen_mpjpe - self._gt_mpjpe
        color = imgui.ImVec4(0.4, 1.0, 0.4, 1.0) if gap > 0 else imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
        imgui.text_colored(color, f"Gap:       {gap:+.1f} mm")

        imgui.separator()
        imgui.text("Layout (left to right):")
        imgui.text_colored(imgui.ImVec4(0.4, 0.6, 1.0, 1.0),  "  1. GT Reference")
        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0),  "  2. GT after PPO")
        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0),  "  3. Gen Reference")
        imgui.text_colored(imgui.ImVec4(1.0, 0.7, 0.2, 1.0),  "  4. Gen after PPO")

        if hasattr(self, '_text'):
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "Description:"
            )
            imgui.text_wrapped(self._text)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip-id", type=int, default=161)
    parser.add_argument("--run", action="store_true",
                        help="Run RL tracker on both GT and generated first")
    parser.add_argument("--gt-npz", type=str, default=None,
                        help="Path to GT result .npz (with sim_joint_q)")
    parser.add_argument("--gen-npz", type=str, default=None,
                        help="Path to generated result .npz")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--person", type=int, default=0, choices=[0, 1])
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200000)

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"

    example = FourWayVisualizer(viewer, args)
    newton.examples.run(example, args)
