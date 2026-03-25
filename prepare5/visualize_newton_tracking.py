"""
Visualize PHC tracker results in Newton's GL viewer.

Shows reference (input) and simulated (physics-corrected) humanoids
side by side, making it clear that the simulation ran through Newton.

The simulated humanoid is placed with an X offset so both are visible.
Reference = left, Simulated = right.

Usage:
    # Visualize from saved tracker output
    python prepare5/visualize_newton_tracking.py \
        --result output/phc_tracker/clip_1129_gt/phc_result.npz \
        --clip-id 1129 --source gt

    # Run tracker then visualize immediately
    python prepare5/visualize_newton_tracking.py \
        --clip-id 1129 --source gt --run

    # Visualize paired result
    python prepare5/visualize_newton_tracking.py \
        --result output/phc_tracker/clip_1129_gt/phc_paired_result.npz \
        --clip-id 1129 --source gt --paired

    # Adjust playback
    python prepare5/visualize_newton_tracking.py \
        --clip-id 1129 --source gt --run --fps 20

    # Show only the simulated (physics) humanoid (no reference)
    python prepare5/visualize_newton_tracking.py \
        --clip-id 1129 --source gt --run --sim-only
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
from prepare5.phc_config import (
    COORDS_PER_PERSON, SMPL_TO_NEWTON, N_SMPL_JOINTS,
)


class TrackingVisualizer:
    """Newton GL viewer showing reference vs simulated humanoid."""

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.sim_time = 0.0
        self._wall_start = None
        self.viewer = viewer
        self.device = args.device
        self._clip_id = getattr(args, 'clip_id', '?')
        self._source = getattr(args, 'source', '?')
        self._paired = getattr(args, 'paired', False)
        self._sim_only = getattr(args, 'sim_only', False)
        self._foot_geom = getattr(args, 'foot_geom', 'sphere')
        self._paused = False
        self._manual_frame = None

        # ── Load or run tracking ─────────────────────────────
        self._load_data(args)

        # ── Build Newton model(s) ────────────────────────────
        self._build_viewer_model()

        # Set initial frame
        self._set_frame(0)
        self.frame = 0

        # Set viewer model
        self.viewer.set_model(self.model)

        # Camera
        self._setup_camera()

        print(f"\nReady! Playing {self.T} frames ({self.T / self.fps:.1f}s). "
              f"Close viewer to exit.")
        if not self._sim_only:
            print(f"  Left = Reference (input motion)")
            print(f"  Right = Simulated (physics-corrected)")

    def _load_data(self, args):
        """Load ref/sim joint_q from saved .npz or by running the tracker."""
        if args.run:
            self._run_tracker(args)
        else:
            self._load_from_npz(args)

    def _run_tracker(self, args):
        """Run the PHC tracker and store results."""
        from prepare5.run_phc_tracker import load_clip, retarget_person
        from prepare5.phc_tracker import PHCTracker

        clip_id = args.clip_id
        source = args.source
        device = args.device

        print(f"Running PHC tracker on clip {clip_id} ({source})...")
        persons, text = load_clip(clip_id, source)
        self._text = text

        joint_q_A, betas_A = retarget_person(persons[0], source, device=device)
        self._betas_list = [betas_A]
        self._ref_jq_list = [joint_q_A]

        tracker = PHCTracker(
            device=device,
            gain_scale=getattr(args, 'gain_scale', 1.0),
            gain_preset=getattr(args, 'gain_preset', 'phc'),
            foot_geom=getattr(args, 'foot_geom', 'sphere'),
            root_mode=getattr(args, 'root_mode', 'free'),
        )

        if self._paired:
            joint_q_B, betas_B = retarget_person(persons[1], source, device=device)
            self._betas_list.append(betas_B)
            self._ref_jq_list.append(joint_q_B)

            result = tracker.track_paired(joint_q_A, joint_q_B, betas_A, betas_B)
            T = min(joint_q_A.shape[0], joint_q_B.shape[0])

            # Simulated joint_q for both persons
            self._sim_jq_list = [
                result['sim_joint_q'][:, :COORDS_PER_PERSON],
                result['sim_joint_q'][:, COORDS_PER_PERSON:2 * COORDS_PER_PERSON],
            ]
            self._ref_jq_list = [joint_q_A[:T], joint_q_B[:T]]
            self._mpjpe_str = (
                f"A={result['mpjpe_A_mm']:.0f}mm  B={result['mpjpe_B_mm']:.0f}mm"
            )
        else:
            result = tracker.track(joint_q_A, betas_A)
            self._sim_jq_list = [result['sim_joint_q']]
            self._mpjpe_str = f"{result['mpjpe_mm']:.0f}mm"

        self.T = self._sim_jq_list[0].shape[0]
        self._per_frame_mpjpe = result.get('per_frame_mpjpe_mm',
                                           result.get('per_frame_mpjpe_A_mm'))

    def _load_from_npz(self, args):
        """Load from a saved .npz result file."""
        result_path = args.result
        if not result_path:
            # Infer default path
            result_path = os.path.join(
                "output", "phc_tracker",
                f"clip_{args.clip_id}_{args.source}",
                "phc_paired_result.npz" if self._paired else "phc_result.npz",
            )

        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"Result file not found: {result_path}\n"
                f"Run the tracker first, or use --run to run+visualize."
            )

        print(f"Loading results from {result_path}")
        data = np.load(result_path)

        # We need betas to build the XML/model. Load from retargeted data.
        clip_id = args.clip_id
        source = args.source

        from prepare5.run_phc_tracker import load_clip, retarget_person
        persons, text = load_clip(clip_id, source)
        self._text = text

        betas_A = persons[0]["betas"]
        if isinstance(betas_A, np.ndarray) and betas_A.ndim == 2:
            betas_A = betas_A[0]
        self._betas_list = [betas_A]

        if self._paired:
            betas_B = persons[1]["betas"]
            if isinstance(betas_B, np.ndarray) and betas_B.ndim == 2:
                betas_B = betas_B[0]
            self._betas_list.append(betas_B)

        # Load sim_joint_q from npz
        if 'sim_joint_q' in data:
            sim_jq = data['sim_joint_q']
            if self._paired:
                self._sim_jq_list = [
                    sim_jq[:, :COORDS_PER_PERSON],
                    sim_jq[:, COORDS_PER_PERSON:2 * COORDS_PER_PERSON],
                ]
            else:
                self._sim_jq_list = [sim_jq]
        else:
            raise ValueError(
                f"No sim_joint_q in {result_path}. "
                f"Re-run the tracker to save joint_q data."
            )

        # Load ref joint_q — from npz if available, else retarget
        if 'ref_joint_q' in data:
            self._ref_jq_list = [data['ref_joint_q']]
        else:
            joint_q_A, _ = retarget_person(persons[0], source, device=args.device)
            self._ref_jq_list = [joint_q_A]
        if self._paired and 'ref_joint_q' not in data:
            joint_q_B, _ = retarget_person(persons[1], source, device=args.device)
            self._ref_jq_list.append(joint_q_B)

        self.T = self._sim_jq_list[0].shape[0]

        # Truncate ref to match sim length
        for i in range(len(self._ref_jq_list)):
            self._ref_jq_list[i] = self._ref_jq_list[i][:self.T]

        # Compute MPJPE string from npz data
        if 'sim_positions' in data and 'ref_positions' in data:
            err = np.linalg.norm(
                data['sim_positions'] - data['ref_positions'], axis=-1
            ).mean() * 1000
            self._mpjpe_str = f"{err:.0f}mm"
        else:
            self._mpjpe_str = "N/A"
        self._per_frame_mpjpe = None

    def _build_viewer_model(self):
        """Build a Newton model with humanoids for visualization.

        Layout:
          - If sim_only: just the simulated humanoid(s)
          - Otherwise: reference humanoid(s) on the left, simulated on the right
            separated by X_OFFSET
        """
        self._x_offset = 2.0  # meters between ref and sim humanoids

        xml_paths = []
        person_labels = []
        n_persons = len(self._betas_list)

        if not self._sim_only:
            # Reference humanoid(s) — same body shape
            for i, betas in enumerate(self._betas_list):
                xml_path = get_or_create_xml(betas, foot_geom=self._foot_geom)
                xml_paths.append(xml_path)
                person_labels.append(f"ref_{i}")

        # Simulated humanoid(s)
        for i, betas in enumerate(self._betas_list):
            xml_path = get_or_create_xml(betas, foot_geom=self._foot_geom)
            xml_paths.append(xml_path)
            person_labels.append(f"sim_{i}")

        # Build combined model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for xml_path in xml_paths:
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        self.state = self.model.state()
        self.jqd = wp.zeros(
            self.model.joint_dof_count, dtype=wp.float32, device=self.device
        )

        self._n_humanoids = len(xml_paths)
        self._n_per = COORDS_PER_PERSON  # 76 coords per humanoid
        self._person_labels = person_labels

        print(f"Viewer model: {self._n_humanoids} humanoids "
              f"({self.model.joint_coord_count} joint coords)")

    def _set_frame(self, t):
        """Set all humanoids to frame t."""
        combined_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        n_persons = len(self._betas_list)
        idx = 0

        if not self._sim_only:
            # Reference humanoids
            for i in range(n_persons):
                frame = min(t, self._ref_jq_list[i].shape[0] - 1)
                jq = self._ref_jq_list[i][frame].astype(np.float32)
                combined_q[idx * self._n_per:(idx + 1) * self._n_per] = jq
                idx += 1

        # Simulated humanoids — offset in X
        for i in range(n_persons):
            frame = min(t, self._sim_jq_list[i].shape[0] - 1)
            jq = self._sim_jq_list[i][frame].astype(np.float32).copy()
            if not self._sim_only:
                jq[0] += self._x_offset  # shift X for side-by-side
            combined_q[idx * self._n_per:(idx + 1) * self._n_per] = jq
            idx += 1

        self.state.joint_q = wp.array(
            combined_q, dtype=wp.float32, device=self.device
        )
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def _setup_camera(self):
        """Position camera to see both ref and sim humanoids."""
        # Center between ref and sim
        ref_start = self._ref_jq_list[0][0, :3]
        if self._sim_only:
            center = ref_start.copy()
        else:
            sim_start = self._sim_jq_list[0][0, :3].copy()
            sim_start[0] += self._x_offset
            center = (ref_start + sim_start) / 2.0

        cam_dist = 6.0
        cam_pos = wp.vec3(
            float(center[0]),
            float(center[1]) - cam_dist,
            2.0,
        )
        yaw = 90.0
        pitch = -15.0
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
        """Side-panel info in the Newton GL viewer."""
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
            "[ PHC TRACKING PLAYBACK ]"
        )
        imgui.separator()
        imgui.text(f"Clip:    {self._clip_id}")
        imgui.text(f"Source:  {self._source}")
        imgui.text(f"FPS:     {self.fps}")
        imgui.text(f"Frames:  {self.T}")
        imgui.separator()

        pct = int(100 * self.frame / max(self.T - 1, 1))
        imgui.text(f"Frame:   {self.frame} / {self.T - 1}  ({pct}%)")

        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Tracking Metrics:"
        )
        imgui.text(f"MPJPE: {self._mpjpe_str}")

        # Per-frame MPJPE for current frame
        if self._per_frame_mpjpe is not None and self.frame < len(self._per_frame_mpjpe):
            imgui.text(f"Frame MPJPE: {self._per_frame_mpjpe[self.frame]:.0f}mm")

        imgui.separator()
        if not self._sim_only:
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 1.0, 1.0), "Left:  Reference (input)"
            )
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.6, 0.6, 1.0), "Right: Simulated (physics)"
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.6, 0.6, 1.0), "Showing: Simulated (physics)"
            )

        if hasattr(self, '_text'):
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "Description:"
            )
            imgui.text_wrapped(self._text)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip-id", type=int, default=1129,
                        help="InterHuman clip ID")
    parser.add_argument("--source", choices=["gt", "generated"], default="gt",
                        help="Data source")
    parser.add_argument("--result", type=str, default=None,
                        help="Path to .npz result file (skip --run)")
    parser.add_argument("--run", action="store_true",
                        help="Run the tracker first, then visualize")
    parser.add_argument("--paired", action="store_true",
                        help="Paired (two-person) mode")
    parser.add_argument("--fps", type=int, default=20,
                        help="Playback FPS")
    parser.add_argument("--sim-only", action="store_true",
                        help="Show only the simulated humanoid (no reference)")
    parser.add_argument("--gain-scale", type=float, default=1.0,
                        help="PD gain multiplier (used with --run)")
    parser.add_argument("--gain-preset", choices=["phc", "old"], default="phc",
                        help="PD gain preset (used with --run)")
    parser.add_argument("--foot-geom", choices=["box", "sphere", "capsule"],
                        default="sphere",
                        help="Foot collision geometry (used with --run)")
    parser.add_argument("--root-mode", choices=["free", "orient", "skyhook"],
                        default="free",
                        help="Root force mode (used with --run): "
                             "free (no root forces), orient (orientation only), "
                             "skyhook (full position+orientation PD)")

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"

    example = TrackingVisualizer(viewer, args)
    newton.examples.run(example, args)
