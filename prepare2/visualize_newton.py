"""
Visualize retargeted motion using per-subject skeleton in Newton viewer.

Loads joint_q (exact rotations) and per-subject betas to reconstruct
the correct skeleton — no IK needed, no bone-length mismatch.

Usage:
    # Single person, full motion
    python prepare2/visualize_newton.py --clip 1000 --person 0

    # Both persons side-by-side
    python prepare2/visualize_newton.py --clip 1000

    # REST POSE visualization (all joint angles = 0)
    python prepare2/visualize_newton.py --clip 1000 --rest

    # Custom data directory
    python prepare2/visualize_newton.py --clip 1000 \
        --data-dir data/retargeted_v2/interhuman

    # Adjust playback speed
    python prepare2/visualize_newton.py --clip 1000 --fps 30
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

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Annotations directory per dataset name (mirrors utils/get_opt.py mapping)
_DATASET_ANNOTS = {
    "interhuman": os.path.join(PROJECT_ROOT, "data", "InterHuman", "annots"),
    "interx":     os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset", "texts_processed"),
}


def _infer_annots_dir(data_dir: str) -> str | None:
    """Return the annotation directory matching the dataset embedded in data_dir."""
    dataset_name = os.path.basename(data_dir.rstrip("/\\")).lower()
    return _DATASET_ANNOTS.get(dataset_name)

from prepare2.gen_smpl_xml import generate_smpl_xml
from prepare2.retarget import get_or_create_xml


class PerSubjectVisualizer:
    """Newton viewer that plays back retargeted motion with per-subject skeletons."""

    def __init__(self, viewer, args=None):
        self.fps = args.fps if args else 20
        self.sim_time = 0.0
        self._wall_start = None
        self.viewer = viewer
        self.device = args.device if args else "cuda:0"
        self.rest_mode = args.rest if args and hasattr(args, 'rest') else False

        data_dir = args.data_dir if args else "data/retargeted_v2/interhuman"
        clip_id = args.clip if args else "1000"
        self._clip_id = clip_id

        # ── Determine persons to load ────────────────────────
        if args and args.person is not None:
            personas = [args.person]
        else:
            personas = [0, 1]

        # ── Load motion text descriptions ─────────────────────
        self.motion_texts = []
        annots_dir = (
            (args.annots_dir if args and hasattr(args, 'annots_dir') else None)
            or _infer_annots_dir(data_dir)
        )
        if annots_dir:
            annots_path = os.path.join(annots_dir, f"{clip_id}.txt")
            if os.path.exists(annots_path):
                with open(annots_path, "r") as f:
                    self.motion_texts = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(self.motion_texts)} motion description(s)")
            else:
                print(f"No annotation found: {annots_path}")

        # ── Load joint_q and betas for each person ───────────
        self.all_joint_q = []
        self.all_betas = []
        self.all_xml_paths = []
        self.person_labels = []

        for p_idx in personas:
            jq_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
            betas_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_betas.npy")

            if not os.path.exists(jq_path):
                print(f"joint_q not found: {jq_path}")
                continue
            if not os.path.exists(betas_path):
                print(f"betas not found: {betas_path}")
                continue

            jq = np.load(jq_path)
            betas = np.load(betas_path)
            xml_path = get_or_create_xml(betas)

            self.all_joint_q.append(jq)
            self.all_betas.append(betas)
            self.all_xml_paths.append(xml_path)
            self.person_labels.append(f"person{p_idx}")
            print(f"Loaded {self.person_labels[-1]}: {jq.shape[0]} frames, "
                  f"XML={os.path.basename(xml_path)}")

        if not self.person_labels:
            raise FileNotFoundError(
                f"No joint_q/betas files found for clip {clip_id} in {data_dir}\n"
                f"Run: python prepare2/retarget.py --dataset interhuman --clip {clip_id}"
            )

        # ── Build multi-person Newton model ──────────────────
        # With up_axis=Z the Newton world frame matches InterHuman's Z-up;
        # retarget.py maps positions directly (no coordinate rotation).\n        # joint_q root Z ≈ 0.83 m is the standing pelvis height.
        self.n_persons = len(self.person_labels)
        print(f"\nClip {clip_id}: {self.n_persons} person(s), FPS={self.fps}")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for xml_path in self.all_xml_paths:
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        self.state = self.model.state()
        self.jqd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32,
                            device=self.device)

        # ── Frame count ──────────────────────────────────────
        self.T = min(jq.shape[0] for jq in self.all_joint_q)

        # ── Combined joint_q size ────────────────────────────
        self.combined_n_coords = self.model.joint_coord_count
        self.n_per = self.all_joint_q[0].shape[1]  # 76
        print(f"Combined model: {self.combined_n_coords} joint coords "
              f"({self.n_persons} x {self.n_per})")

        # Set initial frame
        self._set_frame(0)
        self.frame = 0

        # Set viewer model
        self.viewer.set_model(self.model)

        # ── Rest pose mode ───────────────────────────────────
        if self.rest_mode:
            # T-pose: upright orientation with all hinge angles zeroed.
            # Place pelvis at an estimated height so feet sit on the ground.
            combined_q = np.zeros(self.combined_n_coords, dtype=np.float32)

            for i, jq in enumerate(self.all_joint_q):
                base = i * self.n_per

                # Step 1: identity root at origin with all hinges zeroed
                combined_q[base + 6] = 1.0  # qw=1 (identity quaternion)

            # FK to find how far below ground the feet are
            self.state.joint_q = wp.array(combined_q, dtype=wp.float32,
                                          device=self.device)
            newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)
            body_q = self.state.body_q.numpy().reshape(-1, 7)

            # Find lowest Z across all persons' ankle/toe bodies
            n_bodies_per = 24  # bodies per person
            min_z = 0.0
            for i in range(self.n_persons):
                off = i * n_bodies_per
                for b in [3, 4, 7, 8]:  # L_Ankle, L_Toe, R_Ankle, R_Toe
                    min_z = min(min_z, float(body_q[off + b, 2]))

            # Step 2: raise pelvis so feet sit at ground level (Z=0)
            foot_clearance = 0.01  # 1cm above ground
            height_offset = -min_z + foot_clearance
            for i in range(self.n_persons):
                base = i * self.n_per
                combined_q[base + 0] = float(i) * 1.0  # spread persons in X
                combined_q[base + 2] = height_offset    # Z = pelvis height

            self.state.joint_q = wp.array(combined_q, dtype=wp.float32,
                                          device=self.device)
            newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

        # ── Camera: always point at character ────────────────
        self._setup_camera()

        if self.rest_mode:
            print(f"\nREST POSE: Root from frame 0, all hinges = 0. Close viewer to exit.")
        else:
            print(f"\nReady! Playing {self.T} frames ({self.T/self.fps:.1f}s). "
                  f"Close viewer to exit.")

    def _setup_camera(self):
        """Position camera to look at the character(s)."""
        # In our Newton world, Z = height (Z-up), X/Y = horizontal plane.
        # Place camera behind the characters (low-Y side), at chest height,
        # looking forward (+Y direction, yaw=90°) with a gentle downward tilt.
        if self.rest_mode:
            # Rest pose places characters near origin
            n = self.n_persons
            center = np.array([(n - 1) * 0.5, 0.0, 0.9])
        else:
            centers = [jq[0, :3] for jq in self.all_joint_q]
            center = np.mean(centers, axis=0)  # (px, py, pz), z = height

        cam_dist = 5.0
        cam_pos = wp.vec3(
            float(center[0]),              # x: centred on characters
            float(center[1]) - cam_dist,   # y: 5 m behind (lower-Y)
            2.0,                           # z: above character height
        )
        # Z-up camera model (camera.py):
        #   front_x = cos(yaw)*cos(pitch)
        #   front_y = sin(yaw)*cos(pitch)
        #   front_z = sin(pitch)
        # yaw=90° → look toward +Y; pitch=-15° → 15° downward tilt
        yaw = 90.0
        pitch = -15.0

        self.viewer.set_camera(cam_pos, pitch, yaw)

    def _set_frame(self, t):
        """Set model state to frame t."""
        combined_q = np.zeros(self.combined_n_coords, dtype=np.float32)

        for i, jq in enumerate(self.all_joint_q):
            frame = min(t, jq.shape[0] - 1)
            combined_q[i * self.n_per:(i + 1) * self.n_per] = jq[frame]

        self.state.joint_q = wp.array(combined_q, dtype=wp.float32,
                                      device=self.device)
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def step(self):
        if self.rest_mode:
            # In rest mode, freeze at frame 0
            return
        
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
        """Side-panel info displayed in the Newton GL viewer window."""
        imgui.separator()
        if self.rest_mode:
            imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.0, 1.0), "[ REST POSE MODE ]")
        else:
            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), "[ MOTION PLAYBACK ]")

        imgui.separator()
        imgui.text(f"Clip:    {self._clip_id}")
        imgui.text(f"Persons: {self.n_persons}")
        imgui.text(f"FPS:     {self.fps}")
        imgui.text(f"Frames:  {self.T}")
        imgui.separator()
        if not self.rest_mode:
            pct = int(100 * self.frame / max(self.T - 1, 1))
            imgui.text(f"Frame:   {self.frame} / {self.T - 1}  ({pct}%)")
        else:
            imgui.text("T-Pose: all hinges = 0")
            imgui.text("Root orientation from frame 0")

        if self.motion_texts:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Motion Description:")
            imgui.spacing()
            for i, text in enumerate(self.motion_texts, 1):
                imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"[{i}]")
                imgui.same_line()
                imgui.text_wrapped(text)



if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, default="1000",
                        help="Clip ID (e.g. '1000' for InterHuman)")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Person index (omit for both)")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman",
                        help="Directory with joint_q + betas .npy files")
    parser.add_argument("--fps", type=int, default=20,
                        help="Playback FPS")
    parser.add_argument("--rest", action="store_true",
                        help="Visualize skeleton in rest pose (all joint angles = 0)")
    parser.add_argument("--annots-dir", type=str, default=None,
                        help="Override annotation directory (auto-detected from --data-dir)")
    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"
    if args.rest:
        print("\n" + "="*60)
        print("REST POSE VISUALIZATION")
        print("="*60)
        print(f"Loading betas for clip {args.clip}, person {args.person if args.person is not None else 'both'}")
        print("All joint angles will be set to 0 (neutral skeleton geometry)")
        print("="*60 + "\n")
    example = PerSubjectVisualizer(viewer, args)
    newton.examples.run(example, args)
