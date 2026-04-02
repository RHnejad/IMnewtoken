"""
visualize_newton_headless_4way.py — Newton GL headless 4-way comparison → MP4.

Uses Newton's offscreen GL renderer (same as prepare4/view_gt_vs_gen.py --save-mp4)
to render 4 humanoids side-by-side:
  1. GT Reference (input)
  2. GT after PPO (physics-corrected)
  3. Generated Reference
  4. Generated after PPO (physics-corrected)

Requires: conda mimickit (newton + warp), X display or virtual framebuffer.

Usage:
    # From existing npz files (must have sim_joint_q + ref_joint_q)
    conda run -n mimickit --no-capture-output python prepare6/visualize_newton_headless_4way.py \
        --clip-id 161 --headless

    # Re-run RL tracker first, then render
    conda run -n mimickit --no-capture-output python prepare6/visualize_newton_headless_4way.py \
        --clip-id 161 --run --headless

    # Custom resolution
    conda run -n mimickit --no-capture-output python prepare6/visualize_newton_headless_4way.py \
        --clip-id 161 --headless --mp4-width 1920 --mp4-height 1080
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np

# Start virtual display if headless (before pyglet import)
_vdisplay = None
if "--headless" in sys.argv:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        from pyvirtualdisplay import Display
        _vdisplay = Display(visible=False, size=(1280, 720))
        _vdisplay.start()
    except Exception:
        pass

import warp as wp
wp.config.verbose = False
warnings.filterwarnings("ignore", message="Custom attribute")

import newton
import newton.examples

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare4.gen_xml import get_or_create_xml
from prepare5.phc_config import COORDS_PER_PERSON


class FourWayHeadless:
    """Newton GL 4-way comparison with headless MP4 recording."""

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.sim_time = 0.0
        self._wall_start = None
        self.viewer = viewer
        self.device = args.device
        self._clip_id = args.clip_id
        self._x_spacing = 2.5

        self._load_data(args)
        self._build_viewer_model()
        self.frame = 0
        self._set_frame(0)
        self.viewer.set_model(self.model)
        self._setup_camera()

        print(f"\nReady: {self.T} frames, 4 humanoids.")

    def _load_data(self, args):
        """Load joint_q arrays for all 4 versions."""
        from prepare5.run_phc_tracker import load_clip, retarget_person

        persons_gt, self._text = load_clip(args.clip_id, "gt")
        self._betas = persons_gt[args.person]["betas"]
        if isinstance(self._betas, np.ndarray) and self._betas.ndim == 2:
            self._betas = self._betas[0]

        if args.run:
            self._run_both(args)
        else:
            self._load_from_npz(args)

    def _run_both(self, args):
        """Run RL tracker on GT and generated to get joint_q."""
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
        """Load from saved npz files (must have sim_joint_q + ref_joint_q)."""
        out_dir = os.path.join(PROJECT_ROOT, "output", "rl_comparison",
                               f"clip_{args.clip_id}")
        gt_path = args.gt_npz or os.path.join(out_dir, "gt_result.npz")
        gen_path = args.gen_npz or os.path.join(out_dir, "gen_result.npz")

        gt_data = np.load(gt_path)
        gen_data = np.load(gen_path)

        for label, data, path in [("GT", gt_data, gt_path), ("Gen", gen_data, gen_path)]:
            if 'sim_joint_q' not in data or 'ref_joint_q' not in data:
                raise ValueError(
                    f"{label} npz at {path} missing sim_joint_q/ref_joint_q. "
                    f"Re-run with --run to generate them.")

        self._gt_ref_jq = gt_data['ref_joint_q']
        self._gt_sim_jq = gt_data['sim_joint_q']
        self._gen_ref_jq = gen_data['ref_joint_q']
        self._gen_sim_jq = gen_data['sim_joint_q']

        self._gt_mpjpe = np.linalg.norm(
            gt_data['sim_positions'] - gt_data['ref_positions'], axis=-1
        ).mean() * 1000
        self._gen_mpjpe = np.linalg.norm(
            gen_data['sim_positions'] - gen_data['ref_positions'], axis=-1
        ).mean() * 1000

        self.T = min(self._gt_ref_jq.shape[0], self._gen_ref_jq.shape[0])

    def _build_viewer_model(self):
        """Build Newton model with 4 humanoids."""
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

    def _set_frame(self, t):
        """Set all 4 humanoids to frame t with X offsets."""
        combined_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        C = COORDS_PER_PERSON

        jq_list = [
            self._gt_ref_jq,
            self._gt_sim_jq,
            self._gen_ref_jq,
            self._gen_sim_jq,
        ]

        for i, jq_arr in enumerate(jq_list):
            frame = min(t, jq_arr.shape[0] - 1)
            jq = jq_arr[frame].astype(np.float32).copy()
            jq[0] += i * self._x_spacing
            combined_q[i * C:(i + 1) * C] = jq

        self.state.joint_q = wp.array(
            combined_q, dtype=wp.float32, device=self.device
        )
        self.jqd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def _setup_camera(self):
        """Position camera to see all 4 humanoids."""
        center_x = 1.5 * self._x_spacing
        cam_pos = wp.vec3(center_x, -8.0, 2.5)
        self.viewer.set_camera(cam_pos, -10.0, 90.0)

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


def record_mp4(viewer, example, mp4_path, width=1280, height=720):
    """Record all frames to MP4 using Newton's offscreen renderer."""
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
    writer = imageio.get_writer(mp4_path, fps=example.fps, codec='libx264', quality=8)

    # Load font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_title = font

    # Load annotation
    annots_path = os.path.join(PROJECT_ROOT, "data", "InterHuman", "annots",
                               f"{example._clip_id}.txt")
    motion_text = ""
    if os.path.isfile(annots_path):
        with open(annots_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            motion_text = lines[0]

    labels = ["GT Ref", "GT PPO", "Gen Ref", "Gen PPO"]
    colors = [(50, 128, 255), (255, 80, 80), (80, 200, 80), (255, 160, 40)]

    gap = example._gen_mpjpe - example._gt_mpjpe

    def overlay(frame_np, t):
        img = Image.fromarray(frame_np)
        draw = ImageDraw.Draw(img, "RGBA")

        # Background box
        box_w = 420
        box_h = 30 + len(labels) * 20 + 40
        if motion_text:
            box_h += 20
        draw.rectangle([5, 5, 5 + box_w, 5 + box_h], fill=(0, 0, 0, 160))

        # Title
        draw.text((10, 8),
                  f"Clip {example._clip_id}  frame {t}/{example.T - 1}",
                  fill=(255, 255, 255), font=font_title)
        y = 28

        if motion_text:
            draw.text((10, y), motion_text[:55], fill=(200, 200, 200), font=font)
            y += 18

        # Legend
        for label, color in zip(labels, colors):
            draw.rectangle([10, y + 2, 22, y + 14], fill=color)
            draw.text((28, y), label, fill=(255, 255, 255), font=font)
            y += 20

        # Metrics
        y += 5
        draw.text((10, y),
                  f"GT: {example._gt_mpjpe:.0f}mm  Gen: {example._gen_mpjpe:.0f}mm  "
                  f"Gap: {gap:+.0f}mm",
                  fill=(255, 255, 100), font=font)

        return np.array(img)

    print(f"Recording {example.T} frames to {mp4_path} ...")
    t0 = time.time()

    for t in range(example.T):
        sim_time = t / example.fps
        example.frame = t
        example._set_frame(t)

        viewer.begin_frame(sim_time)
        viewer.log_state(example.state)
        viewer.end_frame()

        frame_gpu = viewer.get_frame()
        frame_np = frame_gpu.numpy()
        frame_np = overlay(frame_np, t)
        writer.append_data(frame_np)

    writer.close()
    elapsed = time.time() - t0
    print(f"Saved: {mp4_path} ({example.T} frames in {elapsed:.1f}s)")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip-id", type=int, default=161)
    parser.add_argument("--run", action="store_true",
                        help="Run RL tracker first")
    parser.add_argument("--gt-npz", type=str, default=None)
    parser.add_argument("--gen-npz", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--person", type=int, default=0, choices=[0, 1])
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--save-mp4", type=str, default=None,
                        help="Output MP4 path (auto if not given)")
    parser.add_argument("--mp4-width", type=int, default=1280)
    parser.add_argument("--mp4-height", type=int, default=720)

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"

    example = FourWayHeadless(viewer, args)

    if args.save_mp4 or "--headless" in sys.argv:
        mp4_path = args.save_mp4 or os.path.join(
            PROJECT_ROOT, "output", "rl_comparison",
            f"clip_{args.clip_id}", "newton_4way_gl.mp4")
        record_mp4(viewer, example, mp4_path,
                   width=args.mp4_width, height=args.mp4_height)
    else:
        newton.examples.run(example, args)
