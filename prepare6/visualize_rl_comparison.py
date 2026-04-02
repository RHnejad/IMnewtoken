"""
Visualize GT vs GT-after-PPO vs Generated vs Generated-after-PPO.

Same format as prepare4/view_gt_vs_gen.py: both persons at correct relative
positions, with different representations stacked at Y offsets:

  Y = -offset:  Stick figures (raw 22-joint positions)
  Y = 0:        GT Reference (Newton body, both persons)
  Y = +offset:  GT after RL PPO (Newton body, both persons)
  Y = +2*offset: Gen Reference (Newton body, both persons)
  Y = +3*offset: Gen after RL PPO (Newton body, both persons)

Usage:
    # Run RL tracker for both GT+Gen, both persons, then visualize
    python prepare6/visualize_rl_comparison.py --clip-id 161

    # Load saved results (skip training)
    python prepare6/visualize_rl_comparison.py --clip-id 161 --no-run

    # Single person only
    python prepare6/visualize_rl_comparison.py --clip-id 161 --person 0
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
from prepare4.view_gt_vs_gen import (
    SMPL_BONES, load_gt_positions, load_gen_positions,
)
from prepare5.phc_config import COORDS_PER_PERSON


def run_rl_for_person(clip_id, source, person, device, n_envs, timesteps):
    """Run RL tracker on one person of a clip. Returns (ref_jq, sim_jq, betas)."""
    from prepare5.run_phc_tracker import load_clip, retarget_person
    from prepare6.rl_tracker import RLTracker

    persons, text = load_clip(clip_id, source)
    joint_q, betas = retarget_person(persons[person], source, device=device)

    tracker = RLTracker(
        device=device, n_envs=n_envs, total_timesteps=timesteps, verbose=True,
    )
    result = tracker.train_and_evaluate(joint_q, betas)

    # Save for reuse
    out_dir = os.path.join(
        PROJECT_ROOT, "output", "rl_tracker",
        f"clip_{clip_id}_{source}_p{person}"
    )
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "rl_result.npz"),
        sim_joint_q=result['sim_joint_q'],
        ref_joint_q=joint_q,
        sim_positions=result['sim_positions'],
        ref_positions=result['ref_positions'],
    )

    return joint_q, result['sim_joint_q'], betas, result['mpjpe_mm'], text


def load_saved_rl(clip_id, source, person, device):
    """Load saved RL result. Returns (ref_jq, sim_jq, betas, mpjpe) or None."""
    npz_path = os.path.join(
        PROJECT_ROOT, "output", "rl_tracker",
        f"clip_{clip_id}_{source}_p{person}", "rl_result.npz"
    )
    if not os.path.exists(npz_path):
        return None

    from prepare5.run_phc_tracker import load_clip, retarget_person
    data = np.load(npz_path)
    persons, text = load_clip(clip_id, source)
    betas = persons[person]["betas"]
    if isinstance(betas, np.ndarray) and betas.ndim == 2:
        betas = betas[0]

    ref_jq = data.get('ref_joint_q')
    if ref_jq is None:
        ref_jq, _ = retarget_person(persons[person], source, device=device)

    sim_jq = data['sim_joint_q']
    if 'sim_positions' in data and 'ref_positions' in data:
        mpjpe = float(np.linalg.norm(
            data['sim_positions'] - data['ref_positions'], axis=-1
        ).mean() * 1000)
    else:
        mpjpe = -1.0

    return ref_jq, sim_jq, betas, mpjpe, text


def get_or_run(clip_id, source, person, device, n_envs, timesteps, no_run):
    """Get RL results — load if available, else run."""
    if not no_run:
        saved = load_saved_rl(clip_id, source, person, device)
        if saved is not None:
            print(f"  Loaded saved result for {source} p{person}")
            return saved
        return run_rl_for_person(clip_id, source, person, device, n_envs, timesteps)
    else:
        saved = load_saved_rl(clip_id, source, person, device)
        if saved is None:
            return None
        return saved


class RLComparisonVisualizer:
    """Newton viewer: 4-row comparison (GT ref, GT PPO, Gen ref, Gen PPO).

    Both persons shown at correct relative positions within each row.
    Rows stacked along Y axis. Raw position stick figures also shown.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = args.fps
        self.device = args.device
        self.sim_time = 0.0
        self._wall_start = None
        self._clip_id = args.clip_id

        person_ids = [0, 1] if args.person is None else [args.person]

        # ── Compute Y offset from position extents ──
        gt_positions = load_gt_positions(args.clip_id)
        gen_positions = load_gen_positions(args.clip_id)
        all_y = []
        for pos_list in [gt_positions, gen_positions]:
            for pos, _ in pos_list:
                all_y.append(pos[:, :, 1].max() - pos[:, :, 1].min())
        y_offset = max(all_y) + 2.0 if all_y else 4.0
        self._y_offset = y_offset
        print(f"  Auto y_offset: {y_offset:.1f}m")

        # ── Load reference joint_q directly (no RL needed) ──
        from prepare5.run_phc_tracker import load_clip, retarget_person
        from prepare4.view_gt_vs_gen import load_gen_persons

        all_entries = []
        self._mpjpe = {}
        self._text = ""

        # Load GT reference for all persons
        gt_persons, text = load_clip(args.clip_id, "gt")
        self._text = text
        gt_retargeted = {}
        print(f"\n── GT Reference ──")
        for person in person_ids:
            if person < len(gt_persons):
                jq, betas = retarget_person(gt_persons[person], "gt", device=args.device)
                gt_retargeted[person] = (jq, betas)
                label = f"GT Reference_p{person}"
                all_entries.append((jq.astype(np.float32), betas, label, 0.0))
                print(f"  {label}: {jq.shape[0]} frames")

        # Load Gen reference for all persons
        gen_entries = load_gen_persons(args.clip_id)
        gen_retargeted = {}
        print(f"\n── Gen Reference ──")
        for person in person_ids:
            if person < len(gen_entries):
                jq, betas, glabel = gen_entries[person]
                gen_retargeted[person] = (jq, betas)
                label = f"Gen Reference_p{person}"
                all_entries.append((jq.astype(np.float32), betas, label, 2 * y_offset))
                print(f"  {label}: {jq.shape[0]} frames")

        # Load RL "after PPO" results (only where available)
        for source, source_label, y_off, retargeted in [
            ("gt", "GT after PPO", y_offset, gt_retargeted),
            ("generated", "Gen after PPO", 3 * y_offset, gen_retargeted),
        ]:
            print(f"\n── {source_label} ──")
            for person in person_ids:
                result = get_or_run(
                    args.clip_id, source, person,
                    args.device, args.n_envs, args.total_timesteps, args.no_run,
                )
                if result is None:
                    print(f"  Skipping {source_label}_p{person} (no saved RL result)")
                    continue
                ref_jq, sim_jq, betas, mpjpe, _ = result
                self._mpjpe[f"{source}_p{person}"] = mpjpe
                label = f"{source_label}_p{person}"
                all_entries.append((sim_jq.astype(np.float32), betas, label, y_off))
                print(f"  {label}: {sim_jq.shape[0]} frames, MPJPE={mpjpe:.0f}mm")

        # ── Stick figure entries (raw positions) ──
        self.pos_entries = []

        # GT positions at Y=-offset
        for pos, label in gt_positions:
            if args.person is not None and not label.endswith(f"p{args.person}"):
                continue
            pos_shifted = pos.copy()
            pos_shifted[:, :, 1] -= y_offset
            self.pos_entries.append((pos_shifted, f"GT_{label}", -y_offset))

        # Gen positions at Y=+4*offset
        for pos, label in gen_positions:
            if args.person is not None and not label.endswith(f"p{args.person}"):
                continue
            pos_shifted = pos.copy()
            pos_shifted[:, :, 1] += 4 * y_offset
            self.pos_entries.append((pos_shifted, f"Gen_{label}", 4 * y_offset))

        # ── Fix ground penetration ──
        from prepare4.retarget import forward_kinematics as fk
        for prefix in ["GT Ref", "GT after", "Gen Ref", "Gen after"]:
            idxs = [i for i, (_, _, l, _) in enumerate(all_entries)
                    if l.startswith(prefix)]
            if not idxs:
                continue
            global_min_z = 0.0
            for idx in idxs:
                jq, betas, _, _ = all_entries[idx]
                pos = fk(jq, betas)
                global_min_z = min(global_min_z, float(pos[:, :, 2].min()))
            if global_min_z < 0:
                z_lift = -global_min_z + 0.005
                for idx in idxs:
                    jq, betas, label, y_off = all_entries[idx]
                    jq[:, 2] += z_lift
                    all_entries[idx] = (jq, betas, label, y_off)

        # Fix ground for stick figures
        if self.pos_entries:
            global_min_z = min(float(p[:, :, 2].min()) for p, _, _ in self.pos_entries)
            if global_min_z < 0:
                z_lift = -global_min_z + 0.005
                for i, (pos, lbl, y_off) in enumerate(self.pos_entries):
                    pos[:, :, 2] += z_lift
                    self.pos_entries[i] = (pos, lbl, y_off)

        # ── Apply Y offsets to joint_q ──
        for i, (jq, betas, label, y_off) in enumerate(all_entries):
            if y_off != 0:
                jq[:, 1] += y_off
                all_entries[i] = (jq, betas, label, y_off)

        self.all_entries = all_entries
        self.n_persons = len(all_entries)

        # ── Build Newton model ──
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        for jq, betas, label, y_off in all_entries:
            xml_path = get_or_create_xml(betas, foot_geom="sphere")
            builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        self.jqd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32,
                            device=self.device)

        # ── Frame count ──
        frame_counts = [jq.shape[0] for jq, _, _, _ in all_entries]
        frame_counts += [pos.shape[0] for pos, _, _ in self.pos_entries]
        self.T = min(frame_counts) if frame_counts else 1

        # ── Initial frame + camera ──
        self.frame = 0
        self._set_frame(0)
        self.viewer.set_model(self.model)
        self._setup_camera()

        # ── Summary ──
        print(f"\n{'='*60}")
        print(f"Clip {args.clip_id}: {self.n_persons} bodies + "
              f"{len(self.pos_entries)} stick-figures, {self.T} frames")
        print(f"Text: {self._text}")
        for key, val in self._mpjpe.items():
            print(f"  MPJPE {key}: {val:.0f}mm")
        if 'gt_p0' in self._mpjpe and 'generated_p0' in self._mpjpe:
            gap = self._mpjpe['generated_p0'] - self._mpjpe['gt_p0']
            print(f"  Gap (p0): {gap:+.0f}mm")
        print(f"{'='*60}")

    def _set_frame(self, t):
        combined_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        for i, (jq, _, _, _) in enumerate(self.all_entries):
            base = i * COORDS_PER_PERSON
            frame = min(t, jq.shape[0] - 1)
            combined_q[base:base + COORDS_PER_PERSON] = jq[frame]
        self.state.joint_q = wp.array(combined_q, dtype=wp.float32,
                                      device=self.device)
        newton.eval_fk(self.model, self.state.joint_q, self.jqd, self.state)

    def _draw_stick_figures(self, t):
        gt_colors = [(0.2, 0.5, 1.0), (0.8, 0.2, 0.8)]
        gen_colors = [(0.0, 1.0, 0.4), (1.0, 0.6, 0.0)]

        gt_idx, gen_idx = 0, 0
        for pos, label, y_off in self.pos_entries:
            frame = min(t, pos.shape[0] - 1)
            joints = pos[frame]

            if "GT_" in label:
                color = gt_colors[gt_idx % len(gt_colors)]
                gt_idx += 1
            else:
                color = gen_colors[gen_idx % len(gen_colors)]
                gen_idx += 1

            n_joints = joints.shape[0]
            pts = wp.array(joints, dtype=wp.vec3, device=self.device)
            radii = wp.array(np.full(n_joints, 0.03, dtype=np.float32),
                             dtype=wp.float32, device=self.device)
            colors_arr = wp.full(n_joints, wp.vec3(*color),
                                 dtype=wp.vec3, device=self.device)
            self.viewer.log_points(f"pos_{label}", pts, radii, colors_arr)

            starts = np.array([joints[p] for p, c in SMPL_BONES
                               if p < 22 and c < 22], dtype=np.float32)
            ends = np.array([joints[c] for p, c in SMPL_BONES
                             if p < 22 and c < 22], dtype=np.float32)
            self.viewer.log_lines(
                f"bone_{label}",
                wp.array(starts, dtype=wp.vec3, device=self.device),
                wp.array(ends, dtype=wp.vec3, device=self.device),
                color, width=0.012,
            )

    def _setup_camera(self):
        centers = [jq[0, :3] for jq, _, _, _ in self.all_entries]
        centers += [pos[0, 0] for pos, _, _ in self.pos_entries]
        center = np.mean(centers, axis=0)
        y_vals = [c[1] for c in centers]
        y_extent = max(y_vals) - min(y_vals) if y_vals else 4.0
        scale = max(1.0, (y_extent / 8.0) ** 0.65)

        dist = 10.0 * scale
        height = 3.0 * scale
        cam_pos = wp.vec3(float(center[0]) + dist, float(center[1]), height)
        self.viewer.set_camera(cam_pos, -10.0, 180.0)

    def step(self):
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        self.sim_time = time.perf_counter() - self._wall_start
        self.frame = int(self.sim_time * self.fps) % self.T
        self._set_frame(self.frame)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self._draw_stick_figures(self.frame)
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.separator()
        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.8, 1.0),
                           "[ RL TRACKING COMPARISON ]")
        imgui.separator()
        imgui.text(f"Clip:  {self._clip_id}")
        pct = int(100 * self.frame / max(self.T - 1, 1))
        imgui.text(f"Frame: {self.frame} / {self.T - 1}  ({pct}%)")
        imgui.separator()

        imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Rows (Y offset):")

        row_colors = {
            "GT Ref": imgui.ImVec4(0.6, 0.9, 0.6, 1.0),
            "GT after": imgui.ImVec4(0.4, 1.0, 0.6, 1.0),
            "Gen Ref": imgui.ImVec4(0.9, 0.9, 0.4, 1.0),
            "Gen after": imgui.ImVec4(1.0, 0.5, 0.3, 1.0),
        }
        for jq, betas, label, y_off in self.all_entries:
            color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
            for prefix, c in row_colors.items():
                if label.startswith(prefix):
                    color = c
                    break
            imgui.text_colored(color, f"  {label} [Y={y_off:.1f}]")

        for pos, label, y_off in self.pos_entries:
            if "GT" in label:
                c = imgui.ImVec4(0.3, 0.6, 1.0, 1.0)
            else:
                c = imgui.ImVec4(0.0, 1.0, 0.4, 1.0)
            imgui.text_colored(c, f"  {label} [Y={y_off:.1f}] (stick)")

        imgui.separator()
        imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "MPJPE:")
        for key, val in self._mpjpe.items():
            imgui.text(f"  {key}: {val:.0f}mm")
        if 'gt_p0' in self._mpjpe and 'generated_p0' in self._mpjpe:
            gap = self._mpjpe['generated_p0'] - self._mpjpe['gt_p0']
            color = imgui.ImVec4(0.4, 1.0, 0.4, 1.0) if gap > 0 else \
                    imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
            imgui.text_colored(color, f"  Gap: {gap:+.0f}mm")

        if self._text:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Description:")
            imgui.text_wrapped(self._text)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip-id", type=int, default=161)
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Single person (default: both)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-run", action="store_true",
                        help="Load saved results only (don't run tracker)")
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200000)

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda:0"

    example = RLComparisonVisualizer(viewer, args)
    newton.examples.run(example, args)
