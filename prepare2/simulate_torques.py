"""
Visualize motion driven by PD torques in Newton physics simulation.

Uses SolverMuJoCo with explicit PD control to track retargeted
joint_q reference trajectories. Virtual forces on each root keep the
characters at reference positions/orientations; joint PD torques
track the hinge angles. Gravity, contacts, and dynamics are active.

Supports both persons side-by-side (default) or a single person.

Key simulation parameters (discovered empirically):
  - SolverMuJoCo with solver="newton" (implicit contact solver)
  - Armature: 0.5 (hinges), 5.0 (root) — prevents instability
    from low body inertia of SMPL-X thin limbs
  - Explicit PD: τ = kp*(q_ref - q) - kd*q̇, applied via joint_f
  - Per-body gains from MJCF (Torso~1000, Hip~500, Hand~50 Nm/rad)
  - Sim frequency 480 Hz (24 substeps at 20 Hz control)

Usage:
    # Both persons side-by-side (default)
    python prepare2/simulate_torques.py --clip 1000

    # Single person only
    python prepare2/simulate_torques.py --clip 1000 --person 1

    # Adjust PD tracking stiffness
    python prepare2/simulate_torques.py --clip 1000 --gain-scale 1.5
"""
import os
import sys
import time
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

from prepare2.retarget import get_or_create_xml
from prepare2.pd_utils import (
    DOFS_PER_PERSON, COORDS_PER_PERSON,
    BODY_GAINS, DEFAULT_SIM_FREQ, DEFAULT_TORQUE_LIMIT,
    build_pd_gains, setup_model_properties, create_mujoco_solver,
    compute_all_pd_torques_np, downsample_trajectory,
    create_contact_sensors, update_contact_sensors,
)

# Annotations directory per dataset name (mirrors utils/get_opt.py mapping)
_DATASET_ANNOTS = {
    "interhuman": os.path.join(PROJECT_ROOT, "data", "InterHuman", "annots"),
    "interx":     os.path.join(PROJECT_ROOT, "data", "Inter-X_Dataset", "texts_processed"),
}


def _infer_annots_dir(data_dir: str) -> str | None:
    dataset_name = os.path.basename(data_dir.rstrip("/\\")).lower()
    return _DATASET_ANNOTS.get(dataset_name)


# ═══════════════════════════════════════════════════════════════
# Force plot window (ImGui overlay)
# ═══════════════════════════════════════════════════════════════

class ForcePlotWindow:
    """
    Rolling-window ImGui plot for contact forces.

    Shows foot GRF and hand contact forces as separate time-series
    graphs in a floating window overlay. Used when --plot-forces is set.
    """

    def __init__(self, viewer, n_points=150, n_foot_groups=8, n_hand_groups=4):
        self.viewer = viewer
        self.n_points = n_points
        self.n_foot = n_foot_groups
        self.n_hand = n_hand_groups

        # Rolling buffers: vertical GRF per foot group
        self.foot_fz = np.zeros((n_foot_groups, n_points), dtype=np.float32)
        # Combined left/right per person (summed Fz)
        self.person_grf = np.zeros((2, n_points), dtype=np.float32)
        # Hand contact magnitudes
        self.hand_mag = np.zeros((n_hand_groups, n_points), dtype=np.float32)
        # Total hand contact (any hand touching anything)
        self.total_hand = np.zeros(n_points, dtype=np.float32)

    def update(self, contact_forces):
        """Push one frame of contact data into the rolling buffers."""
        if contact_forces is None:
            return

        foot_f = contact_forces.get('foot_forces')
        if foot_f is not None:
            # Column 0 = total force (include_total=True)
            for g in range(min(foot_f.shape[0], self.n_foot)):
                fz = float(foot_f[g, 0, 2])
                self.foot_fz[g] = np.roll(self.foot_fz[g], -1)
                self.foot_fz[g, -1] = abs(fz)

            # Person 0: foot groups 0-3 (L_Ankle, L_Toe, R_Ankle, R_Toe)
            # Person 1: foot groups 4-7
            for p in range(2):
                start = p * 4
                end = min(start + 4, foot_f.shape[0])
                total_fz = sum(
                    abs(float(foot_f[g, 0, 2]))
                    for g in range(start, end)
                )
                self.person_grf[p] = np.roll(self.person_grf[p], -1)
                self.person_grf[p, -1] = total_fz

        hand_f = contact_forces.get('hand_forces')
        if hand_f is not None:
            total = 0.0
            for g in range(min(hand_f.shape[0], self.n_hand)):
                mag = float(np.linalg.norm(hand_f[g, 0]))
                self.hand_mag[g] = np.roll(self.hand_mag[g], -1)
                self.hand_mag[g, -1] = mag
                total += mag
            self.total_hand = np.roll(self.total_hand, -1)
            self.total_hand[-1] = total

    def render(self, imgui):
        """ImGui callback for rendering the force plot window."""
        if not self.viewer or not self.viewer.ui.is_available:
            return

        io = self.viewer.ui.io
        win_w, win_h = 420, 520
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - win_w - 10,
                         io.display_size[1] - win_h - 10),
        )
        imgui.set_next_window_size(imgui.ImVec2(win_w, win_h))

        flags = imgui.WindowFlags_.no_resize.value
        if imgui.begin("Contact Forces", flags=flags):
            graph_size = imgui.ImVec2(win_w - 30, 100)

            # ── GRF per person ───────────────────────────
            imgui.text_colored(
                imgui.ImVec4(0.5, 1.0, 1.0, 1.0), "Ground Reaction Force (Fz)")
            for p in range(min(2, self.person_grf.shape[0])):
                peak = float(self.person_grf[p].max())
                current = float(self.person_grf[p, -1])
                imgui.text(f"Person {p}: {current:.0f}N (peak {peak:.0f}N)")
                imgui.plot_lines(
                    f"##grf_p{p}", self.person_grf[p],
                    scale_min=0.0,
                    graph_size=graph_size,
                )

            imgui.separator()

            # ── Hand contact ─────────────────────────────
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.8, 0.3, 1.0), "Hand Contact Forces")
            current = float(self.total_hand[-1])
            peak = float(self.total_hand.max())
            imgui.text(f"Total: {current:.0f}N (peak {peak:.0f}N)")
            imgui.plot_lines(
                "##hand_total", self.total_hand,
                scale_min=0.0,
                graph_size=graph_size,
            )
        imgui.end()


class TorqueSimVisualizer:
    """
    Newton physics simulation driven by explicit PD torques.

    Supports one or two persons side-by-side. Uses SolverMuJoCo for
    stability with the thin-limbed SMPL-X model. Virtual forces on each
    root free joint keep characters at reference positions. Joint PD
    torques track reference hinge angles.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device if args.device else "cuda:0"
        self.fps = args.fps
        self.downsample = getattr(args, 'downsample', 1)
        self._wall_start = None
        self.sim_time = 0.0
        self._clip_id = args.clip
        gain_scale = args.gain_scale
        self.torque_mode = getattr(args, "torque_mode", "pd")

        # ── Simulation parameters ────────────────────────────
        self.sim_freq = DEFAULT_SIM_FREQ
        self.sim_substeps = self.sim_freq // self.fps  # substeps per frame
        self.sim_dt = 1.0 / self.sim_freq
        self.torque_limit = DEFAULT_TORQUE_LIMIT

        data_dir = args.data_dir
        self.torque_dir = getattr(args, 'torque_dir', None) or data_dir
        clip_id = args.clip

        # ── Determine persons to load ────────────────────────
        if args.person is not None:
            personas = [args.person]
        else:
            personas = [0, 1]

        # ── Load motion text descriptions ─────────────────────
        self.motion_texts = []
        annots_dir = (
            (args.annots_dir if hasattr(args, 'annots_dir') else None)
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

        # ── Load reference trajectories ──────────────────────
        self.all_ref_jq = []
        self.all_precomputed_torques = []  # for "full" or "solo" torque modes
        self.person_labels = []

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        for p_idx in personas:
            jq_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_joint_q.npy")
            betas_path = os.path.join(data_dir, f"{clip_id}_person{p_idx}_betas.npy")

            if not os.path.exists(jq_path):
                print(f"joint_q not found: {jq_path}")
                continue
            if not os.path.exists(betas_path):
                print(f"betas not found: {betas_path}")
                continue

            jq = np.load(jq_path).astype(np.float32)
            betas = np.load(betas_path)
            xml_path = get_or_create_xml(betas)

            # Downsample from data FPS to target FPS (e.g. 60→30)
            jq = downsample_trajectory(jq, self.downsample)

            builder.add_mjcf(xml_path, enable_self_collisions=False)
            self.all_ref_jq.append(jq)
            self.person_labels.append(f"person{p_idx}")
            print(f"Loaded {self.person_labels[-1]}: {jq.shape[0]} frames, "
                  f"XML={os.path.basename(xml_path)}")

            # Load precomputed torques if available and requested
            # Look in --torque-dir first, fall back to --data-dir
            tdir = self.torque_dir
            if self.torque_mode == "full":
                full_path = os.path.join(tdir, f"{clip_id}_person{p_idx}_torques_full.npy")
                solo_path = os.path.join(tdir, f"{clip_id}_person{p_idx}_torques_solo.npy")
                if os.path.exists(full_path):
                    t = np.load(full_path).astype(np.float32)
                    if self.downsample > 1:
                        t = t[::self.downsample]
                    self.all_precomputed_torques.append(t)
                    print(f"  Loaded full torques (solo+ΔT): {t.shape}")
                elif os.path.exists(solo_path):
                    t = np.load(solo_path).astype(np.float32)
                    if self.downsample > 1:
                        t = t[::self.downsample]
                    self.all_precomputed_torques.append(t)
                    print(f"  Loaded solo torques (no ΔT found): {t.shape}")
                else:
                    self.all_precomputed_torques.append(None)
                    print(f"  WARNING: no precomputed torques, falling back to PD")
            elif self.torque_mode == "solo":
                solo_path = os.path.join(tdir, f"{clip_id}_person{p_idx}_torques_solo.npy")
                if os.path.exists(solo_path):
                    t = np.load(solo_path).astype(np.float32)
                    if self.downsample > 1:
                        t = t[::self.downsample]
                    self.all_precomputed_torques.append(t)
                    print(f"  Loaded solo torques: {t.shape}")
                else:
                    self.all_precomputed_torques.append(None)
                    print(f"  WARNING: {os.path.basename(solo_path)} not found "
                          f"in {tdir}, falling back to PD")

        if not self.person_labels:
            raise FileNotFoundError(
                f"No joint_q/betas files found for clip {clip_id} in {data_dir}\n"
                f"Run: python prepare2/retarget.py --dataset interhuman "
                f"--clip {clip_id}"
            )

        self.n_persons = len(self.person_labels)
        self.T = min(jq.shape[0] for jq in self.all_ref_jq)
        print(f"\nClip {clip_id}: {self.n_persons} person(s), "
              f"{self.T} frames, FPS={self.fps}, torques={self.torque_mode}")
        if self.torque_mode != "pd":
            print(f"  Torque dir: {self.torque_dir}")

        # ── Build Newton model ────────────────────────────────
        builder.add_ground_plane()
        self.model = builder.finalize(device=self.device)

        n_dof = self.model.joint_dof_count
        n_coords = self.model.joint_coord_count

        # ── Configure model (passive springs, armature) ──────
        setup_model_properties(self.model, self.n_persons, device=self.device)

        # ── Build per-DOF PD gains (replicated per person) ──
        self.kp, self.kd = build_pd_gains(
            self.model, self.n_persons, gain_scale=gain_scale
        )

        # ── Initialize solver (MuJoCo for stability) ────────
        self.solver = create_mujoco_solver(self.model, self.n_persons)

        # ── Contact sensors (feet GRF + hand inter-person) ──
        self.sensor_dict = create_contact_sensors(
            self.model, self.solver, self.n_persons, verbose=True
        )
        self._last_contact_forces = None

        # ── Force plot window (optional, --plot-forces) ──────
        self.plot_forces = getattr(args, 'plot_forces', False)
        self._force_plot = None
        if self.plot_forces and self.sensor_dict is not None:
            n_foot = len(self.sensor_dict['foot_sensor'].sensing_objs)
            n_hand = (len(self.sensor_dict['hand_sensor'].sensing_objs)
                      if self.sensor_dict['hand_sensor'] else 0)
            self._force_plot = ForcePlotWindow(
                viewer, n_points=150,
                n_foot_groups=n_foot, n_hand_groups=n_hand,
            )
            if hasattr(viewer, 'register_ui_callback'):
                viewer.register_ui_callback(self._force_plot.render, "free")
            print("Force plotting enabled (--plot-forces)")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Set initial state from first frame (all persons combined)
        init_q = np.zeros(n_coords, dtype=np.float32)
        for i, jq in enumerate(self.all_ref_jq):
            c = i * COORDS_PER_PERSON
            init_q[c:c + COORDS_PER_PERSON] = jq[0]

        self.state_0.joint_q = wp.array(
            init_q, dtype=wp.float32, device=self.device
        )
        self.state_0.joint_qd = wp.zeros(
            n_dof, dtype=wp.float32, device=self.device
        )
        newton.eval_fk(
            self.model, self.state_0.joint_q,
            self.state_0.joint_qd, self.state_0,
        )

        self.n_dof = n_dof
        self.n_coords = n_coords
        self.frame = 0

        # Set viewer model
        self.viewer.set_model(self.model)

        # ── Camera: point at characters ──────────────────────
        self._setup_camera()
        print(f"Model: {n_dof} DOF ({self.n_persons} person(s)), "
              f"sim_dt={self.sim_dt:.5f}s "
              f"({self.sim_substeps} substeps/frame)")
        print(f"Gains (scaled {gain_scale:.1f}x): "
              f"root_pos={self.kp[0]:.0f} root_rot={self.kp[3]:.0f}")
        print(f"\nReady! {self.T} frames ({self.T/self.fps:.1f}s). "
              f"Close viewer to exit.")

    def _setup_camera(self):
        """Position camera to look at the character(s) (Z-up).

        Camera is placed behind the characters (negative Y direction),
        elevated slightly above character height, looking down at them.
        Convention matches Newton's AMASS motion transfer example.
        """
        centers = [jq[0, :3] for jq in self.all_ref_jq]
        center = np.mean(centers, axis=0).astype(float)

        cam_dist = 5.0
        cam_pos = wp.vec3(
            float(center[0]),
            float(center[1]) - cam_dist,
            2.0,   # slightly above character height (~0.8m pelvis)
        )
        self.viewer.set_camera(cam_pos, -15.0, 90.0)

    def _compute_pd_torques(self, frame_idx):
        """
        Compute torques for all persons using shared PD utility.

        If torque_mode is "full" or "solo", uses precomputed torques
        (with 10% PD correction). Otherwise computes full PD from scratch.
        """
        cq = self.state_0.joint_q.numpy()
        cqd = self.state_0.joint_qd.numpy()

        # Determine precomputed torques and PD scale
        has_precomputed = (
            self.torque_mode in ("full", "solo")
            and self.all_precomputed_torques
        )
        precomputed = self.all_precomputed_torques if has_precomputed else None

        return compute_all_pd_torques_np(
            cq, cqd, self.all_ref_jq, frame_idx,
            self.kp, self.kd, self.n_persons,
            pd_scale=1.0, torque_limit=self.torque_limit,
            precomputed_torques=precomputed,
            precomputed_pd_scale=0.1,
        )

    def step(self):
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now

        self.sim_time = now - self._wall_start
        target_frame = int(self.sim_time * self.fps) % self.T

        # Apply PD torques at each substep for stability
        for _ in range(self.sim_substeps):
            tau = self._compute_pd_torques(target_frame)
            self.control.joint_f = wp.array(
                tau, dtype=wp.float32, device=self.device
            )
            contacts = self.model.collide(self.state_0)
            self.solver.step(
                self.state_0, self.state_1,
                self.control, contacts, self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Read contact sensors (once per control frame, uses last substep)
        self._last_contact_forces = update_contact_sensors(
            self.solver, self.state_0, self.sensor_dict
        )

        # Feed force plot
        if self._force_plot is not None:
            self._force_plot.update(self._last_contact_forces)

        self.frame = target_frame

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, imgui):
        """Side-panel info displayed in the Newton GL viewer window."""
        imgui.separator()
        mode_label = {
            "pd": "PD TORQUE SIM",
            "solo": "SOLO TORQUE SIM",
            "full": "FULL TORQUE SIM (solo+ΔT)",
        }.get(self.torque_mode, "TORQUE SIM")
        color = {
            "pd": imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
            "solo": imgui.ImVec4(0.4, 0.8, 1.0, 1.0),
            "full": imgui.ImVec4(1.0, 0.8, 0.4, 1.0),
        }.get(self.torque_mode, imgui.ImVec4(0.4, 1.0, 0.4, 1.0))
        imgui.text_colored(color, f"[ {mode_label} ]")
        imgui.separator()
        imgui.text(f"Clip:      {self._clip_id}")
        imgui.text(f"Persons:   {self.n_persons}")
        imgui.text(f"FPS:       {self.fps}")
        imgui.text(f"Frames:    {self.T}")
        imgui.text(f"Sim freq:  {self.sim_freq} Hz")
        imgui.text(f"Substeps:  {self.sim_substeps}")
        imgui.text(f"Torques:   {self.torque_mode}")
        if self.torque_mode != "pd":
            # Show truncated torque directory
            td = self.torque_dir
            if len(td) > 35:
                td = "..." + td[-32:]
            imgui.text(f"Torque dir: {td}")
        imgui.separator()
        pct = int(100 * self.frame / max(self.T - 1, 1))
        imgui.text(f"Frame:     {self.frame} / {self.T - 1}  ({pct}%)")

        if self.motion_texts:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.5, 1.0), "Motion Description:")
            imgui.spacing()
            for i, text in enumerate(self.motion_texts, 1):
                imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"[{i}]")
                imgui.same_line()
                imgui.text_wrapped(text)

        # ── Contact sensor readout ───────────────────────────
        if self._last_contact_forces is not None:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.5, 1.0, 1.0, 1.0), "Contact Forces:")

            foot_f = self._last_contact_forces.get('foot_forces')
            if foot_f is not None:
                # foot_f shape: (n_groups, n_counterparts, 3)
                # Column 0 = total (include_total=True), so use [g, 0]
                for g in range(foot_f.shape[0]):
                    fz = float(foot_f[g, 0, 2])  # vertical GRF (total)
                    f_mag = float(np.linalg.norm(foot_f[g, 0]))
                    label = f"Foot {g}"
                    if fz > 1.0:
                        imgui.text_colored(
                            imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
                            f"  {label}: {f_mag:.0f}N (Fz={fz:.0f}N)")
                    else:
                        imgui.text(f"  {label}: {f_mag:.0f}N")

            hand_f = self._last_contact_forces.get('hand_forces')
            if hand_f is not None:
                for g in range(hand_f.shape[0]):
                    f_mag = float(np.linalg.norm(hand_f[g, 0]))
                    label = f"Hand {g}"
                    if f_mag > 1.0:
                        imgui.text_colored(
                            imgui.ImVec4(1.0, 0.8, 0.3, 1.0),
                            f"  {label}: {f_mag:.0f}N (CONTACT)")
                    else:
                        imgui.text(f"  {label}: {f_mag:.0f}N")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--clip", type=str, default="1000",
                        help="Clip ID")
    parser.add_argument("--person", type=int, default=None, choices=[0, 1],
                        help="Person index (omit for both)")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman",
                        help="Directory with joint_q + betas files")
    parser.add_argument("--fps", type=int, default=30,
                        help="Motion FPS (default 30 = InterMask eval rate)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample loaded data by this factor "
                             "(2 = 60→30fps to match InterMask)")
    parser.add_argument("--gain-scale", type=float, default=1.0,
                        help="Scale all PD gains (>1 = stiffer tracking)")
    parser.add_argument("--torque-mode", type=str, default="pd",
                        choices=["pd", "solo", "full"],
                        help="pd=PD tracking, solo=inverse dynamics torques, "
                             "full=solo+ΔT (after optimization)")
    parser.add_argument("--torque-dir", type=str, default=None,
                        help="Directory to load precomputed torques from "
                             "(for --torque-mode solo/full). "
                             "Defaults to --data-dir if not specified")
    parser.add_argument("--annots-dir", type=str, default=None,
                        help="Override annotation directory (auto-detected from --data-dir)")
    parser.add_argument("--plot-forces", action="store_true",
                        help="Show real-time contact force plots (GRF + hand contacts)")

    viewer, args = newton.examples.init(parser)
    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda:0"

    example = TorqueSimVisualizer(viewer, args)
    newton.examples.run(example, args)
