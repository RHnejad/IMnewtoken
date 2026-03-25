#!/usr/bin/env python
"""
visualize_policy.py — Visualize trained RL policy rollouts.

Supports two modes:
  1. Newton GL viewer (interactive, requires display)
  2. Headless MP4 video (no display needed, saves skeleton animation)

Shows a physics-simulated character controlled by a trained PPO policy
tracking a reference motion, with optional reference "ghost" overlay.

Usage:
    # === HEADLESS MP4 (recommended for SSH / no display) ===

    # Zero-action: PD tracks reference, save MP4
    python prepare3/visualize_policy.py \
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \
        --zero-action --headless --mp4 output/prepare3/test_run/rollout.mp4

    # Trained policy rollout → MP4
    python prepare3/visualize_policy.py \
        --checkpoint output/prepare3/test_run/best_policy.pt \
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \
        --headless --mp4 output/prepare3/test_run/policy_rollout.mp4

    # With reference ghost overlay
    python prepare3/visualize_policy.py \
        --checkpoint output/prepare3/test_run/best_policy.pt \
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy \
        --headless --show-ref --mp4 output/prepare3/test_run/compare.mp4

    # === INTERACTIVE GL VIEWER (requires display) ===

    python prepare3/visualize_policy.py \
        --checkpoint output/prepare3/test_run/best_policy.pt \
        --motion data/mimickit_motions/interhuman/1000_person0.pkl \
        --betas data/retargeted_v2/interhuman/1000_person0_betas.npy
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

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prepare3.newton_mimic_env import NewtonMimicEnv, DOFS_PER_PERSON, COORDS_PER_PERSON
from prepare3.train import ActorCritic, RunningMeanStd

# ═══════════════════════════════════════════════════════════════
# Skeleton topology for 3D rendering
# ═══════════════════════════════════════════════════════════════
# Body indices: 0=Pelvis, 1=L_Hip, 2=L_Knee, 3=L_Ankle, 4=L_Toe,
#   5=R_Hip, 6=R_Knee, 7=R_Ankle, 8=R_Toe, 9=Torso, 10=Spine,
#   11=Chest, 12=Neck, 13=Head, 14=L_Thorax, 15=L_Shoulder,
#   16=L_Elbow, 17=L_Wrist, 18=L_Hand, 19=R_Thorax, 20=R_Shoulder,
#   21=R_Elbow, 22=R_Wrist, 23=R_Hand
#
# parent indices: [-1,0,1,2,3, 0,5,6,7, 0,9,10, 11,12, 11,14,15,16,17, 11,19,20,21,22]
SKELETON_BONES = [
    # Left leg
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Right leg
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Spine
    (0, 9), (9, 10), (10, 11),
    # Neck/head
    (11, 12), (12, 13),
    # Left arm
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    # Right arm
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23),
]

# ═══════════════════════════════════════════════════════════════
# Headless MP4 renderer
# ═══════════════════════════════════════════════════════════════

def _get_body_positions_from_joint_q(model, joint_q_np, device="cuda:0"):
    """Run FK and return body positions (N, 3) from a joint_q array."""
    n_dof = model.joint_dof_count
    state = model.state()
    state.joint_q = wp.array(joint_q_np, dtype=wp.float32, device=device)
    jqd = wp.zeros(n_dof, dtype=wp.float32, device=device)
    newton.eval_fk(model, state.joint_q, jqd, state)
    body_q = state.body_q.numpy().reshape(-1, 7)
    return body_q[:, :3]  # (N_bodies, 3) positions


def render_headless_mp4(args):
    """Run a full rollout headlessly and save a 3D skeleton MP4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    device = args.device or "cuda:0"

    # ── Build environment ────────────────────────────────────
    env_config = {
        "motion_file": args.motion,
        "betas_file": args.betas,
        "device": device,
        "sim_freq": args.sim_freq,
        "control_freq": args.control_freq,
        "control_mode": "pd",
        "max_episode_length": 999.0,
        "enable_early_termination": not args.no_termination,
        "rand_init": False,
        "enable_tar_obs": True,
    }
    env = NewtonMimicEnv(env_config)
    obs_dim = env.get_obs_dim()
    action_dim = env.get_action_dim()
    ref_T = env.ref_T
    ref_duration = env.ref_duration

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")
    print(f"Ref motion:      {ref_T} frames, {ref_duration:.2f}s")

    # ── Load policy ──────────────────────────────────────────
    policy = None
    obs_normalizer = RunningMeanStd(obs_dim)

    if args.checkpoint and not args.zero_action:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        print(f"Checkpoint: {args.checkpoint}")

        policy_state = ckpt["policy"]
        hidden_dims = []
        i = 0
        while f"backbone.{i * 2}.weight" in policy_state:
            hidden_dims.append(policy_state[f"backbone.{i * 2}.weight"].shape[0])
            i += 1
        if not hidden_dims:
            hidden_dims = [512, 256, 128]

        policy = ActorCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dims=tuple(hidden_dims),
        ).to(device)
        policy.load_state_dict(policy_state)
        policy.eval()

        if "obs_normalizer" in ckpt:
            obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    else:
        print("MODE: Zero-action (PD tracking of reference)")

    # ── Build FK model for reference overlay ─────────────────
    ref_model = None
    if args.show_ref:
        from prepare3.xml_builder import get_or_create_xml
        betas = np.load(args.betas)
        xml_path = get_or_create_xml(betas)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        ref_model = builder.finalize(device=device)

    # ── Run rollout and collect body positions ───────────────
    print("\nRunning rollout...")
    obs, _ = env.reset(seed=0)

    sim_positions = []   # list of (24, 3) arrays
    ref_positions = []   # list of (24, 3) arrays (if show_ref)
    rewards = []
    max_frames = int(ref_duration * args.control_freq) + 10

    for step_i in range(max_frames):
        # Get sim body positions
        body_q = env.state_0.body_q.numpy().reshape(-1, 7)
        sim_pos = body_q[:24, :3].copy()
        sim_positions.append(sim_pos)

        # Get reference body positions
        if args.show_ref and ref_model is not None:
            ref_jq = env._get_ref_joint_q(env.motion_time)
            ref_pos = _get_body_positions_from_joint_q(
                ref_model, ref_jq, device=device
            )
            ref_positions.append(ref_pos[:24].copy())

        # Get action
        if policy is not None:
            obs_norm = obs_normalizer.normalize(obs)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_norm, deterministic=True)
        else:
            action = np.zeros(action_dim, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            print(f"  Episode ended at step {step_i + 1}: "
                  f"reward_sum={sum(rewards):.3f}, "
                  f"survival={min(env.time / ref_duration, 1.0):.0%}")
            break

    sim_positions = np.array(sim_positions)  # (T, 24, 3)
    if ref_positions:
        ref_positions = np.array(ref_positions)  # (T, 24, 3)

    n_frames = len(sim_positions)
    print(f"Collected {n_frames} frames")

    # ── Create MP4 animation ─────────────────────────────────
    mp4_path = args.mp4
    if not mp4_path:
        mp4_path = os.path.join(
            os.path.dirname(args.checkpoint or args.motion),
            "rollout.mp4"
        )
    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)

    print(f"Rendering MP4: {mp4_path}")

    # Compute axis limits from all positions
    all_pos = sim_positions.reshape(-1, 3)
    if ref_positions is not None and len(ref_positions) > 0:
        all_pos = np.concatenate([all_pos, ref_positions.reshape(-1, 3)], axis=0)

    center = all_pos.mean(axis=0)
    span = max(all_pos.max(axis=0) - all_pos.min(axis=0)) * 0.6
    span = max(span, 1.5)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw_skeleton(ax, positions, bones, color="blue", alpha=1.0, lw=2.5,
                      joint_size=25, label=None):
        """Draw a skeleton on a 3D axis."""
        xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
        ax.scatter(xs, ys, zs, c=color, s=joint_size, alpha=alpha,
                   depthshade=True, label=label)
        for (i, j) in bones:
            if i < len(positions) and j < len(positions):
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    [positions[i, 2], positions[j, 2]],
                    color=color, alpha=alpha, linewidth=lw,
                )

    def update(frame_idx):
        ax.cla()

        # Draw sim skeleton
        draw_skeleton(ax, sim_positions[frame_idx], SKELETON_BONES,
                      color="dodgerblue", alpha=1.0, lw=3, joint_size=30,
                      label="Sim (policy)")

        # Draw reference ghost
        if ref_positions is not None and len(ref_positions) > frame_idx:
            draw_skeleton(ax, ref_positions[frame_idx], SKELETON_BONES,
                          color="red", alpha=0.4, lw=1.5, joint_size=15,
                          label="Reference")

        # Draw ground plane
        gx = np.linspace(center[0] - span, center[0] + span, 2)
        gy = np.linspace(center[1] - span, center[1] + span, 2)
        gx, gy = np.meshgrid(gx, gy)
        gz = np.zeros_like(gx)
        ax.plot_surface(gx, gy, gz, alpha=0.1, color="gray")

        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(0, span * 2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        t = frame_idx / args.control_freq
        r = rewards[frame_idx] if frame_idx < len(rewards) else 0.0
        mode = "Policy" if policy else "Zero-action"
        title = f"{mode} | t={t:.2f}s | frame {frame_idx}/{n_frames} | r={r:.3f}"
        ax.set_title(title, fontsize=12)

        ax.view_init(elev=20, azim=-60 + frame_idx * 0.3)

        if ref_positions is not None and len(ref_positions) > 0:
            ax.legend(loc="upper right", fontsize=9)

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // 30, blit=False)
    writer = FFMpegWriter(fps=30, bitrate=3000)
    anim.save(mp4_path, writer=writer)
    plt.close(fig)

    print(f"\nSaved: {mp4_path}")
    print(f"Duration: {n_frames / args.control_freq:.2f}s, {n_frames} frames")
    print(f"Total reward: {sum(rewards):.3f}")

    env.close()
    return mp4_path


# ═══════════════════════════════════════════════════════════════
# Interactive GL Viewer
# ═══════════════════════════════════════════════════════════════

class PolicyVisualizer:
    """Newton viewer that runs a trained policy in the physics simulator."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = args.device or "cuda:0"
        self.show_ref = args.show_ref
        self.zero_action = args.zero_action
        self.sim_time = 0.0
        self._wall_start = None
        self._paused = False
        self._step_by_step = False

        # ── Build environment ────────────────────────────────
        env_config = {
            "motion_file": args.motion,
            "betas_file": args.betas,
            "device": self.device,
            "sim_freq": args.sim_freq,
            "control_freq": args.control_freq,
            "control_mode": "pd",
            "max_episode_length": 999.0,  # let it run long
            "enable_early_termination": not args.no_termination,
            "rand_init": False,  # always start from t=0
            "enable_tar_obs": True,
        }
        self.env = NewtonMimicEnv(env_config)
        self.obs_dim = self.env.get_obs_dim()
        self.action_dim = self.env.get_action_dim()
        self.ref_T = self.env.ref_T
        self.ref_fps = self.env.ref_fps
        self.ref_duration = self.env.ref_duration

        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim:      {self.action_dim}")
        print(f"Ref motion:      {self.ref_T} frames, {self.ref_duration:.2f}s")

        # ── Load policy (optional) ───────────────────────────
        self.policy = None
        self.obs_normalizer = RunningMeanStd(self.obs_dim)

        if args.checkpoint and not self.zero_action:
            ckpt = torch.load(args.checkpoint, map_location=self.device,
                              weights_only=False)
            print(f"Checkpoint: {args.checkpoint}")

            # Detect hidden dims from checkpoint
            policy_state = ckpt["policy"]
            hidden_dims = []
            i = 0
            while f"backbone.{i * 2}.weight" in policy_state:
                hidden_dims.append(policy_state[f"backbone.{i * 2}.weight"].shape[0])
                i += 1
            if not hidden_dims:
                hidden_dims = [512, 256, 128]

            print(f"Hidden dims: {hidden_dims}")

            self.policy = ActorCritic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                hidden_dims=tuple(hidden_dims),
            ).to(self.device)
            self.policy.load_state_dict(policy_state)
            self.policy.eval()

            if "obs_normalizer" in ckpt:
                self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

            if "iteration" in ckpt:
                print(f"  Iteration: {ckpt['iteration']}")
            if "best_mean_reward" in ckpt:
                print(f"  Best reward: {ckpt['best_mean_reward']:.4f}")
        elif self.zero_action:
            print("MODE: Zero-action (PD tracking of reference)")
        else:
            print("WARNING: No checkpoint provided, using zero actions")
            self.zero_action = True

        # ── Reference ghost model (optional) ─────────────────
        self.ref_model = None
        self.ref_state = None
        if self.show_ref:
            self._build_ref_model(args)

        # ── Set up primary viewer model ──────────────────────
        if self.show_ref:
            # Combined model: physics person + reference ghost
            self._build_combined_viewer_model(args)
        else:
            # Just the physics model
            self.viewer_model = self.env.model
            self.viewer_state = self.env.state_0

        self.viewer.set_model(self.viewer_model)

        # ── Reset environment ────────────────────────────────
        self.obs, _ = self.env.reset(seed=0)
        self.done = False
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_count = 0
        self._last_reward = 0.0

        # ── Camera ───────────────────────────────────────────
        self._setup_camera()

        mode = "Policy" if self.policy else "Zero-action"
        print(f"\nReady! Mode: {mode}, Ref: {self.ref_T} frames "
              f"({self.ref_duration:.1f}s). Close viewer to exit.")

    def _build_ref_model(self, args):
        """Build a separate Newton model for the reference ghost."""
        from prepare3.xml_builder import get_or_create_xml
        betas = np.load(args.betas)
        xml_path = get_or_create_xml(betas)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.ref_model = builder.finalize(device=self.device)
        self.ref_state = self.ref_model.state()
        self._ref_jqd = wp.zeros(DOFS_PER_PERSON, dtype=wp.float32,
                                 device=self.device)

    def _build_combined_viewer_model(self, args):
        """Build a viewer model containing both physics person + ref ghost."""
        from prepare3.xml_builder import get_or_create_xml
        betas = np.load(args.betas)
        xml_path = get_or_create_xml(betas)

        # Build combined model: two copies of the same skeleton
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_mjcf(xml_path, enable_self_collisions=False)
        builder.add_ground_plane()
        self.viewer_model = builder.finalize(device=self.device)
        self.viewer_state = self.viewer_model.state()
        self._viewer_jqd = wp.zeros(
            self.viewer_model.joint_dof_count, dtype=wp.float32,
            device=self.device
        )

    def _setup_camera(self):
        """Position camera to look at the character."""
        root_pos = self.env.state_0.joint_q.numpy()[:3]
        cam_dist = 5.0
        cam_pos = wp.vec3(
            float(root_pos[0]),
            float(root_pos[1]) - cam_dist,
            2.0,
        )
        self.viewer.set_camera(cam_pos, pitch=-15.0, yaw=90.0)

    def _update_viewer_state(self):
        """Update the viewer state with current sim + reference poses."""
        if self.show_ref:
            # Combined state: [sim_person | ref_person]
            sim_q = self.env.state_0.joint_q.numpy()  # (76,)
            ref_q = self.env._get_ref_joint_q(self.env.motion_time)  # (76,)

            combined_q = np.zeros(COORDS_PER_PERSON * 2, dtype=np.float32)
            combined_q[:COORDS_PER_PERSON] = sim_q
            combined_q[COORDS_PER_PERSON:] = ref_q

            self.viewer_state.joint_q = wp.array(
                combined_q, dtype=wp.float32, device=self.device
            )
            self._viewer_jqd.zero_()
            newton.eval_fk(self.viewer_model, self.viewer_state.joint_q,
                           self._viewer_jqd, self.viewer_state)
        else:
            self.viewer_state = self.env.state_0

    def step(self):
        """Called each frame by newton.examples.run."""
        if self._paused:
            return

        if self.done:
            # Auto-restart episode
            self.obs, _ = self.env.reset(seed=self.ep_count)
            self.done = False
            self.ep_reward = 0.0
            self.ep_length = 0
            self.ep_count += 1

        # ── Get action ───────────────────────────────────────
        if self.policy is not None:
            obs_norm = self.obs_normalizer.normalize(self.obs)
            with torch.no_grad():
                action, _, _ = self.policy.get_action(obs_norm, deterministic=True)
        else:
            action = np.zeros(self.action_dim, dtype=np.float32)

        # ── Step environment ─────────────────────────────────
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.ep_reward += reward
        self.ep_length += 1
        self._last_reward = reward

        if self.done:
            print(f"  Episode {self.ep_count}: steps={self.ep_length}, "
                  f"reward={self.ep_reward:.3f}, "
                  f"survival={min(self.env.time / self.ref_duration, 1.0):.0%}")

        # Track wall clock for replay timing
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = now
        self.sim_time = now - self._wall_start

        # Update the viewer state
        self._update_viewer_state()

    def render(self):
        """Called each frame by newton.examples.run."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viewer_state)
        self.viewer.end_frame()

    def gui(self, imgui):
        """Side panel displayed in the Newton GL viewer."""
        imgui.separator()
        if self.policy is not None:
            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0),
                               "[ POLICY ROLLOUT ]")
        else:
            imgui.text_colored(imgui.ImVec4(1.0, 0.85, 0.0, 1.0),
                               "[ ZERO-ACTION (PD ref track) ]")

        imgui.separator()
        imgui.text(f"Episode:   {self.ep_count}")
        imgui.text(f"Step:      {self.ep_length}")

        # Time info
        sim_t = self.env.time
        ref_t = self.env.motion_time
        imgui.text(f"Sim time:  {sim_t:.2f}s")
        imgui.text(f"Ref time:  {ref_t:.2f}s / {self.ref_duration:.2f}s")

        pct = min(ref_t / self.ref_duration, 1.0) * 100
        imgui.text(f"Progress:  {pct:.0f}%")

        imgui.separator()
        imgui.text(f"Reward:    {self._last_reward:.4f}")
        imgui.text(f"Ep reward: {self.ep_reward:.3f}")

        # Root height
        root_z = self.env.state_0.joint_q.numpy()[2]
        imgui.text(f"Root h:    {root_z:.3f}m")

        imgui.separator()
        if self.show_ref:
            imgui.text_colored(imgui.ImVec4(0.5, 0.8, 1.0, 1.0),
                               "Sim=left, Ref=right")
        imgui.text(f"Ref:  {self.ref_T} frames @ {self.ref_fps} fps")
        imgui.text(f"Ctrl: {self.env.control_freq} Hz")
        imgui.text(f"Sim:  {self.env.sim_freq} Hz")


def build_parser():
    parser = newton.examples.create_parser()

    # Required
    parser.add_argument("--motion", required=True,
                        help="Path to .pkl motion file (from convert_to_mimickit)")
    parser.add_argument("--betas", required=True,
                        help="Path to .npy betas file")

    # Policy
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained policy checkpoint (.pt)")
    parser.add_argument("--zero-action", action="store_true",
                        help="Use zero actions (PD tracks reference only)")

    # Visualization
    parser.add_argument("--show-ref", action="store_true",
                        help="Show reference motion ghost alongside sim")
    parser.add_argument("--no-termination", action="store_true",
                        help="Disable early termination (let character fall)")

    # Headless / MP4
    parser.add_argument("--mp4", default=None,
                        help="Path for output MP4 file (implies --headless)")

    # Environment
    parser.add_argument("--sim-freq", type=int, default=480)
    parser.add_argument("--control-freq", type=int, default=30)

    return parser


if __name__ == "__main__":
    parser = build_parser()

    # Peek at args to decide headless vs interactive
    known, _ = parser.parse_known_args()

    is_headless = getattr(known, "headless", False) or known.mp4

    if is_headless:
        # ── Headless MP4 mode ─────────────────────────────────
        args = parser.parse_args()
        if not hasattr(args, "device") or args.device is None:
            args.device = "cuda:0"
        render_headless_mp4(args)
    else:
        # ── Interactive GL viewer ─────────────────────────────
        viewer, args = newton.examples.init(parser)
        if not hasattr(args, "device") or args.device is None:
            args.device = "cuda:0"
        example = PolicyVisualizer(viewer, args)
        newton.examples.run(example, args)
