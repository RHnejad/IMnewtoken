#!/usr/bin/env python3
"""Record MP4 videos from .motion files — three rendering backends.

Backends:
  1. "newton" (default) — Newton ViewerGL headless (EGL). GPU-accelerated,
     proper 3D lighting, SMPL capsule skeleton. Needs: conda install libegl libglvnd
  2. "skeleton" — matplotlib 3D stick-figure. Zero GL deps, works everywhere.
  3. "mujoco" — MuJoCo offscreen EGL. Full SMPL humanoid mesh. Needs EGL.

Usage:
    cd /media/rh/codes/sim/IMnewtoken

    # Newton renderer (default — GPU, proper lighting):
    python prepare7/record_video.py \
        --motion-dir prepare7/data/interhuman_test \
        --output-dir prepare7/output/videos \
        --num-motions 3

    # Matplotlib skeleton (fallback, no display needed):
    python prepare7/record_video.py \
        --motion-dir prepare7/data/interhuman_test \
        --output-dir prepare7/output/videos \
        --renderer skeleton --num-motions 3

    # MuJoCo mesh renderer:
    python prepare7/record_video.py \
        --motion-dir prepare7/data/interhuman_test \
        --output-dir prepare7/output/videos \
        --renderer mujoco --num-motions 3
"""

import os
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROTO_ROOT = Path(__file__).resolve().parent / "ProtoMotions"
sys.path.insert(0, str(PROTO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("record_video")

# ---------------------------------------------------------------------------
# SMPL skeleton connectivity (SMPL_MUJOCO_NAMES order, 24 joints)
# 0:Pelvis 1:L_Hip 2:L_Knee 3:L_Ankle 4:L_Toe 5:R_Hip 6:R_Knee 7:R_Ankle
# 8:R_Toe 9:Torso 10:Spine 11:Chest 12:Neck 13:Head 14:L_Thorax
# 15:L_Shoulder 16:L_Elbow 17:L_Wrist 18:L_Hand 19:R_Thorax 20:R_Shoulder
# 21:R_Elbow 22:R_Wrist 23:R_Hand
# ---------------------------------------------------------------------------

SKELETON_BONES = [
    # Spine
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    # Left leg
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Right leg
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Left arm
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    # Right arm
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23),
]

# RGB colors per bone group
_BONE_RGB = [
    (0.29, 0.56, 0.85),  # spine    — blue
    (0.29, 0.56, 0.85),
    (0.29, 0.56, 0.85),
    (0.29, 0.56, 0.85),
    (0.29, 0.56, 0.85),
    (0.91, 0.30, 0.24),  # left leg — red
    (0.91, 0.30, 0.24),
    (0.91, 0.30, 0.24),
    (0.91, 0.30, 0.24),
    (0.18, 0.80, 0.44),  # right leg — green
    (0.18, 0.80, 0.44),
    (0.18, 0.80, 0.44),
    (0.18, 0.80, 0.44),
    (0.90, 0.49, 0.13),  # left arm — orange
    (0.90, 0.49, 0.13),
    (0.90, 0.49, 0.13),
    (0.90, 0.49, 0.13),
    (0.90, 0.49, 0.13),
    (0.61, 0.35, 0.71),  # right arm — purple
    (0.61, 0.35, 0.71),
    (0.61, 0.35, 0.71),
    (0.61, 0.35, 0.71),
    (0.61, 0.35, 0.71),
]


def load_motion(motion_path):
    """Load a .motion file, return (motion, positions_np)."""
    from protomotions.simulator.base_simulator.simulator_state import (
        RobotState,
        StateConversion,
    )
    data = torch.load(motion_path, map_location="cpu", weights_only=False)
    motion = RobotState.from_dict(data, state_conversion=StateConversion.COMMON)
    positions = motion.rigid_body_pos.numpy()  # (T, num_bodies, 3)
    return motion, positions


def _setup_egl():
    """Set EGL env vars (idempotent). Call before importing mujoco/newton's GL layer."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        lib_dir = os.path.join(conda_prefix, "lib")
        if os.path.isdir(lib_dir):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in ld:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld}" if ld else lib_dir
        egl_json = os.path.join(
            conda_prefix, "share", "glvnd", "egl_vendor.d", "10_nvidia.json"
        )
        if os.path.exists(egl_json):
            os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", egl_json)
            logger.info(f"  EGL vendor JSON: {egl_json}")


# ---------------------------------------------------------------------------
# Backend 1: Newton ViewerGL headless (GPU, proper 3D lighting)
# ---------------------------------------------------------------------------

def _quat_from_z_to_vec(target_dir):
    """Quaternion (xyzw) rotating Z-axis to target_dir."""
    z = np.array([0.0, 0.0, 1.0])
    target = np.array(target_dir, dtype=float)
    norm = np.linalg.norm(target)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    target /= norm

    dot = np.clip(np.dot(z, target), -1.0, 1.0)
    if dot > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.9999:
        # 180° rotation around X
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = np.cross(z, target)
    axis /= np.linalg.norm(axis)
    half = np.arccos(dot) / 2.0
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


def render_newton_video(motion_path, output_path, fps=30, width=1280, height=720,
                        subsample=1):
    """Render .motion to MP4 via Newton ViewerGL headless (EGL, GPU-accelerated)."""
    # Must set headless BEFORE pyglet.graphics or pyglet.window is imported.
    # Newton's RendererGL does a lazy `from pyglet.graphics.shader import ...` inside
    # __init__, which triggers pyglet's module-level shadow-window creation via X11.
    # Setting options['headless'] = True first makes pyglet use EGL instead.
    import pyglet
    pyglet.options['headless'] = True

    import warp as wp
    import newton

    _, positions = load_motion(motion_path)
    T, num_bodies, _ = positions.shape

    frame_indices = list(range(0, T, max(1, subsample)))
    render_fps = fps // max(1, subsample)
    logger.info(f"  {T} frames → rendering {len(frame_indices)} @ {render_fps}fps")

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)

    # Compute scene centre for camera placement
    centre = positions[T // 2, 0]  # pelvis at mid-sequence
    cam_pos = wp.vec3(float(centre[0]), float(centre[1]) - 3.0, float(centre[2]) + 1.5)
    viewer.set_camera(cam_pos, pitch=10.0, yaw=0.0)

    frames = []

    for fi, t in enumerate(frame_indices):
        pos = positions[t]  # (num_bodies, 3)

        # Build capsule arrays for all bones
        xforms_list, scales_list, colors_list = [], [], []
        for bi, (j1, j2) in enumerate(SKELETON_BONES):
            if j1 >= num_bodies or j2 >= num_bodies:
                continue
            p1 = pos[j1]
            p2 = pos[j2]
            mid = (p1 + p2) * 0.5
            bone_vec = p2 - p1
            bone_len = float(np.linalg.norm(bone_vec))
            if bone_len < 1e-6:
                continue

            q = _quat_from_z_to_vec(bone_vec)
            xforms_list.append(wp.transform(
                wp.vec3(float(mid[0]), float(mid[1]), float(mid[2])),
                wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            ))
            radius = 0.04 if bi < 5 else 0.03  # slightly thicker spine
            scales_list.append(wp.vec3(radius, radius, bone_len * 0.5))
            r, g, b = _BONE_RGB[bi]
            colors_list.append(wp.vec3(r, g, b))

        if not xforms_list:
            continue

        xforms_wp = wp.array(xforms_list, dtype=wp.transform, device="cuda")
        scales_wp = wp.array(scales_list, dtype=wp.vec3, device="cuda")
        colors_wp = wp.array(colors_list, dtype=wp.vec3, device="cuda")

        # Slowly orbit the camera
        angle = (t / T) * 360.0
        viewer.set_camera(cam_pos, pitch=15.0, yaw=float(angle))

        viewer.begin_frame(float(t) / fps)
        viewer.log_capsules("skeleton", None, xforms_wp, scales_wp, colors_wp, None)
        viewer.end_frame()

        frame = viewer.get_frame().numpy()  # [H, W, 3] uint8
        frames.append(frame.copy())

        if (fi + 1) % 50 == 0:
            logger.info(f"    rendered {fi + 1}/{len(frame_indices)} frames")

    viewer.close()
    _write_video(frames, output_path, fps=render_fps)


# ---------------------------------------------------------------------------
# Backend 2: Matplotlib skeleton renderer (no GL needed)
# ---------------------------------------------------------------------------

def render_skeleton_video(motion_path, output_path, fps=30, width=1280, height=720,
                          subsample=1, trail_frames=0):
    """Render .motion to MP4 using matplotlib 3D skeleton plots (no GL needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _, positions = load_motion(motion_path)
    T, num_bodies, _ = positions.shape

    frame_indices = list(range(0, T, max(1, subsample)))
    render_fps = fps // max(1, subsample)
    logger.info(f"  {T} frames → rendering {len(frame_indices)} @ {render_fps}fps")

    all_pos = positions[frame_indices]
    pad = 0.5
    x_min, x_max = all_pos[:, :, 0].min() - pad, all_pos[:, :, 0].max() + pad
    y_min, y_max = all_pos[:, :, 1].min() - pad, all_pos[:, :, 1].max() + pad
    max_range = max(x_max - x_min, y_max - y_min, all_pos[:, :, 2].max() - all_pos[:, :, 2].min() + pad) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (all_pos[:, :, 2].max() + all_pos[:, :, 2].min()) / 2.0

    dpi = 100
    frames = []
    _hex_colors = ["#4A90D9"] * 5 + ["#E74C3C"] * 4 + ["#2ECC71"] * 4 + ["#E67E22"] * 5 + ["#9B59B6"] * 5

    for fi, t in enumerate(frame_indices):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        pos = positions[t]

        for bi, (j1, j2) in enumerate(SKELETON_BONES):
            if j1 >= num_bodies or j2 >= num_bodies:
                continue
            ax.plot([pos[j1, 0], pos[j2, 0]], [pos[j1, 1], pos[j2, 1]],
                    [pos[j1, 2], pos[j2, 2]], color=_hex_colors[bi], linewidth=2.5)

        ax.scatter(pos[:num_bodies, 0], pos[:num_bodies, 1], pos[:num_bodies, 2],
                   c="black", s=12, depthshade=True)

        if trail_frames > 0:
            for dt in range(1, trail_frames + 1):
                t_prev = t - dt * max(1, subsample)
                if t_prev < 0:
                    break
                prev = positions[t_prev]
                alpha = 0.12 * (1.0 - dt / (trail_frames + 1))
                for j1, j2 in SKELETON_BONES:
                    if j1 < num_bodies and j2 < num_bodies:
                        ax.plot([prev[j1, 0], prev[j2, 0]], [prev[j1, 1], prev[j2, 1]],
                                [prev[j1, 2], prev[j2, 2]], color="gray",
                                linewidth=1.0, alpha=alpha)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(max(0, mid_z - max_range), mid_z + max_range)
        ax.view_init(elev=20, azim=135 + t * 0.3)
        ax.set_title(f"Frame {t}/{T}", fontsize=10)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        fig.tight_layout()

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

        if (fi + 1) % 50 == 0:
            logger.info(f"    rendered {fi + 1}/{len(frame_indices)} frames")

    _write_video(frames, output_path, fps=render_fps)


# ---------------------------------------------------------------------------
# Backend 3: MuJoCo offscreen renderer (full SMPL mesh, needs EGL)
# ---------------------------------------------------------------------------

def render_mujoco_video(motion_path, mjcf_path, output_path, fps=30, width=1280, height=720):
    """Render .motion to MP4 via MuJoCo offscreen EGL (full SMPL humanoid mesh)."""
    import mujoco

    motion, positions = load_motion(motion_path)
    T = positions.shape[0]
    logger.info(f"  {T} frames, {positions.shape[1]} bodies")

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.elevation = -20.0
    camera.azimuth = 135.0

    frames = []
    for t in range(T):
        root_pos = positions[t, 0]
        rot = motion.rigid_body_rot.numpy()
        rxyzw = rot[t, 0]
        rwxyz = np.array([rxyzw[3], rxyzw[0], rxyzw[1], rxyzw[2]])
        if model.nq >= 7:
            data.qpos[:3] = root_pos
            data.qpos[3:7] = rwxyz
        if motion.dof_pos is not None:
            dof = motion.dof_pos[t].numpy()
            n = min(len(dof), model.nq - 7)
            if n > 0:
                data.qpos[7:7 + n] = dof[:n]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = root_pos
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())

    renderer.close()
    _write_video(frames, output_path, fps=fps)


# ---------------------------------------------------------------------------
# Shared video writer
# ---------------------------------------------------------------------------

def _write_video(frames, output_path, fps=30):
    if not frames:
        logger.warning("  No frames to write")
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from moviepy import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(output_path), codec="libx264", audio=False,
                             threads=4, preset="veryfast",
                             ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)
        logger.info(f"  -> {output_path} ({len(frames)} frames, {fps}fps)")
    except Exception as e:
        logger.warning(f"  moviepy failed ({e}), falling back to ffmpeg pipe")
        _write_video_ffmpeg(frames, output_path, fps)


def _write_video_ffmpeg(frames, output_path, fps=30):
    import subprocess
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-an",
           str(output_path)]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0:
            logger.info(f"  -> {output_path} ({len(frames)} frames, ffmpeg)")
        else:
            logger.error(f"  ffmpeg failed: {proc.stderr.read().decode()[:300]}")
            _write_pngs(frames, output_path)
    except FileNotFoundError:
        _write_pngs(frames, output_path)


def _write_pngs(frames, output_path):
    import matplotlib.pyplot as plt
    png_dir = Path(output_path).parent / Path(output_path).stem
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        plt.imsave(str(png_dir / f"{i:04d}.png"), f)
    logger.info(f"  -> {len(frames)} PNGs in {png_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record MP4 videos from .motion files (headless)"
    )
    parser.add_argument("--motion-dir", required=True)
    parser.add_argument("--output-dir", default="prepare7/output/videos")
    parser.add_argument("--num-motions", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--renderer", choices=["newton", "skeleton", "mujoco"], default="skeleton",
        help="skeleton=matplotlib (default, always works), newton=Newton ViewerGL EGL, mujoco=MuJoCo EGL"
    )
    parser.add_argument("--subsample", type=int, default=1,
                        help="Render every Nth frame (newton/skeleton only)")
    parser.add_argument("--trail", type=int, default=3,
                        help="Ghost trail frames (skeleton only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_dir = Path(args.motion_dir)
    motion_files = sorted(motion_dir.glob("*.motion"))[:args.num_motions]
    if not motion_files:
        logger.error(f"No .motion files in {motion_dir}")
        sys.exit(1)

    logger.info(f"Rendering {len(motion_files)} motions from {motion_dir}")
    logger.info(f"  Backend:    {args.renderer}")
    logger.info(f"  Output:     {output_dir}")
    logger.info(f"  Resolution: {args.width}x{args.height} @ {args.fps}fps")

    mjcf_path = None
    if args.renderer in ("newton", "mujoco"):
        _setup_egl()
        if args.renderer == "mujoco":
            mjcf_path = PROTO_ROOT / "protomotions" / "data" / "assets" / "mjcf" / "smpl_humanoid.xml"
            if not mjcf_path.exists():
                logger.error(f"MJCF not found: {mjcf_path}")
                sys.exit(1)

    for i, mf in enumerate(motion_files):
        name = mf.stem
        logger.info(f"[{i + 1}/{len(motion_files)}] {name}")
        output_path = output_dir / f"{name}.mp4"
        try:
            if args.renderer == "newton":
                try:
                    render_newton_video(
                        mf, output_path,
                        fps=args.fps, width=args.width, height=args.height,
                        subsample=args.subsample,
                    )
                except Exception as e:
                    logger.warning(f"  Newton renderer failed ({type(e).__name__}: {e})")
                    logger.warning("  Newton ViewerGL requires CUDA-GL EGL interop.")
                    logger.warning("  Falling back to skeleton renderer.")
                    render_skeleton_video(
                        mf, output_path,
                        fps=args.fps, width=args.width, height=args.height,
                        subsample=args.subsample, trail_frames=args.trail,
                    )
            elif args.renderer == "mujoco":
                render_mujoco_video(
                    mf, mjcf_path, output_path,
                    fps=args.fps, width=args.width, height=args.height,
                )
            else:
                render_skeleton_video(
                    mf, output_path,
                    fps=args.fps, width=args.width, height=args.height,
                    subsample=args.subsample, trail_frames=args.trail,
                )
        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=True)

    logger.info(f"Done. Videos in {output_dir}")


if __name__ == "__main__":
    main()
