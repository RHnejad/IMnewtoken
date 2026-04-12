#!/usr/bin/env python3
"""Interactive Newton GUI Visualizer for .motion files.

This script launches the Newton ViewerGL interactive GUI (not headless)
for visualizing .motion files with full 3D controls.

Usage:
    cd /media/rh/codes/sim/IMnewtoken
    
    # Visualize a single motion file
    python prepare7/visualize_motions.py \
        --motion-file prepare7/data/interhuman_test/1_person1.motion
    
    # Visualize multiple motions side-by-side
    python prepare7/visualize_motions.py \
        --motion-file prepare7/data/interhuman_test/1_person1.motion \
        --motion-file prepare7/data/interhuman_test/1_person2.motion

Controls:
    - Left click + drag: Rotate camera
    - Right click + drag: Pan camera
    - Scroll wheel: Zoom
    - Space: Pause/Resume
    - R: Reset camera
    - Q/Esc: Quit
"""

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
logger = logging.getLogger("visualize_motions")

# SMPL skeleton connectivity (24 joints)
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


def visualize_interactive(motion_paths, fps=30, loop=True):
    """Launch Newton ViewerGL GUI for interactive visualization."""
    import warp as wp
    import newton

    logger.info("=" * 60)
    logger.info("Newton Interactive Visualizer")
    logger.info("=" * 60)
    
    # Load all motions
    motions_data = []
    for mp in motion_paths:
        logger.info(f"Loading: {mp}")
        motion, positions = load_motion(mp)
        T, num_bodies, _ = positions.shape
        logger.info(f"  {T} frames, {num_bodies} bodies")
        motions_data.append({
            "path": mp,
            "motion": motion,
            "positions": positions,
            "T": T,
            "num_bodies": num_bodies,
            "frame": 0,
        })

    # Create interactive viewer (headless=False for GUI)
    logger.info("\n🎮 Launching Newton ViewerGL (Interactive)")
    logger.info("Controls:")
    logger.info("  - Left click + drag: Rotate camera")
    logger.info("  - Right click + drag: Pan camera")
    logger.info("  - Scroll wheel: Zoom")
    logger.info("  - Space: Pause/Resume")
    logger.info("  - R: Reset camera")
    logger.info("  - Q/Esc: Quit")
    logger.info("=" * 60)
    
    viewer = newton.viewer.ViewerGL(width=1920, height=1080, headless=False)

    # Compute initial camera position to look at skeletons
    all_positions = np.concatenate([m["positions"] for m in motions_data], axis=1)
    centre = np.mean(all_positions[0], axis=0)  # center of all skeletons at frame 0
    
    # Position camera in front and slightly above to get good view
    cam_distance = 4.0
    cam_pos = wp.vec3(
        float(centre[0]), 
        float(centre[1]) - cam_distance,  # in front (negative Y)
        float(centre[2]) + 1.2              # slightly above
    )
    
    # Look at the center point with appropriate pitch
    viewer.set_camera(cam_pos, pitch=10.0, yaw=0.0)
    
    # Set camera target to the center of the skeleton
    viewer.camera_target = wp.vec3(float(centre[0]), float(centre[1]), float(centre[2]) + 1.0)

    dt = 1.0 / fps
    paused = False
    frame_time = 0.0

    try:
        import time
        last_time = time.time()
        
        while viewer.is_running():
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time

            if not paused:
                frame_time += elapsed

            # Begin frame once
            viewer.begin_frame(frame_time)

            # Update frame for each motion and draw
            for i, m in enumerate(motions_data):
                if not paused:
                    m["frame"] = int((frame_time * fps) % m["T"])
                    if loop and frame_time * fps >= m["T"]:
                        frame_time = 0.0

                t = m["frame"]
                pos = m["positions"][t]
                num_bodies = m["num_bodies"]

                # Offset for side-by-side display
                x_offset = i * 2.5  # 2.5 meters apart

                # Build capsule arrays for skeleton
                xforms_list, scales_list, colors_list = [], [], []
                for bi, (j1, j2) in enumerate(SKELETON_BONES):
                    if j1 >= num_bodies or j2 >= num_bodies:
                        continue
                    p1 = pos[j1] + np.array([x_offset, 0, 0])
                    p2 = pos[j2] + np.array([x_offset, 0, 0])
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
                    radius = 0.045 if bi < 5 else 0.035
                    scales_list.append(wp.vec3(radius, radius, bone_len * 0.5))
                    r, g, b = _BONE_RGB[bi]
                    colors_list.append(wp.vec3(r, g, b))

                if xforms_list:
                    xforms_wp = wp.array(xforms_list, dtype=wp.transform, device="cuda")
                    scales_wp = wp.array(scales_list, dtype=wp.vec3, device="cuda")
                    colors_wp = wp.array(colors_list, dtype=wp.vec3, device="cuda")
                    
                    viewer.log_capsules(f"skeleton_{i}", None, xforms_wp, scales_wp, colors_wp, None)

            # End frame once
            viewer.end_frame()

            # Small sleep to limit frame rate
            time.sleep(max(0, dt - elapsed))

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        viewer.close()
        logger.info("Viewer closed")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Newton GUI Visualizer for .motion files"
    )
    parser.add_argument(
        "--motion-file",
        type=Path,
        action="append",
        required=True,
        help="Path to .motion file (can specify multiple times for side-by-side view)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--no-loop", action="store_true", help="Don't loop playback")
    args = parser.parse_args()

    # Validate motion files
    for mf in args.motion_file:
        if not mf.exists():
            logger.error(f"Motion file not found: {mf}")
            sys.exit(1)

    visualize_interactive(args.motion_file, fps=args.fps, loop=not args.no_loop)


if __name__ == "__main__":
    main()
