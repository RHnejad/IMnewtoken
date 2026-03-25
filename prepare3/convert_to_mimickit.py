"""
convert_to_mimickit.py — Convert retargeted Newton arrays to MimicKit motion format.

Reads prepare2's retargeted output (joint_q.npy + betas.npy) and packages
them into MimicKit's .pkl motion format for RL training.

MimicKit motion format:
    {
        "loop_mode": int (0=CLAMP, 1=WRAP),
        "fps": int,
        "frames": list of (3 + 3 + dof_size) arrays per frame
                  [root_pos(3), root_rot_expmap(3), hinge_dofs(69)]
    }

Newton joint_q format: (T, 76)
    [root_pos(3), root_quat_xyzw(4), hinge_angles(69)]

The conversion:
    1. root_pos    = joint_q[:, 0:3]
    2. root_rot    = quat_to_exp_map(joint_q[:, 3:7])
    3. dof_pos     = joint_q[:, 7:76]
    4. frames      = concat([root_pos, root_rot, dof_pos], axis=1)  → (T, 75)

Usage:
    # Convert a single clip for both persons
    python prepare3/convert_to_mimickit.py \\
        --clip 1000 \\
        --data-dir data/retargeted_v2/interhuman \\
        --output-dir data/mimickit_motions/interhuman

    # Convert entire dataset
    python prepare3/convert_to_mimickit.py \\
        --dataset interhuman \\
        --output-dir data/mimickit_motions/interhuman
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════
# Quaternion → Exponential Map Conversion (numpy, no torch needed)
# ═══════════════════════════════════════════════════════════════
def _quat_to_exp_map_np(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to axis-angle / exponential map.

    Args:
        q: (..., 4) quaternion array in [x, y, z, w] format.

    Returns:
        exp_map: (..., 3) exponential map (axis * angle).
    """
    # Ensure positive w hemisphere
    q = q.copy()
    neg_w = q[..., 3:4] < 0
    q = np.where(neg_w, -q, q)

    # Normalize
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / (norm + 1e-12)

    # Extract axis and angle
    xyz = q[..., :3]
    w = q[..., 3:4]

    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    cos_half = w

    # angle = 2 * atan2(sin_half, cos_half)
    half_angle = np.arctan2(sin_half, cos_half)
    angle = 2.0 * half_angle

    # axis = xyz / sin_half  (handle near-zero case)
    axis = np.where(
        sin_half > 1e-8,
        xyz / sin_half,
        np.zeros_like(xyz)
    )

    return (axis * angle).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Single Clip Conversion
# ═══════════════════════════════════════════════════════════════
def convert_clip(joint_q: np.ndarray, fps: int = 30,
                 loop_mode: str = "clamp",
                 downsample: int = 1) -> dict:
    """Convert a Newton joint_q array to MimicKit motion dict.

    Args:
        joint_q: (T, 76) Newton joint coordinates.
        fps: target FPS for the motion.
        loop_mode: "clamp" or "wrap".
        downsample: downsample factor (e.g., 2 = 60fps → 30fps).

    Returns:
        motion_dict: {"loop_mode": int, "fps": int, "frames": list}
    """
    if downsample > 1:
        joint_q = joint_q[::downsample]

    T = joint_q.shape[0]
    assert joint_q.shape[1] == 76, \
        f"Expected 76 coords, got {joint_q.shape[1]}"

    # Extract components
    root_pos = joint_q[:, 0:3].astype(np.float32)          # (T, 3)
    root_quat = joint_q[:, 3:7].astype(np.float32)         # (T, 4) xyzw
    hinge_dofs = joint_q[:, 7:76].astype(np.float32)       # (T, 69)

    # Convert root quaternion → exponential map
    root_expmap = _quat_to_exp_map_np(root_quat)           # (T, 3)

    # Concatenate into MimicKit frame format
    frames = np.concatenate([root_pos, root_expmap, hinge_dofs], axis=1)
    assert frames.shape == (T, 75), \
        f"Expected (T, 75), got {frames.shape}"

    loop_mode_int = 1 if loop_mode == "wrap" else 0

    return {
        "loop_mode": loop_mode_int,
        "fps": fps,
        "frames": frames.tolist(),
    }


def save_motion(motion_dict: dict, output_path: str):
    """Save motion dict as a MimicKit-compatible pickle file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(motion_dict, f)


# ═══════════════════════════════════════════════════════════════
# Motion File List (YAML format for MimicKit)
# ═══════════════════════════════════════════════════════════════
def generate_motion_list(motion_files: list, output_path: str):
    """Generate a MimicKit-compatible motion file list (YAML).

    MimicKit's MotionLib expects a YAML file listing motion paths and weights.

    Args:
        motion_files: list of .pkl motion file paths (relative or absolute).
        output_path: where to write the YAML file.
    """
    import yaml

    entries = []
    for path in motion_files:
        entries.append({
            "file": path,
            "weight": 1.0,
        })

    motion_list = {"motions": entries}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(motion_list, f, default_flow_style=False)

    print(f"Motion list: {output_path} ({len(entries)} motions)")


# ═══════════════════════════════════════════════════════════════
# Batch Conversion
# ═══════════════════════════════════════════════════════════════
def convert_all_clips(data_dir: str, output_dir: str, fps: int = 30,
                      downsample: int = 2, loop_mode: str = "clamp",
                      clip_ids: list = None):
    """Convert all retargeted clips in a directory.

    Args:
        data_dir: directory containing *_joint_q.npy files.
        output_dir: where to save .pkl motion files.
        fps: target FPS.
        downsample: downsample factor.
        loop_mode: "clamp" or "wrap".
        clip_ids: optional list of specific clip IDs to convert.

    Returns:
        output_files: list of generated .pkl paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # Find all joint_q files
    jq_pattern = os.path.join(data_dir, "*_joint_q.npy")
    jq_files = sorted(glob.glob(jq_pattern))

    if not jq_files:
        print(f"No *_joint_q.npy files found in {data_dir}")
        return output_files

    for jq_path in jq_files:
        basename = os.path.basename(jq_path)
        # Parse clip_id and person_idx from "1000_person0_joint_q.npy"
        parts = basename.replace("_joint_q.npy", "").rsplit("_person", 1)
        if len(parts) != 2:
            print(f"  Skipping {basename} (unexpected format)")
            continue

        clip_id = parts[0]
        person_idx = parts[1]

        if clip_ids is not None and clip_id not in clip_ids:
            continue

        # Load
        joint_q = np.load(jq_path)
        print(f"  Converting clip {clip_id} person {person_idx}: "
              f"{joint_q.shape[0]} frames → ", end="")

        # Convert
        motion_dict = convert_clip(joint_q, fps=fps,
                                   loop_mode=loop_mode,
                                   downsample=downsample)

        # Save
        out_name = f"{clip_id}_person{person_idx}.pkl"
        out_path = os.path.join(output_dir, out_name)
        save_motion(motion_dict, out_path)
        output_files.append(out_path)

        T = len(motion_dict["frames"])
        print(f"{T} frames @ {fps}fps → {out_name}")

    # Generate motion list YAML
    if output_files:
        list_path = os.path.join(output_dir, "motion_list.yaml")
        generate_motion_list(output_files, list_path)

    return output_files


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Convert retargeted Newton joint_q arrays to MimicKit motion format"
    )
    parser.add_argument("--clip", type=str, default=None,
                        help="Single clip ID to convert")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., 'interhuman')")
    parser.add_argument("--data-dir", type=str,
                        default="data/retargeted_v2/interhuman",
                        help="Directory with retargeted joint_q files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .pkl files "
                             "(default: data/mimickit_motions/{dataset})")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (default: 30)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample factor (default: 2 → 60fps→30fps)")
    parser.add_argument("--loop", choices=["clamp", "wrap"],
                        default="clamp",
                        help="Loop mode (default: clamp)")
    args = parser.parse_args()

    # Resolve paths
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    if not os.path.isdir(data_dir):
        data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    if args.output_dir:
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    elif args.dataset:
        output_dir = os.path.join(
            PROJECT_ROOT, "data", "mimickit_motions", args.dataset
        )
    else:
        output_dir = os.path.join(
            PROJECT_ROOT, "data", "mimickit_motions", "default"
        )

    clip_ids = [args.clip] if args.clip else None

    print(f"Converting motions:")
    print(f"  Source:  {data_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  FPS:     {args.fps}, downsample: {args.downsample}x")
    print(f"  Loop:    {args.loop}")
    print()

    results = convert_all_clips(
        data_dir=data_dir,
        output_dir=output_dir,
        fps=args.fps,
        downsample=args.downsample,
        loop_mode=args.loop,
        clip_ids=clip_ids,
    )

    print(f"\nDone: {len(results)} motions converted")


if __name__ == "__main__":
    main()
