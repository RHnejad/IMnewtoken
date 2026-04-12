#!/usr/bin/env python3
"""Physics-based visualization using ProtoMotions simulator.

This uses the actual Newton physics simulator to track and visualize motions,
showing realistic physics-based humanoid movement (not just kinematic playback).

It reuses the same inference pipeline as evaluation (inference_agent.main()),
loading the pretrained SMPL motion tracker checkpoint and running the simulator
interactively.

Usage:
    cd /media/rh/codes/sim/IMnewtoken

    # Visualize single motion with physics:
    python prepare7/visualize_physics.py \
        --motion-file prepare7/data/interhuman_test/1_person1.motion

    # Visualize a directory of motions:
    python prepare7/visualize_physics.py \
        --motion-file prepare7/data/interhuman_test

    # Multiple motions (collected into a temp directory):
    python prepare7/visualize_physics.py \
        --motion-file prepare7/data/interhuman_test/1_person1.motion \
        --motion-file prepare7/data/interhuman_test/1_person2.motion
"""

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

# Add ProtoMotions to path BEFORE importing torch
PROTO_ROOT = Path(__file__).resolve().parent / "ProtoMotions"
sys.path.insert(0, str(PROTO_ROOT))

# Import simulator setup utilities
from protomotions.utils.simulator_imports import import_simulator_before_torch

# Parse args early (before torch import)
parser = argparse.ArgumentParser(description="Physics-based motion visualization")
parser.add_argument(
    "--motion-file",
    type=Path,
    action="append",
    required=True,
    help="Path to .motion file or directory (can specify multiple files)",
)
parser.add_argument(
    "--simulator",
    type=str,
    choices=["newton", "isaacgym", "isaaclab"],
    default="newton",
    help="Physics simulator to use (default: newton)",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run in headless mode (no GUI)",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=None,
    help="Number of parallel environments (default: auto based on motion count)",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(__file__).resolve().parent
    / "ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt",
    help="Path to pretrained tracker checkpoint",
)
args = parser.parse_args()

# Import simulator before torch
AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import torch and other modules
import logging
import torch

from lightning import Fabric
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)


def resolve_motion_path(motion_files):
    """Given a list of --motion-file arguments, return a single path for MotionLib.

    If a single directory is given, return it directly.
    If a single .motion file is given, return it directly.
    If multiple .motion files are given, symlink them into a temp directory
    and return that directory.
    """
    # Single argument that is a directory -> use directly
    if len(motion_files) == 1 and motion_files[0].is_dir():
        return str(motion_files[0].resolve())

    # Single .motion file -> use directly
    if len(motion_files) == 1 and motion_files[0].is_file():
        return str(motion_files[0].resolve())

    # Multiple files -> collect into temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="protomotions_viz_"))
    for mf in motion_files:
        if not mf.exists():
            print(f"ERROR: Motion file not found: {mf}")
            sys.exit(1)
        dst = tmp_dir / mf.name
        if not dst.exists():
            os.symlink(mf.resolve(), dst)
    return str(tmp_dir)


def main():
    # Validate motion files
    for mf in args.motion_file:
        if not mf.exists():
            print(f"ERROR: Motion file not found: {mf}")
            sys.exit(1)

    checkpoint = args.checkpoint
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    motion_path = resolve_motion_path(args.motion_file)

    # ProtoMotions uses relative paths for robot assets (e.g. protomotions/data/assets/mjcf/...)
    # so we must run from inside the ProtoMotions directory.
    os.chdir(str(PROTO_ROOT))

    print("=" * 60)
    print("ProtoMotions Physics-Based Visualizer")
    print("=" * 60)
    print(f"  Motion:     {motion_path}")
    print(f"  Simulator:  {args.simulator}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Headless:   {args.headless}")
    print("=" * 60)

    # ---- Load resolved configs from checkpoint (same as inference_agent) ----
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert resolved_configs_path.exists(), (
        f"Could not find resolved configs at {resolved_configs_path}"
    )

    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # ---- Switch simulator if needed ----
    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    # Apply backward compatibility fixes
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # ---- CLI overrides ----
    motion_lib_config.motion_file = motion_path

    # For visualization, default to 1 env (much faster startup)
    simulator_config.num_envs = args.num_envs if args.num_envs is not None else 1

    simulator_config.headless = args.headless

    # ---- Create Fabric (single-GPU inference) ----
    from protomotions.utils.fabric_config import FabricConfig
    fabric_config = FabricConfig(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    # IsaacLab needs simulation_app
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher(
            {"headless": args.headless, "device": str(fabric.device)}
        )
        simulator_extra_params["simulation_app"] = app_launcher.app

    # Terrain friction conversion
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # ---- Build all components (motion_lib, simulator, terrain, etc.) ----
    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **simulator_extra_params,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # ---- Create env ----
    from protomotions.envs.base_env.env import BaseEnv
    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    # ---- Load agent (tracker model) ----
    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config,
        env=env,
        fabric=fabric,
        root_dir=checkpoint.parent,
    )
    agent.setup()
    print(f"Loading model from checkpoint: {checkpoint}")
    agent.load(str(checkpoint), load_env=False)

    # ---- Run interactive simulation ----
    if not args.headless:
        print("\n🎮 Physics Simulation Running")
        print("Controls:")
        print("  - Left click + drag: Rotate camera")
        print("  - Right click + drag: Pan camera")
        print("  - Scroll wheel: Zoom")
        print("  - V: Toggle viewer sync")
        print("  - Esc: Quit")
    print("=" * 60)

    try:
        # Use simple_test_policy for interactive visualization (renders in real-time)
        agent.evaluator.simple_test_policy(collect_metrics=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()

    print("Visualization complete.")


if __name__ == "__main__":
    main()
