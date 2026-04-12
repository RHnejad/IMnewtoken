#!/usr/bin/env python3
"""Record proper physics-rendered videos from InterHuman motions via ProtoMotions.

Uses Newton simulator's ViewerGL to render the actual SMPL humanoid mesh
alongside reference-motion markers (green spheres) — the same visualization
style as PHC / ProtoMotions demos, but automated.

The script:
  1. Patches Newton's _update_simulator_markers to render reference markers as
     green spheres via viewer.log_points (Newton has a stub here by default).
  2. Patches Newton's render() to auto-start recording on frame 1 and
     capture every frame via get_frame() → PNG → MP4.
  3. Runs the full ProtoMotions inference pipeline (physics simulation + tracking).

Output: MP4 video + .motion file in the output directory.

Usage:
    cd /media/rh/codes/sim/IMnewtoken
    conda run -p /data2/rh/conda_envs/protomotions python prepare7/record_physics_video.py \
        --motion-dir prepare7/data/interhuman_test \
        --output-dir prepare7/output/physics_videos \
        --num-envs 1

    # More envs (shows multiple humanoids tracking different motions):
    conda run -p /data2/rh/conda_envs/protomotions python prepare7/record_physics_video.py \
        --motion-dir prepare7/data/interhuman_test \
        --output-dir prepare7/output/physics_videos \
        --num-envs 4 --width 1920 --height 1080
"""

import argparse
import logging
import os
import sys
import types
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("record_physics")

SCRIPT_DIR = Path(__file__).resolve().parent
PROTO_ROOT = SCRIPT_DIR / "ProtoMotions"
CHECKPOINT = PROTO_ROOT / "data" / "pretrained_models" / "motion_tracker" / "smpl" / "last.ckpt"


def main():
    parser = argparse.ArgumentParser(
        description="Record physics-rendered videos from .motion files via ProtoMotions"
    )
    parser.add_argument("--motion-dir", required=True, help="Dir with .motion files")
    parser.add_argument("--output-dir", default="prepare7/output/physics_videos")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = Path(args.motion_dir).resolve()

    motion_files = sorted(motion_dir.glob("*.motion"))
    if not motion_files:
        logger.error(f"No .motion files in {motion_dir}")
        sys.exit(1)
    logger.info(f"Found {len(motion_files)} motions in {motion_dir}")

    # ── Setup ProtoMotions ──
    sys.path.insert(0, str(PROTO_ROOT))
    os.chdir(str(PROTO_ROOT))

    import torch
    import warp as wp
    import numpy as np
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    import_simulator_before_torch("newton")

    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.fabric_config import FabricConfig
    from lightning.fabric import Fabric
    from dataclasses import asdict

    checkpoint_path = Path(args.checkpoint)
    resolved_configs_path = checkpoint_path.parent / "resolved_configs_inference.pt"
    resolved_configs = torch.load(resolved_configs_path, map_location="cpu", weights_only=False)

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Switch simulator to newton
    current_simulator = simulator_config._target_.split(".")[-3]
    if current_simulator != "newton":
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator="newton",
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    # NOT headless → viewer + recording system are active
    simulator_config.headless = False
    simulator_config.num_envs = args.num_envs
    motion_lib_config.motion_file = str(motion_dir)

    fabric_config = FabricConfig(
        accelerator="gpu", devices=1, num_nodes=1,
        loggers=[], callbacks=[],
    )
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # ── Monkey-patch Newton ViewerGL to use requested resolution ──
    import newton
    _OrigViewerGL = newton.viewer.ViewerGL

    _viewer_width = args.width
    _viewer_height = args.height

    class SizedViewerGL(_OrigViewerGL):
        def __init__(self, width=1920, height=1080, vsync=False, headless=False):
            logger.info(f"  Creating ViewerGL ({_viewer_width}x{_viewer_height})")
            super().__init__(
                width=_viewer_width, height=_viewer_height,
                vsync=vsync, headless=headless,
            )

    newton.viewer.ViewerGL = SizedViewerGL

    from protomotions.utils.component_builder import build_all_components
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

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

    # ── Patch: implement _update_simulator_markers for Newton ──
    # Renders reference-motion markers as green spheres via log_points
    from protomotions.simulator.base_simulator.config import MarkerState

    def _newton_update_markers(self, markers_state=None):
        if not markers_state or self.viewer is None:
            return
        for name, ms in markers_state.items():
            if name == "terrain_markers":
                continue
            # Use env 0's markers for the viewer
            pos = ms.translation[0].cpu().numpy()  # (num_markers, 3)
            n = pos.shape[0]
            # Green spheres for reference markers
            radii = np.full(n, 0.04, dtype=np.float32)
            colors = np.tile(np.array([0.2, 0.9, 0.3], dtype=np.float32), (n, 1))
            # Newton's viewer needs CUDA arrays
            self.viewer.log_points(
                name,
                wp.array(pos, dtype=wp.vec3, device="cuda:0"),
                wp.array(radii, dtype=wp.float32, device="cuda:0"),
                wp.array(colors, dtype=wp.vec3, device="cuda:0"),
            )

    simulator._update_simulator_markers = types.MethodType(_newton_update_markers, simulator)

    # ── Patch: override recording output path ──
    simulator._user_recording_video_path = os.path.join(
        str(output_dir), "recording-%s"
    )

    # ── Patch: auto-start recording on first render, disable keyboard polling ──
    _NewtonSimClass = type(simulator)
    _orig_render = _NewtonSimClass.render
    _auto_started = [False]

    def _patched_render(self):
        """Render with auto-record and skip keyboard (no interactive window)."""
        if not self.headless:
            if not self._camera_initialized:
                self._init_camera()
                self._camera_initialized = True
            else:
                self._update_camera()

            # Skip keyboard handling (no interactive window needed)

            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

        # Auto-start recording on first frame
        if not _auto_started[0]:
            _auto_started[0] = True
            logger.info("  Auto-starting video recording...")
            self._toggle_video_record()

        # Call the base RecordingMixin.render() for frame capture
        from protomotions.simulator.base_simulator.record import RecordingMixin
        RecordingMixin.render(self)

    simulator.render = types.MethodType(_patched_render, simulator)

    # ── Patch: _write_viewport_to_file to actually save frames ──
    def _newton_write_viewport(self, file_name):
        import matplotlib.pyplot as plt
        viewport = self.viewer.get_frame().numpy()  # [H, W, 3] uint8
        plt.imsave(file_name, viewport)

    simulator._write_viewport_to_file = types.MethodType(_newton_write_viewport, simulator)

    # ── Create agent and run ──
    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric,
        root_dir=checkpoint_path.parent,
    )
    agent.setup()
    agent.load(str(args.checkpoint), load_env=False)

    logger.info("Running full evaluation with physics rendering...")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Motions: {len(motion_files)} from {motion_dir}")
    logger.info(f"  Envs: {args.num_envs}")

    try:
        agent.evaluator.eval_count = 0
        evaluation_log, evaluated_score = agent.evaluator.evaluate()

        # Auto-stop recording → triggers video compilation
        logger.info("  Auto-stopping video recording → compiling MP4...")
        simulator._toggle_video_record()

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        for key, value in sorted(evaluation_log.items()):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
        logger.info(f"  Success rate: {evaluated_score:.4f}")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Evaluation failed")
        try:
            simulator._toggle_video_record()
        except Exception:
            pass
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()

    # List output files
    logger.info(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file() and not f.name.endswith(".png"):
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
