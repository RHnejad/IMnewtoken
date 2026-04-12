#!/usr/bin/env python3
"""Run ProtoMotions SMPL tracker evaluation on converted InterHuman motions.

Monkey-patches MimicEvaluator.process_eval_results to save per-motion
tracking MPJPE (gt_error) to JSON, then delegates to inference_agent.main().

Must be run from prepare7/ProtoMotions/ directory with GPU available.

Usage (on GPU node):
    cd prepare7/ProtoMotions
    python ../../prepare7/run_evaluation.py \
        --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
        --simulator isaacgym \
        --full-eval --headless \
        --num-envs 64 \
        --motion-file ../data/interhuman_motions \
        --save-per-motion ../output/eval_gt.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Must set up paths before any ProtoMotions imports
PROTO_ROOT = Path(__file__).resolve().parent / "ProtoMotions"
sys.path.insert(0, str(PROTO_ROOT))

# Parse --save-per-motion before ProtoMotions' argparse consumes args
_output_json = None
for i, arg in enumerate(sys.argv):
    if arg == "--save-per-motion" and i + 1 < len(sys.argv):
        _output_json = sys.argv[i + 1]
        # Remove from sys.argv so inference_agent's parser doesn't choke
        sys.argv.pop(i)
        sys.argv.pop(i)
        break

if _output_json is None:
    print("ERROR: --save-per-motion <path.json> is required", file=sys.stderr)
    print("Pass all other flags as normal inference_agent.py args.", file=sys.stderr)
    sys.exit(1)

# --- Logging setup ---
_log_path = _output_json.replace(".json", ".log") if _output_json.endswith(".json") else _output_json + ".log"
os.makedirs(Path(_log_path).parent, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_evaluation")


def _install_per_motion_hook(output_path):
    """Monkey-patch MimicEvaluator at the class level to save per-motion metrics."""
    from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator

    _original = MimicEvaluator.process_eval_results

    def _patched(self):
        logger.info("Processing evaluation results...")
        t0 = time.time()
        to_log, success_rate = _original(self)

        per_motion = {}
        num_motions = len(self._motion_failed)
        motion_lib = self.motion_lib
        num_failed = 0

        for i in range(num_motions):
            motion_file = motion_lib.motion_files[i]
            motion_name = Path(motion_file).stem
            failed = bool(self._motion_failed[i].item())
            if failed:
                num_failed += 1
            entry = {
                "file": str(motion_file),
                "failed": failed,
            }
            for name in self._component_value_sum:
                step_count = self._component_step_count[name][i].item()
                if step_count > 0:
                    entry[f"{name}_mean"] = (
                        self._component_value_sum[name][i].item() / step_count
                    )
                    entry[f"{name}_min"] = self._component_value_min[name][i].item()
                    entry[f"{name}_max"] = self._component_value_max[name][i].item()
                    entry[f"{name}_steps"] = int(step_count)
                else:
                    entry[f"{name}_mean"] = None

            per_motion[motion_name] = entry

        results = {
            "aggregate": to_log,
            "per_motion": per_motion,
            "success_rate": success_rate,
            "num_motions": num_motions,
        }
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Output:       {output_path}")
        logger.info(f"  Num motions:  {num_motions}")
        logger.info(f"  Num failed:   {num_failed}")
        logger.info(f"  Success rate: {success_rate:.4f}")
        for k, v in sorted(to_log.items()):
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.6f}")
            else:
                logger.info(f"  {k}: {v}")
        logger.info(f"  Processing time: {elapsed:.1f}s")
        logger.info("=" * 60)

        return to_log, success_rate

    MimicEvaluator.process_eval_results = _patched


# Install the hook before main() runs
_install_per_motion_hook(_output_json)

logger.info("Starting ProtoMotions evaluation")
logger.info(f"  Output JSON: {_output_json}")
logger.info(f"  Log file:    {_log_path}")
logger.info(f"  Args:        {sys.argv[1:]}")

# Now run ProtoMotions' inference_agent.main()
_t_start = time.time()
try:
    from protomotions.inference_agent import main
    main()
    logger.info(f"Total wall time: {time.time() - _t_start:.1f}s")
except Exception:
    logger.exception("FATAL: Evaluation failed with exception")
    sys.exit(1)
