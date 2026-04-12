#!/bin/bash
# PP-Motion evaluation pipeline for InterHuman GT and generated motions.
#
# Step 1: Convert InterHuman motions (CPU, run on haas001)
# Step 2: Run tracker evaluation (GPU, run via runai)
# Step 3: Compare results (CPU)
#
# Usage:
#   bash prepare7/run_pp_motion.sh convert   # Step 1
#   bash prepare7/run_pp_motion.sh eval_gt   # Step 2a (GPU)
#   bash prepare7/run_pp_motion.sh eval_gen  # Step 2b (GPU)
#   bash prepare7/run_pp_motion.sh compare   # Step 3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$SCRIPT_DIR/ProtoMotions"

INTERHUMAN_DIR="$REPO_ROOT/data/InterHuman"
GENERATED_DIR="/mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/data/generated/interhuman"
GT_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gt_motions"
GEN_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gen_motions"
OUTPUT_DIR="$SCRIPT_DIR/output"

CHECKPOINT="$PROTO_DIR/data/pretrained_models/motion_tracker/smpl/last.ckpt"
NUM_ENVS=64
OUTPUT_FPS=30

# Conda setup (persistent env on shared storage)
CONDA_SH="/mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-protomotions}"

setup_conda() {
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  echo "Using python: $(which python)"
}

case "${1:-help}" in
  convert)
    echo "=== Step 1: Converting InterHuman motions to ProtoMotions format ==="
    cd "$PROTO_DIR"

    echo "Converting GT motions..."
    python "$SCRIPT_DIR/convert_interhuman_to_proto.py" \
      --interhuman-dir "$INTERHUMAN_DIR" \
      --output-dir "$GT_MOTION_DIR" \
      --output-fps "$OUTPUT_FPS"

    echo "Converting generated motions from $GENERATED_DIR ..."
    python "$SCRIPT_DIR/convert_interhuman_to_proto.py" \
      --interhuman-dir "$GENERATED_DIR" \
      --output-dir "$GEN_MOTION_DIR" \
      --output-fps "$OUTPUT_FPS"

    echo "Conversion complete."
    echo "  GT motions:  $GT_MOTION_DIR"
    echo "  Gen motions: $GEN_MOTION_DIR"
    ;;

  eval_gt)
    echo "=== Step 2a: Evaluating GT motions (requires GPU) ==="
    setup_conda
    cd "$PROTO_DIR"
    python "$SCRIPT_DIR/run_evaluation.py" \
      --checkpoint "$CHECKPOINT" \
      --simulator newton \
      --full-eval --headless \
      --num-envs "$NUM_ENVS" \
      --motion-file "$GT_MOTION_DIR" \
      --save-per-motion "$OUTPUT_DIR/eval_gt.json"
    ;;

  eval_gen)
    echo "=== Step 2b: Evaluating generated motions (requires GPU) ==="
    setup_conda
    cd "$PROTO_DIR"
    python "$SCRIPT_DIR/run_evaluation.py" \
      --checkpoint "$CHECKPOINT" \
      --simulator newton \
      --full-eval --headless \
      --num-envs "$NUM_ENVS" \
      --motion-file "$GEN_MOTION_DIR" \
      --save-per-motion "$OUTPUT_DIR/eval_generated.json"
    ;;

  compare)
    echo "=== Step 3: Comparing GT vs Generated results ==="
    python "$SCRIPT_DIR/compare_results.py" \
      --gt-json "$OUTPUT_DIR/eval_gt.json" \
      --gen-json "$OUTPUT_DIR/eval_generated.json" \
      --output-dir "$OUTPUT_DIR/comparison"
    ;;

  convert_test)
    echo "=== Quick test: Converting 5 clips ==="
    cd "$PROTO_DIR"
    python "$SCRIPT_DIR/convert_interhuman_to_proto.py" \
      --interhuman-dir "$INTERHUMAN_DIR" \
      --output-dir "$SCRIPT_DIR/data/interhuman_test" \
      --output-fps "$OUTPUT_FPS" \
      --clip-ids "1,2,3,4,5" \
      --force
    ;;

  help|*)
    echo "Usage: bash prepare7/run_pp_motion.sh <command>"
    echo ""
    echo "Commands:"
    echo "  convert       Convert all InterHuman motions (CPU)"
    echo "  convert_test  Convert 5 clips for quick testing (CPU)"
    echo "  eval_gt       Run tracker on GT motions (GPU required)"
    echo "  eval_gen      Run tracker on generated motions (GPU required)"
    echo "  compare       Compare GT vs generated results (CPU)"
    ;;
esac
