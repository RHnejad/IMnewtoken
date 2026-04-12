#!/bin/bash
# PP-Motion evaluation pipeline for InterHuman GT and generated motions (WORKSTATION VERSION).
#
# Step 1: Convert InterHuman motions (CPU)
# Step 2: Run tracker evaluation (GPU)
# Step 3: Compare results (CPU)
#
# Usage:
#   bash prepare7/run_pp_motion_workstation.sh convert   # Step 1
#   bash prepare7/run_pp_motion_workstation.sh eval_gt   # Step 2a (GPU)
#   bash prepare7/run_pp_motion_workstation.sh eval_gen  # Step 2b (GPU)
#   bash prepare7/run_pp_motion_workstation.sh compare   # Step 3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$SCRIPT_DIR/ProtoMotions"

INTERHUMAN_DIR="$REPO_ROOT/data/InterHuman"
GENERATED_DIR="$REPO_ROOT/data/generated/interhuman"
GT_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gt_motions"
GEN_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gen_motions"
OUTPUT_DIR="$SCRIPT_DIR/output"

CHECKPOINT="$PROTO_DIR/data/pretrained_models/motion_tracker/smpl/last.ckpt"
NUM_ENVS=64
OUTPUT_FPS=30

# Conda setup (local workstation)
CONDA_SH="/home/rh/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-protomotions}"

setup_conda() {
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  
  # EGL/GLVND setup for headless MuJoCo rendering
  if [ -d "$CONDA_PREFIX/lib" ]; then
      export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
  local EGL_JSON="$CONDA_PREFIX/share/glvnd/egl_vendor.d/10_nvidia.json"
  if [ -f "$EGL_JSON" ]; then
      export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_JSON"
  fi
  
  echo "Using python: $(which python)"
  echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())' 2>&1)"
}

case "${1:-help}" in
  convert)
    echo "=== Step 1: Converting InterHuman motions to ProtoMotions format ==="
    setup_conda
    cd "$PROTO_DIR"

    echo "Converting GT motions..."
    python "$SCRIPT_DIR/convert_interhuman_to_proto.py" \
      --interhuman-dir "$INTERHUMAN_DIR" \
      --output-dir "$GT_MOTION_DIR" \
      --output-fps "$OUTPUT_FPS"

    echo "Converting generated motions from $GENERATED_DIR ..."
    if [ -d "$GENERATED_DIR" ]; then
        python "$SCRIPT_DIR/convert_interhuman_to_proto.py" \
          --interhuman-dir "$GENERATED_DIR" \
          --output-dir "$GEN_MOTION_DIR" \
          --output-fps "$OUTPUT_FPS"
    else
        echo "WARNING: Generated dir not found: $GENERATED_DIR"
    fi

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
    setup_conda
    python "$SCRIPT_DIR/compare_results.py" \
      --gt-json "$OUTPUT_DIR/eval_gt.json" \
      --gen-json "$OUTPUT_DIR/eval_generated.json" \
      --output-dir "$OUTPUT_DIR/comparison"
    ;;

  convert_test)
    echo "=== Quick test: Converting first 5 clips ==="
    setup_conda
    cd "$PROTO_DIR"
    python "$SCRIPT_DIR/convert_interhuman_to_proto.py" \
      --interhuman-dir "$INTERHUMAN_DIR" \
      --output-dir "$SCRIPT_DIR/data/interhuman_test" \
      --output-fps "$OUTPUT_FPS" \
      --clip-ids "1,2,3,4,5" \
      --force
    ;;

  *)
    echo "Usage: $0 {convert|eval_gt|eval_gen|compare|convert_test}"
    echo ""
    echo "  convert      - Convert InterHuman motions to ProtoMotions format"
    echo "  eval_gt      - Evaluate GT motions (GPU required)"
    echo "  eval_gen     - Evaluate generated motions (GPU required)"
    echo "  compare      - Compare GT vs generated results"
    echo "  convert_test - Quick test with 5 clips"
    exit 1
    ;;
esac
