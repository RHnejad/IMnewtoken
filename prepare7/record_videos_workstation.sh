#!/bin/bash
# Record MP4 videos from .motion files (WORKSTATION VERSION).
#
# Usage:
#   cd /media/rh/codes/sim/IMnewtoken
#
#   # Quick test with 3 motions (skeleton renderer - always works):
#   bash prepare7/record_videos_workstation.sh
#
#   # Record from GT motions with Newton renderer (GPU-accelerated):
#   bash prepare7/record_videos_workstation.sh gt 5 newton
#
#   # Record from generated motions with MuJoCo mesh renderer:
#   bash prepare7/record_videos_workstation.sh gen 10 mujoco
#
# Renderers:
#   skeleton - Matplotlib 3D stick-figure (default, no GPU needed, always works)
#   newton   - Newton ViewerGL with EGL (GPU-accelerated, proper 3D lighting)
#   mujoco   - MuJoCo offscreen with full SMPL mesh (GPU-accelerated)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Data directories
GT_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gt_motions"
GEN_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gen_motions"
TEST_MOTION_DIR="$SCRIPT_DIR/data/interhuman_test"

# Conda setup
CONDA_SH="/home/rh/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="${CONDA_ENV:-protomotions}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$ENV_NAME"

    # EGL/GLVND setup for headless rendering
    if [ -d "$CONDA_PREFIX/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    fi
    local EGL_JSON="$CONDA_PREFIX/share/glvnd/egl_vendor.d/10_nvidia.json"
    if [ -f "$EGL_JSON" ]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_JSON"
    fi

    echo "Python: $(which python)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())' 2>&1)"

    # Check critical deps
    if ! python -c "import torch, numpy, matplotlib" 2>/dev/null; then
        echo "ERROR: Missing dependencies. Run: bash prepare7/setup_deps_workstation.sh"
        exit 1
    fi
}

# Parse arguments
DATASET="${1:-test}"
NUM_MOTIONS="${2:-3}"
RENDERER="${3:-skeleton}"

# Determine motion directory
case "$DATASET" in
    gt|GT)
        MOTION_DIR="$GT_MOTION_DIR"
        LABEL="GT motions"
        ;;
    gen|generated)
        MOTION_DIR="$GEN_MOTION_DIR"
        LABEL="Generated motions"
        ;;
    test)
        MOTION_DIR="$TEST_MOTION_DIR"
        LABEL="Test motions"
        ;;
    *)
        # Treat as custom path
        MOTION_DIR="$1"
        LABEL="Custom motions"
        if [ ! -d "$MOTION_DIR" ]; then
            echo "ERROR: Motion directory not found: $MOTION_DIR"
            echo ""
            echo "Usage: $0 [gt|gen|test|<path>] [num_motions] [skeleton|newton|mujoco]"
            exit 1
        fi
        ;;
esac

# Check if motion directory exists and has .motion files
if [ ! -d "$MOTION_DIR" ]; then
    echo "ERROR: Motion directory not found: $MOTION_DIR"
    echo ""
    echo "Available datasets:"
    echo "  gt   - GT InterHuman motions ($GT_MOTION_DIR)"
    echo "  gen  - Generated motions ($GEN_MOTION_DIR)"
    echo "  test - Test subset ($TEST_MOTION_DIR)"
    exit 1
fi

NUM_FILES=$(ls "$MOTION_DIR"/*.motion 2>/dev/null | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "ERROR: No .motion files found in $MOTION_DIR"
    exit 1
fi

echo "========================================"
echo "PP-Motion Video Recorder (Workstation)"
echo "========================================"
echo "Dataset:      $LABEL"
echo "Motion dir:   $MOTION_DIR"
echo "Available:    $NUM_FILES .motion files"
echo "Recording:    $NUM_MOTIONS videos"
echo "Renderer:     $RENDERER"
echo "Output dir:   $OUTPUT_DIR/videos"
echo "========================================"

activate_env

mkdir -p "$OUTPUT_DIR/videos"

cd "$REPO_ROOT"
python "$SCRIPT_DIR/record_video.py" \
    --motion-dir "$MOTION_DIR" \
    --output-dir "$OUTPUT_DIR/videos" \
    --num-motions "$NUM_MOTIONS" \
    --renderer "$RENDERER"

echo "========================================"
echo "Videos saved to: $OUTPUT_DIR/videos/"
echo "========================================"

# List generated videos
ls -lh "$OUTPUT_DIR/videos/"*.mp4 2>/dev/null || echo "Note: Videos may have been saved as PNGs (check output above)"
