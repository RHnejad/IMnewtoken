#!/bin/bash
# Interactive Visualizer (WORKSTATION VERSION).
#
# Two modes available:
#   1. PHYSICS MODE (default) - Uses Newton physics simulator for realistic motion
#   2. KINEMATIC MODE - Simple playback of joint positions (faster, no physics)
#
# Usage:
#   cd /media/rh/codes/sim/IMnewtoken
#
#   # PHYSICS-BASED (realistic, uses simulator):
#   bash prepare7/visualize_workstation.sh test 1
#   bash prepare7/visualize_workstation.sh test 1 both
#   bash prepare7/visualize_workstation.sh test 1 both physics
#
#   # KINEMATIC (fast playback, no physics):
#   bash prepare7/visualize_workstation.sh test 1 person1 kinematic
#
#   # From GT or generated motions:
#   bash prepare7/visualize_workstation.sh gt 1000
#   bash prepare7/visualize_workstation.sh gen 1000 both
#
# Controls in GUI:
#   - Left click + drag: Rotate camera
#   - Right click + drag: Pan camera
#   - Scroll wheel: Zoom
#   - V: Toggle viewer sync
#   - Esc: Quit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

    # Try to find a working X11 display
    FOUND_DISPLAY=false
    
    # Check if current DISPLAY works
    if [ -n "$DISPLAY" ] && xset q &>/dev/null 2>&1; then
        FOUND_DISPLAY=true
        echo "Python: $(which python)"
        echo "Display: $DISPLAY (✓ accessible)"
    else
        # Try common display values
        for disp in :1 :0 :1001 :2; do
            if DISPLAY=$disp xset q &>/dev/null 2>&1; then
                export DISPLAY=$disp
                FOUND_DISPLAY=true
                echo "Python: $(which python)"
                echo "Display: $DISPLAY (✓ accessible, auto-detected)"
                break
            fi
        done
    fi
    
    # If no working display found, exit
    if [ "$FOUND_DISPLAY" = false ]; then
        echo ""
        echo "========================================"
        echo "ERROR: X11 Display Not Available"
        echo "========================================"
        echo "Could not find working X11 display."
        echo "Tried: :1, :0, :1001, :2"
        echo ""
        echo "Options:"
        echo "  1. If on local workstation: start X server"
        echo "  2. If SSH'd in: enable X11 forwarding (ssh -X)"
        echo "  3. Use headless video recording instead:"
        echo "     bash prepare7/record_videos_workstation.sh test 3 skeleton"
        echo ""
        echo "To check X11:"
        echo "  ls -la /tmp/.X11-unix/"
        echo "  DISPLAY=:1 xset q"
        echo "========================================"
        exit 1
    fi

    # Check critical deps
    if ! python -c "import torch, newton, warp" 2>/dev/null; then
        echo "ERROR: Missing dependencies. Run: bash prepare7/setup_deps_workstation.sh"
        exit 1
    fi
}

# Parse arguments
DATASET="${1:-test}"
CLIP_ID="${2:-1}"
MODE="${3:-person1}"  # person1, person2, or both
VIZ_MODE="${4:-physics}"  # physics or kinematic

# Determine motion files
MOTION_FILES=()

case "$DATASET" in
    gt|GT)
        MOTION_DIR="$GT_MOTION_DIR"
        if [ "$MODE" = "both" ]; then
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person1.motion")
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person2.motion")
        elif [ "$MODE" = "person2" ]; then
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person2.motion")
        else
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person1.motion")
        fi
        ;;
    gen|generated)
        MOTION_DIR="$GEN_MOTION_DIR"
        if [ "$MODE" = "both" ]; then
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person1.motion")
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person2.motion")
        elif [ "$MODE" = "person2" ]; then
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person2.motion")
        else
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person1.motion")
        fi
        ;;
    test)
        MOTION_DIR="$TEST_MOTION_DIR"
        if [ "$MODE" = "both" ]; then
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person1.motion")
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person2.motion")
        elif [ "$MODE" = "person2" ]; then
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person2.motion")
        else
            MOTION_FILES+=("$MOTION_DIR/${CLIP_ID}_person1.motion")
        fi
        ;;
    custom)
        # Direct path provided
        MOTION_FILES+=("$CLIP_ID")
        ;;
    *)
        echo "ERROR: Invalid dataset: $DATASET"
        echo ""
        echo "Usage: $0 [gt|gen|test|custom] [clip_id] [person1|person2|both]"
        echo ""
        echo "Examples:"
        echo "  $0 test 1              # Test clip 1, person 1"
        echo "  $0 test 1 both         # Test clip 1, both people side-by-side"
        echo "  $0 gt 1000             # GT clip 1000, person 1"
        echo "  $0 gen 1000 person2    # Generated clip 1000, person 2"
        echo "  $0 custom path/to/file.motion"
        exit 1
        ;;
esac

# Check files exist
for mf in "${MOTION_FILES[@]}"; do
    if [ ! -f "$mf" ]; then
        echo "ERROR: Motion file not found: $mf"
        echo ""
        echo "Available test motions:"
        ls -1 "$TEST_MOTION_DIR"/*.motion 2>/dev/null | head -10 || echo "  (none)"
        exit 1
    fi
done

echo "========================================"
if [ "$VIZ_MODE" = "physics" ]; then
    echo "ProtoMotions Physics-Based Visualizer"
else
    echo "Newton Kinematic Visualizer"
fi
echo "========================================"
echo "Motion files:"
for mf in "${MOTION_FILES[@]}"; do
    echo "  - $(basename $mf)"
done
echo ""
echo "Mode: $VIZ_MODE"
if [ "$VIZ_MODE" = "physics" ]; then
    echo "  ✓ Using Newton physics simulator"
    echo "  ✓ Realistic humanoid tracking"
else
    echo "  ✓ Kinematic playback (no physics)"
fi
echo ""
echo "Controls:"
echo "  Left click + drag: Rotate camera"
echo "  Right click + drag: Pan camera"
echo "  Scroll wheel: Zoom"
if [ "$VIZ_MODE" = "physics" ]; then
    echo "  V: Toggle viewer sync"
else
    echo "  Space: Pause/Resume"
    echo "  R: Reset camera"
fi
echo "  Esc: Quit"
echo "========================================"

activate_env

# Build command arguments
CMD_ARGS=()
for mf in "${MOTION_FILES[@]}"; do
    CMD_ARGS+=(--motion-file "$mf")
done

# Launch appropriate visualizer
if [ "$VIZ_MODE" = "physics" ]; then
    python "$SCRIPT_DIR/visualize_physics.py" "${CMD_ARGS[@]}"
else
    python "$SCRIPT_DIR/visualize_motions.py" "${CMD_ARGS[@]}"
fi
