#!/bin/bash
# generate_phc_comparison_mp4s.sh — Generate MP4s comparing default vs PHC XML style.
#
# For each clip, generates:
#   1. Default style: GT + Gen bodies + stick figures
#   2. PHC style:     GT + Gen bodies + stick figures (PHC XML: wider limits, capsule torso)
#   3. PHC + PD torque: PHC style with PD torque-driven simulation overlaid
#
# Output: output/phc_comparison/clip_{ID}_{default|phc|phc_pd}.mp4
#
# Usage:
#   conda activate intermask
#   bash scripts/generate_phc_comparison_mp4s.sh
#   bash scripts/generate_phc_comparison_mp4s.sh --clips "1129 1147 1187"
#   bash scripts/generate_phc_comparison_mp4s.sh --device cuda:1

set -e

# Default clips (interaction types from prepare4 analysis)
DEFAULT_CLIPS="1129 1147 1187 1006"
CLIPS="${CLIPS:-$DEFAULT_CLIPS}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="output/phc_comparison"
CAM_PRESET="side"
FPS=30
WIDTH=1280
HEIGHT=720

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --clips) CLIPS="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --cam-preset) CAM_PRESET="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo "PHC Comparison MP4 Generation"
echo "============================================="
echo "Clips: $CLIPS"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "Camera: $CAM_PRESET"
echo ""

for CLIP in $CLIPS; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Clip $CLIP"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. Default style
    echo "  [1/3] Default style..."
    python prepare4/view_gt_vs_gen.py \
        --clip "$CLIP" \
        --joint-style default \
        --save-mp4 "$OUTPUT_DIR/clip_${CLIP}_default.mp4" \
        --cam-preset "$CAM_PRESET" \
        --fps $FPS \
        --mp4-width $WIDTH --mp4-height $HEIGHT \
        --device "$DEVICE" \
        2>&1 | tail -5
    echo "  -> $OUTPUT_DIR/clip_${CLIP}_default.mp4"

    # 2. PHC style
    echo "  [2/3] PHC style..."
    python prepare4/view_gt_vs_gen.py \
        --clip "$CLIP" \
        --joint-style phc \
        --save-mp4 "$OUTPUT_DIR/clip_${CLIP}_phc.mp4" \
        --cam-preset "$CAM_PRESET" \
        --fps $FPS \
        --mp4-width $WIDTH --mp4-height $HEIGHT \
        --device "$DEVICE" \
        2>&1 | tail -5
    echo "  -> $OUTPUT_DIR/clip_${CLIP}_phc.mp4"

    # 3. PHC style + PD torque simulation
    echo "  [3/3] PHC style + PD torque..."
    python prepare4/view_gt_vs_gen.py \
        --clip "$CLIP" \
        --joint-style phc \
        --pd-torque \
        --save-mp4 "$OUTPUT_DIR/clip_${CLIP}_phc_pd.mp4" \
        --cam-preset "$CAM_PRESET" \
        --fps $FPS \
        --mp4-width $WIDTH --mp4-height $HEIGHT \
        --device "$DEVICE" \
        2>&1 | tail -5
    echo "  -> $OUTPUT_DIR/clip_${CLIP}_phc_pd.mp4"

    echo ""
done

echo "============================================="
echo "Done! MP4s in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo "============================================="
