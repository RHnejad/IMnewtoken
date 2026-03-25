#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# run_overfit_test.sh — Quick overfit test on small data subset
# ══════════════════════════════════════════════════════════════
#
# Tests the full pipeline by overfitting on a few clips.
# Phase 1: Kinematic-only (no physics) to verify VQ-VAE works.
# Phase 2 (if --with-physics): Adds Newton physics losses.
#
# Usage:
#   bash scripts/run_overfit_test.sh [--with-physics]
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse args
WITH_PHYSICS=false
for arg in "$@"; do
    case $arg in
        --with-physics) WITH_PHYSICS=true ;;
    esac
done

# ── Configuration ──
DATA_DIR="data/InterHuman"
PRETRAINED_CKPT="checkpoints/interhuman/vq_default/model/best_fid.tar"
OUTPUT_DIR="./outputs/overfit_test"

echo "══════════════════════════════════════════════════════════"
echo " Overfit Test: Physics-Informed VQ-VAE"
echo "══════════════════════════════════════════════════════════"
echo " Data dir:    $DATA_DIR"
echo " Pretrained:  $PRETRAINED_CKPT"
echo " Output:      $OUTPUT_DIR"
echo " Physics:     $WITH_PHYSICS"
echo "══════════════════════════════════════════════════════════"

# Create a small subset split file (first 10 clips)
mkdir -p "$OUTPUT_DIR"
SPLIT_DIR="$DATA_DIR/split"

# Backup original splits
cp "$SPLIT_DIR/train.txt" "$SPLIT_DIR/train.txt.bak"
cp "$SPLIT_DIR/val.txt" "$SPLIT_DIR/val.txt.bak"

# Restore on exit
cleanup() {
    echo "Restoring original split files..."
    mv "$SPLIT_DIR/train.txt.bak" "$SPLIT_DIR/train.txt"
    mv "$SPLIT_DIR/val.txt.bak" "$SPLIT_DIR/val.txt"
}
trap cleanup EXIT

# Write subset splits
head -10 "$SPLIT_DIR/train.txt.bak" > "$SPLIT_DIR/train.txt"
head -5 "$SPLIT_DIR/val.txt.bak" > "$SPLIT_DIR/val.txt"

echo ""
echo "Train clips (first 10):"
cat "$SPLIT_DIR/train.txt"
echo ""

# ── Physics args ──  
PHYSICS_ARGS=""
if [ "$WITH_PHYSICS" = true ]; then
    PHYSICS_ARGS="--physics_warmup_epochs 1 --physics_ramp_epochs 2"
fi

# ── Run training ──
python -m newton_vqvae.train \
    --data_dir "$DATA_DIR" \
    --name "overfit_test" \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --batch_size 4 \
    --max_epoch 5 \
    --lr 1e-4 \
    --window_size 64 \
    --physics_every_n_batches 2 \
    --output_dir "$OUTPUT_DIR" \
    $PHYSICS_ARGS

echo ""
echo "Overfit test complete!"
echo "Check outputs in: $OUTPUT_DIR"
