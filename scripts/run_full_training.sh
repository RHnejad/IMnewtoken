#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# run_full_training.sh — Full Physics-Informed VQ-VAE Training
# ══════════════════════════════════════════════════════════════
#
# Phase 1: Kinematic warmup → ramp physics → full training
#
# Usage:
#   bash scripts/run_full_training.sh
#   bash scripts/run_full_training.sh --with-physics
#   bash scripts/run_full_training.sh --resume
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse args
WITH_PHYSICS=false
RESUME=false
GPU_ID=0
for arg in "$@"; do
    case $arg in
        --with-physics) WITH_PHYSICS=true ;;
        --resume)       RESUME=true ;;
        --gpu=*)        GPU_ID="${arg#*=}" ;;
    esac
done

# ── Configuration ──
DATA_DIR="data/InterHuman"
PRETRAINED_CKPT="checkpoints/interhuman/vq_default/model/best_fid.tar"
OUTPUT_DIR="./outputs/newton_vqvae_full"
NAME="newton_vq_phase1"

echo "══════════════════════════════════════════════════════════"
echo " Full Training: Physics-Informed VQ-VAE"
echo "══════════════════════════════════════════════════════════"
echo " Data dir:    $DATA_DIR"
echo " Pretrained:  $PRETRAINED_CKPT"
echo " Output:      $OUTPUT_DIR"
echo " Physics:     $WITH_PHYSICS"
echo " GPU:         $GPU_ID"
echo " Resume:      $RESUME"
echo "══════════════════════════════════════════════════════════"

# ── Physics args ──
PHYSICS_ARGS=""
if [ "$WITH_PHYSICS" = true ]; then
    PHYSICS_ARGS="--physics_warmup_epochs 5 --physics_ramp_epochs 10"
fi

# ── Run training ──
python -m newton_vqvae.train \
    --data_dir "$DATA_DIR" \
    --name "$NAME" \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --batch_size 32 \
    --max_epoch 50 \
    --lr 2e-4 \
    --window_size 64 \
    --physics_every_n_batches 4 \
    --output_dir "$OUTPUT_DIR" \
    --gpu_id "$GPU_ID" \
    $PHYSICS_ARGS

echo ""
echo "Training complete!"
echo "Check outputs in: $OUTPUT_DIR"
