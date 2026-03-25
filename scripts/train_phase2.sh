#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# train_phase2.sh — Phase 2: Paired interaction physics training
# ══════════════════════════════════════════════════════════════
#
# Prerequisites: Phase 1 model trained and available
#
# Usage:
#   conda activate mimickit
#   bash scripts/train_phase2.sh
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Configuration ──
DATA_ROOT="${DATA_ROOT:-/media/rh/codes/sim/InterMask/data/interhuman}"
PHASE1_CKPT="${PHASE1_CKPT:-./outputs/physics_vqvae_phase1/models/best.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/physics_vqvae_phase2}"

echo "══════════════════════════════════════════════════════════"
echo " Phase 2: Paired Interaction Physics VQ-VAE Training"
echo "══════════════════════════════════════════════════════════"
echo " Data root:       $DATA_ROOT"
echo " Phase1 ckpt:     $PHASE1_CKPT"
echo " Output:          $OUTPUT_DIR"
echo "══════════════════════════════════════════════════════════"

# Phase 2 uses Phase 1 model as starting point
python -m newton_vqvae.train \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --intermask_ckpt "$PHASE1_CKPT" \
    --batch_size 16 \
    --max_epoch 30 \
    --lr 5e-5 \
    --window_size 64 \
    --physics_warmup_epochs 2 \
    --physics_ramp_epochs 5 \
    --physics_every_n_batches 2 \
    --smooth_sigma 1.0 \
    --alpha 10.0 \
    --beta 0.001 \
    --gamma 1.5 \
    --delta 50.0 \
    --epsilon 5.0 \
    "$@"

echo ""
echo "Phase 2 training complete!"
echo "Best model: $OUTPUT_DIR/models/best.tar"
