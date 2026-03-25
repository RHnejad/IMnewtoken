#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# train_phase1.sh — Phase 1: Per-character physics training
# ══════════════════════════════════════════════════════════════
#
# Usage:
#   conda activate mimickit
#   bash scripts/train_phase1.sh
#
# Environment: mimickit (Python 3.10+, Newton, Warp, PyTorch)
# ══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Configuration ──
DATA_ROOT="${DATA_ROOT:-/media/rh/codes/sim/InterMask/data/interhuman}"
INTERMASK_CKPT="${INTERMASK_CKPT:-/media/rh/codes/sim/InterMask/checkpoints/interhuman/finest.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/physics_vqvae_phase1}"

echo "══════════════════════════════════════════════════════════"
echo " Phase 1: Per-Character Physics-Informed VQ-VAE Training"
echo "══════════════════════════════════════════════════════════"
echo " Data root:       $DATA_ROOT"
echo " InterMask ckpt:  $INTERMASK_CKPT"
echo " Output:          $OUTPUT_DIR"
echo "══════════════════════════════════════════════════════════"

python -m newton_vqvae.train \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --intermask_ckpt "$INTERMASK_CKPT" \
    --batch_size 32 \
    --max_epoch 50 \
    --lr 2e-4 \
    --window_size 64 \
    --physics_warmup_epochs 5 \
    --physics_ramp_epochs 10 \
    --physics_every_n_batches 4 \
    --smooth_sigma 1.0 \
    --alpha 10.0 \
    --beta 0.001 \
    --gamma 1.0 \
    --delta 50.0 \
    --epsilon 5.0 \
    "$@"

echo ""
echo "Phase 1 training complete!"
echo "Best model: $OUTPUT_DIR/models/best.tar"
echo ""

# Run evaluation
echo "Running evaluation..."
python -m newton_vqvae.evaluate \
    --checkpoint "$OUTPUT_DIR/models/best.tar" \
    --data_root "$DATA_ROOT" \
    --output_file "$OUTPUT_DIR/eval_results.json" \
    --max_clips 50

echo "Done!"
