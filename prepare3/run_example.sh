#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# prepare3/run_example.sh — End-to-end test of the RL pipeline
#
# Usage:
#   bash prepare3/run_example.sh                    # default clip 1000, person0
#   bash prepare3/run_example.sh 1001 person1       # custom clip & person
#   STEPS=100000 bash prepare3/run_example.sh       # override training steps
#
# Environment:
#   Uses `conda activate mimickit` (Python 3.10, torch, warp, newton)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Config ────────────────────────────────────────────────────
CLIP="${1:-1000}"
PERSON="${2:-person0}"
MAX_STEPS="${STEPS:-50000}"       # short run for testing (50k ~ 2-5 min)
DEVICE="${DEVICE:-cuda:0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="/data2/rh/conda_envs/mimickit/bin/python"

# Input data
JOINT_Q="data/retargeted_v2/interhuman/${CLIP}_${PERSON}_joint_q.npy"
BETAS="data/retargeted_v2/interhuman/${CLIP}_${PERSON}_betas.npy"

# Output dirs
MOTION_DIR="data/mimickit_motions/interhuman"
OUTPUT_DIR="output/prepare3/${CLIP}_${PERSON}"
MOTION_PKL="${MOTION_DIR}/${CLIP}_${PERSON}.pkl"

echo "═══════════════════════════════════════════════════════════"
echo "  prepare3 — RL Motion Tracking Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Clip:    ${CLIP}"
echo "  Person:  ${PERSON}"
echo "  Device:  ${DEVICE}"
echo "  Steps:   ${MAX_STEPS}"
echo "  Python:  ${PYTHON}"
echo "───────────────────────────────────────────────────────────"

# ── Validate inputs ──────────────────────────────────────────
if [[ ! -f "$JOINT_Q" ]]; then
    echo "ERROR: joint_q not found: $JOINT_Q"
    echo "  Run the retargeting pipeline first (prepare2/retarget.py)"
    exit 1
fi
if [[ ! -f "$BETAS" ]]; then
    echo "ERROR: betas not found: $BETAS"
    exit 1
fi

# ── Phase 2: Convert motion to MimicKit format ──────────────
echo ""
echo "[Phase 2] Converting motion to MimicKit format..."
mkdir -p "$MOTION_DIR"

$PYTHON -c "
import sys, os, numpy as np
sys.path.insert(0, '.')
from prepare3.convert_to_mimickit import convert_clip, save_motion

joint_q = np.load('${JOINT_Q}')
print(f'  Input: {joint_q.shape[0]} frames, {joint_q.shape[0]/30:.1f}s')

motion_dict = convert_clip(joint_q, fps=30, loop_mode=0)
save_motion(motion_dict, '${MOTION_PKL}')
print(f'  Saved: ${MOTION_PKL}')
"

# ── Phase 4: Train PPO ──────────────────────────────────────
echo ""
echo "[Phase 4] Training PPO (${MAX_STEPS} steps)..."
echo ""

$PYTHON prepare3/train.py \
    --motion "$MOTION_PKL" \
    --betas "$BETAS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --device "$DEVICE" \
    --hidden-dims 512 256 128 \
    --steps-per-iter 2048 \
    --log-interval 1 \
    --save-interval 10

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Output:     ${OUTPUT_DIR}/"
echo "  Best model: ${OUTPUT_DIR}/best_policy.pt"
echo "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}/tb"
echo "═══════════════════════════════════════════════════════════"
