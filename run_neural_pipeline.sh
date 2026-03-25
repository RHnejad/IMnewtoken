#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# Neural actor pipeline for clip 1000 (InterHuman)
#
# Uses a learned MLP instead of flat Δq optimization:
#   obs(t) → actor → Δq(t) → PD(ref + Δq) + solo → physics
#
# Steps:
#   1. Retarget (if needed)
#   2. Compute torques (if needed)
#   3. Run tests
#   4. Train neural actor
# ──────────────────────────────────────────────────────────
set -euo pipefail

CLIP="${1:-1000}"
DATASET="interhuman"
GPU="${2:-cuda:0}"
EPOCHS="${3:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data/retargeted_v2/${DATASET}"
TORQUE_DIR="data/compute_torques/${DATASET}"

echo "============================================"
echo "  Neural Actor Pipeline — clip ${CLIP}"
echo "  GPU: ${GPU}, Epochs: ${EPOCHS}"
echo "============================================"

# ── Step 1: Retarget (skip if files exist) ────────────
JQ_FILE="${DATA_DIR}/${CLIP}_person0_joint_q.npy"
if [[ ! -f "$JQ_FILE" ]]; then
    echo ""
    echo "[1/4] Retargeting clip ${CLIP}..."
    python prepare2/retarget.py \
        --dataset ${DATASET} \
        --clip ${CLIP} \
        --gpu ${GPU}
else
    echo ""
    echo "[1/4] Retarget: SKIP (files exist)"
fi

# ── Step 2: Compute torques (skip if files exist) ────
TORQ_FILE="${TORQUE_DIR}/${CLIP}_person0_torques_solo.npy"
if [[ ! -f "$TORQ_FILE" ]]; then
    echo ""
    echo "[2/4] Computing torques..."
    python prepare2/compute_torques.py \
        --clip ${CLIP} \
        --data-dir ${DATA_DIR} \
        --output-dir ${TORQUE_DIR} \
        --method inverse \
        --fps 30 \
        --downsample 2 \
        --save \
        --force \
        --gpu ${GPU}
else
    echo ""
    echo "[2/4] Torques: SKIP (files exist)"
fi

# ── Step 3: Run unit tests ───────────────────────────
echo ""
echo "[3/4] Running tests..."
python prepare2/test_actor_network.py
echo "  ✓ All tests passed"

# ── Step 4: Train neural actor ───────────────────────
echo ""
echo "[4/4] Training neural actor..."
python prepare2/optimize_neural.py \
    --clip ${CLIP} \
    --data-dir ${DATA_DIR} \
    --sim-freq 120 \
    --fps 30 \
    --downsample 2 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --reg-lambda 0.01 \
    --hidden 256 256 \
    --max-delta 0.05 \
    --window 5 \
    --epochs ${EPOCHS} \
    --grad-clip 10.0 \
    --param-grad-clip 1.0 \
    --patience 10 \
    --normalize-obs \
    --force \
    --device ${GPU}

echo ""
echo "============================================"
echo "  Neural pipeline complete for clip ${CLIP}"
echo "  Output: ${DATA_DIR}/${CLIP}_actor.pt"
echo "============================================"
