#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# Full pipeline for clip 10  (InterHuman dataset)
#   - Runs on cuda:1
#   - Higher learning rate (0.05)
#   - No cache clearing (can run in parallel with other clips)
# ──────────────────────────────────────────────────────────
set -euo pipefail

CLIP="10"
DATASET="interhuman"
GPU="cuda:1"

# Paths (relative to InterMask root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data/retargeted_v2/${DATASET}"
TORQUE_DIR="data/compute_torques/${DATASET}"

echo "============================================"
echo "  Pipeline for clip ${CLIP} (${DATASET}) on ${GPU}"
echo "============================================"

# ── Step 1: Retarget ─────────────────────────────────
echo ""
echo "[1/3] Retargeting clip ${CLIP}..."
python prepare2/retarget.py \
    --dataset ${DATASET} \
    --clip ${CLIP} \
    --gpu ${GPU}
echo "  ✓ Retargeting complete → ${DATA_DIR}/${CLIP}_person{0,1}_joint_q.npy"

# ── Step 2: Compute torques (inverse dynamics) ───────
echo ""
echo "[2/3] Computing inverse-dynamics torques..."
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
echo "  ✓ Torques saved → ${TORQUE_DIR}/${CLIP}_person{0,1}_torques_solo.npy"

# ── Step 3: Optimize interaction (differentiable Δq) ─
echo ""
echo "[3/3] Running interaction optimization..."
python prepare2/optimize_interaction.py \
    --clip ${CLIP} \
    --data-dir ${DATA_DIR} \
    --mode optimize \
    --method joint \
    --sim-freq 120 \
    --fps 30 \
    --downsample 2 \
    --lr 0.05 \
    --reg-lambda 0.01 \
    --ke-root 5000 \
    --kd-root 500 \
    --ke-joint 200 \
    --kd-joint 20 \
    --window 5 \
    --epochs 20 \
    --headless \
    --force \
    --device ${GPU}
echo "  ✓ Optimization complete → ${DATA_DIR}/${CLIP}_delta_torques.npy"

# ── (Optional) Visualize ─────────────────────────────
# Uncomment to launch the Newton GL viewer after optimization:
# python prepare2/simulate_torques.py \
#     --clip ${CLIP} \
#     --data-dir ${DATA_DIR} \
#     --fps 30 \
#     --downsample 2 \
#     --device ${GPU}

echo ""
echo "============================================"
echo "  Pipeline complete for clip ${CLIP}"
echo "============================================"
