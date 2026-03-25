#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# Full pipeline for clip 2000  (InterHuman dataset)
#
# Steps:
#   1. Clear caches (Warp kernel cache + Python __pycache__)
#   2. Retarget  – BVH joint angles → Newton joint_q + betas
#   3. Compute torques – inverse dynamics (MuJoCo solver)
#   4. Optimize interaction – differentiable Δq optimization
#   5. (Optional) Visualize – replay with Newton GL viewer
# ──────────────────────────────────────────────────────────
set -euo pipefail

CLIP="2000"
DATASET="interhuman"
GPU="cuda:0"

# Paths (relative to InterMask root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data/retargeted_v2/${DATASET}"
TORQUE_DIR="data/compute_torques/${DATASET}"

echo "============================================"
echo "  Pipeline for clip ${CLIP} (${DATASET})"
echo "============================================"

# ── Step 0: Clear caches ──────────────────────────────
echo ""
echo "[0/4] Clearing caches..."
rm -rf ~/.cache/warp/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Warp kernel cache cleared"
echo "  ✓ __pycache__ cleared"

# ── Step 1: Retarget ─────────────────────────────────
echo ""
echo "[1/4] Retargeting clip ${CLIP}..."
python prepare2/retarget.py \
    --dataset ${DATASET} \
    --clip ${CLIP} \
    --gpu ${GPU}
echo "  ✓ Retargeting complete → ${DATA_DIR}/${CLIP}_person{0,1}_joint_q.npy"

# ── Step 2: Compute torques (inverse dynamics) ───────
echo ""
echo "[2/4] Computing inverse-dynamics torques..."
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
echo "[3/4] Running interaction optimization..."
python prepare2/optimize_interaction.py \
    --clip ${CLIP} \
    --data-dir ${DATA_DIR} \
    --mode optimize \
    --method joint \
    --sim-freq 120 \
    --fps 30 \
    --downsample 2 \
    --lr 0.01 \
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

# ── Step 4 (optional): Visualize ─────────────────────
# Uncomment to launch the Newton GL viewer after optimization:
# echo ""
# echo "[4/4] Launching visualization..."
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
