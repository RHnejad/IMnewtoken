#!/bin/bash
# Run ImDy per-person scoring on all 4 datasets (GT + Generated for InterHuman + InterX).
# Splits across 2 GPUs. Uses per-person-only mode (no interaction metrics yet).
#
# Usage: bash scripts/run_imdy_perperson.sh
# Or:    nohup bash scripts/run_imdy_perperson.sh > /tmp/imdy_perperson_all.log 2>&1 &

set -e

PYTHON="/data2/rh/conda_envs/torchv2/bin/python"
BASE_DIR="/media/rh/codes/sim/InterMask"
cd "$BASE_DIR"

IMDY_CONFIG="prepare5/ImDy/config/IDFD_mkr.yml"
IMDY_CKPT="prepare5/ImDy/downloaded_checkpoint/imdy_pretrain.pt"
OUT_BASE="data/imdy_metrics"

echo "============================================="
echo " ImDy Per-Person Scoring — $(date)"
echo "============================================="

# --- GPU 0: Generated InterHuman + GT InterHuman ---
echo "[GPU 0] Scoring Generated InterHuman (1098 clips)..."
OMP_NUM_THREADS=1 $PYTHON eval_pipeline/imdy_scorer.py \
    --data-dir data/retargeted_v2/generated_interhuman \
    --dataset-type retargeted \
    --output-dir "$OUT_BASE/generated_interhuman" \
    --imdy-config "$IMDY_CONFIG" \
    --imdy-checkpoint "$IMDY_CKPT" \
    --device cuda:0 \
    --per-person-only \
    --log-every 100 \
    > /tmp/imdy_gen_interhuman.log 2>&1 &
PID1=$!

# --- GPU 1: Generated InterX + GT InterX ---
echo "[GPU 1] Scoring Generated InterX (1706 clips)..."
OMP_NUM_THREADS=1 $PYTHON eval_pipeline/imdy_scorer.py \
    --data-dir data/retargeted_v2/generated_interx \
    --dataset-type retargeted \
    --output-dir "$OUT_BASE/generated_interx" \
    --imdy-config "$IMDY_CONFIG" \
    --imdy-checkpoint "$IMDY_CKPT" \
    --device cuda:1 \
    --per-person-only \
    --log-every 100 \
    > /tmp/imdy_gen_interx.log 2>&1 &
PID2=$!

echo "Waiting for Generated datasets... (PIDs: $PID1, $PID2)"
wait $PID1 && echo "  Generated InterHuman DONE" || echo "  Generated InterHuman FAILED (exit $?)"
wait $PID2 && echo "  Generated InterX DONE" || echo "  Generated InterX FAILED (exit $?)"

# --- Round 2: GT datasets (larger, run after generated finishes) ---
echo ""
echo "[GPU 0] Scoring GT InterHuman (7810 clips)..."
OMP_NUM_THREADS=1 $PYTHON eval_pipeline/imdy_scorer.py \
    --data-dir data/retargeted_v2/interhuman \
    --dataset-type retargeted \
    --output-dir "$OUT_BASE/gt_interhuman" \
    --imdy-config "$IMDY_CONFIG" \
    --imdy-checkpoint "$IMDY_CKPT" \
    --device cuda:0 \
    --per-person-only \
    --log-every 500 \
    > /tmp/imdy_gt_interhuman.log 2>&1 &
PID3=$!

echo "[GPU 1] Scoring GT InterX (11388 clips)..."
OMP_NUM_THREADS=1 $PYTHON eval_pipeline/imdy_scorer.py \
    --data-dir data/retargeted_v2/interx \
    --dataset-type retargeted \
    --output-dir "$OUT_BASE/gt_interx" \
    --imdy-config "$IMDY_CONFIG" \
    --imdy-checkpoint "$IMDY_CKPT" \
    --device cuda:1 \
    --per-person-only \
    --log-every 500 \
    > /tmp/imdy_gt_interx.log 2>&1 &
PID4=$!

echo "Waiting for GT datasets... (PIDs: $PID3, $PID4)"
wait $PID3 && echo "  GT InterHuman DONE" || echo "  GT InterHuman FAILED (exit $?)"
wait $PID4 && echo "  GT InterX DONE" || echo "  GT InterX FAILED (exit $?)"

echo ""
echo "============================================="
echo " All per-person scoring complete — $(date)"
echo "============================================="
echo "Results in: $OUT_BASE/"
ls -lh "$OUT_BASE"/*/summary.json 2>/dev/null
