#!/bin/bash
# eval_pipeline/run_all.sh
# Automates the extensive computationally-heavy physics portions 
# of the InterMask evaluation pipeline. Designed to split the workload
# across two GPUs (cuda:0 and cuda:1).

echo "Starting InterMask Evaluation Physics Pipeline..."
echo "This script assumes you have already run retarget.py for all 4 datasets."
echo "Ensuring mimickit conda environment..."

# Setup 
BASE_DIR="/media/rh/codes/sim/InterMask"
cd $BASE_DIR

# --- PHASE 1: PD Torques ---
echo ""
echo "==========================================="
echo " PHASE 1: Computing PD Torques"
echo "==========================================="

# Generated Datasets (~30fps native -> downsample 1)
echo "[cuda:0] Computing PD torques for Generated InterX..."
OMP_NUM_THREADS=1 python prepare2/compute_torques.py \
    --dataset interx \
    --data-dir data/retargeted_v2/generated_interx \
    --output-dir data/compute_torques/generated_interx \
    --method pd --save --downsample 1 --workers 1 --gpu cuda:0 \
    > /tmp/eval_torques_gen_interx.log 2>&1 &
PID1=$!

echo "[cuda:1] Computing PD torques for Generated InterHuman..."
OMP_NUM_THREADS=1 python prepare2/compute_torques.py \
    --dataset interhuman \
    --data-dir data/retargeted_v2/generated_interhuman \
    --output-dir data/compute_torques/generated_interhuman \
    --method pd --save --downsample 1 --workers 1 --gpu cuda:1 \
    > /tmp/eval_torques_gen_interhuman.log 2>&1 &
PID2=$!

echo "Waiting for Generated torques to finish..."
wait $PID1
wait $PID2
echo "Generated torques COMPLETE ✅"


# GT Datasets (60fps raw -> downsample 2)
echo "[cuda:0] Computing PD torques for GT InterX..."
OMP_NUM_THREADS=1 python prepare2/compute_torques.py \
    --dataset interx \
    --output-dir data/compute_torques/gt_interx \
    --method pd --save --workers 1 --gpu cuda:0 \
    > /tmp/eval_torques_gt_interx.log 2>&1 &
PID3=$!

echo "[cuda:1] Computing PD torques for GT InterHuman..."
OMP_NUM_THREADS=1 python prepare2/compute_torques.py \
    --dataset interhuman \
    --output-dir data/compute_torques/gt_interhuman \
    --method pd --save --workers 1 --gpu cuda:1 \
    > /tmp/eval_torques_gt_interhuman.log 2>&1 &
PID4=$!

echo "Waiting for GT torques to finish..."
wait $PID3
wait $PID4
echo "GT torques COMPLETE ✅"


# --- PHASE 2: Skyhook Metrics ---
echo ""
echo "==========================================="
echo " PHASE 2: Computing Skyhook Metrics"
echo "==========================================="

echo "[cuda:0] Computing Skyhook metrics for Generated InterX..."
OMP_NUM_THREADS=1 python prepare2/compute_skyhook_metrics.py \
    --dataset interx \
    --data-dir data/retargeted_v2/generated_interx \
    --output-dir data/skyhook_metrics/generated_interx \
    --downsample 1 --gpu cuda:0 \
    > /tmp/eval_skyhook_gen_interx.log 2>&1 &
PID5=$!

echo "[cuda:1] Computing Skyhook metrics for Generated InterHuman..."
OMP_NUM_THREADS=1 python prepare2/compute_skyhook_metrics.py \
    --dataset interhuman \
    --data-dir data/retargeted_v2/generated_interhuman \
    --output-dir data/skyhook_metrics/generated_interhuman \
    --downsample 1 --gpu cuda:1 \
    > /tmp/eval_skyhook_gen_interhuman.log 2>&1 &
PID6=$!

echo "Waiting for Generated skyhook to finish..."
wait $PID5
wait $PID6
echo "Generated Skyhook metrics COMPLETE ✅"


echo "[cuda:0] Computing Skyhook metrics for GT InterX..."
OMP_NUM_THREADS=1 python prepare2/compute_skyhook_metrics.py \
    --dataset interx \
    --output-dir data/skyhook_metrics/gt_interx \
    --gpu cuda:0 \
    > /tmp/eval_skyhook_gt_interx.log 2>&1 &
PID7=$!

echo "[cuda:1] Computing Skyhook metrics for GT InterHuman..."
OMP_NUM_THREADS=1 python prepare2/compute_skyhook_metrics.py \
    --dataset interhuman \
    --output-dir data/skyhook_metrics/gt_interhuman \
    --gpu cuda:1 \
    > /tmp/eval_skyhook_gt_interhuman.log 2>&1 &
PID8=$!

echo "Waiting for GT skyhook to finish..."
wait $PID7
wait $PID8
echo "GT Skyhook metrics COMPLETE ✅"

echo ""
echo "ALL EVALUATION PIPELINE PHYSICS STEPS COMPLETE 🎉"
