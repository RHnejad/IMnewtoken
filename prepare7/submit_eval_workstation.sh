#!/bin/bash
# PP-Motion evaluation on workstation GPU (WORKSTATION VERSION).
#
# Usage:
#   cd /media/rh/codes/sim/IMnewtoken
#   bash prepare7/submit_eval_workstation.sh test    # quick test first
#   bash prepare7/submit_eval_workstation.sh gt      # evaluate GT motions
#   bash prepare7/submit_eval_workstation.sh gen     # evaluate generated motions
#   bash prepare7/submit_eval_workstation.sh both    # gt + gen + comparison
#
# Set CONDA_ENV to override the conda environment name:
#   CONDA_ENV=myenv bash prepare7/submit_eval_workstation.sh gt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$SCRIPT_DIR/ProtoMotions"
OUTPUT_DIR="$SCRIPT_DIR/output"
CHECKPOINT="$PROTO_DIR/data/pretrained_models/motion_tracker/smpl/last.ckpt"
NUM_ENVS="${NUM_ENVS:-64}"

# --- Logging helpers ---
mkdir -p "$OUTPUT_DIR"
MASTER_LOG="$OUTPUT_DIR/submit_eval_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

log_separator() {
    log "========================================"
}

# --- Environment activation ---
CONDA_SH="/home/rh/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="${CONDA_ENV:-protomotions}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$ENV_NAME"

    # EGL/GLVND setup for headless MuJoCo rendering
    if [ -d "$CONDA_PREFIX/lib" ]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    fi
    local EGL_JSON="$CONDA_PREFIX/share/glvnd/egl_vendor.d/10_nvidia.json"
    if [ -f "$EGL_JSON" ]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_JSON"
    fi

    log "Python: $(which python) ($(python --version))"
    log "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())' 2>&1)"
    log "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>&1)"

    # Check critical deps; if missing, run setup
    if ! python -c "import lightning, newton, warp" 2>/dev/null; then
        log "ERROR: Missing dependencies. Run first: bash prepare7/setup_deps_workstation.sh"
        exit 1
    fi
}

# --- Eval runner (single batch) ---
BATCH_SIZE="${BATCH_SIZE:-2000}"

run_eval_single() {
    local motion_dir="$1"
    local output_json="$2"
    local label="$3"

    local num_motions
    num_motions=$(ls "$motion_dir"/*.motion 2>/dev/null | wc -l)

    log "  [$label] $num_motions motions in $motion_dir"

    if [ "$num_motions" -eq 0 ]; then
        log "ERROR: No .motion files found in $motion_dir"
        return 1
    fi

    local effective_envs=$NUM_ENVS
    if [ "$num_motions" -lt "$NUM_ENVS" ]; then
        effective_envs=$num_motions
    fi

    cd "$PROTO_DIR"

    python "$SCRIPT_DIR/run_evaluation.py" \
        --checkpoint "$CHECKPOINT" \
        --simulator newton \
        --full-eval --headless \
        --num-envs "$effective_envs" \
        --motion-file "$motion_dir" \
        --save-per-motion "$output_json" 2>&1 | tee -a "$MASTER_LOG"
}

# --- Batched eval runner ---
run_eval_batched() {
    local motion_dir="$1"
    local output_base="$2"
    local label="$3"

    local num_motions
    num_motions=$(ls "$motion_dir"/*.motion 2>/dev/null | wc -l)
    
    log_separator
    log "[$label] Starting batched evaluation"
    log "  Motion dir: $motion_dir"
    log "  Total motions: $num_motions"
    log "  Batch size: $BATCH_SIZE"
    log "  Output base: $output_base"
    log_separator

    if [ "$num_motions" -eq 0 ]; then
        log "ERROR: No .motion files found in $motion_dir"
        return 1
    fi

    # Create temp dir for batches
    local batch_dir="$OUTPUT_DIR/batches_${label}"
    mkdir -p "$batch_dir"

    # Run batched evaluation
    local batch_num=0
    local start_idx=0
    
    while [ "$start_idx" -lt "$num_motions" ]; do
        local end_idx=$((start_idx + BATCH_SIZE))
        if [ "$end_idx" -gt "$num_motions" ]; then
            end_idx=$num_motions
        fi
        
        local batch_output="$batch_dir/batch_${batch_num}.json"
        
        log "Running batch $batch_num: motions $start_idx-$((end_idx-1))"
        
        cd "$PROTO_DIR"
        python "$SCRIPT_DIR/run_evaluation.py" \
            --checkpoint "$CHECKPOINT" \
            --simulator newton \
            --full-eval --headless \
            --num-envs "$NUM_ENVS" \
            --motion-file "$motion_dir" \
            --save-per-motion "$batch_output" \
            --start-idx "$start_idx" \
            --end-idx "$end_idx" 2>&1 | tee -a "$MASTER_LOG"
        
        batch_num=$((batch_num + 1))
        start_idx=$end_idx
    done

    # Merge batches
    log "Merging $batch_num batches..."
    python "$SCRIPT_DIR/merge_eval_batches.py" \
        --input-dir "$batch_dir" \
        --output-json "$output_base" 2>&1 | tee -a "$MASTER_LOG"
    
    log "[$label] Evaluation complete: $output_base"
}

# --- Main ---
GT_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gt_motions"
GEN_MOTION_DIR="$SCRIPT_DIR/data/interhuman_gen_motions"
TEST_MOTION_DIR="$SCRIPT_DIR/data/interhuman_test"

log_separator
log "PP-Motion Evaluation Pipeline (Workstation)"
log "Command: $0 $*"
log_separator

activate_env

case "${1:-help}" in
    test)
        log "=== Running quick test (6 motions) ==="
        run_eval_single "$TEST_MOTION_DIR" "$OUTPUT_DIR/eval_test.json" "test"
        log "Test complete. Check: $OUTPUT_DIR/eval_test.json"
        ;;

    gt)
        log "=== Evaluating GT motions ==="
        run_eval_batched "$GT_MOTION_DIR" "$OUTPUT_DIR/eval_gt.json" "gt"
        ;;

    gen)
        log "=== Evaluating generated motions ==="
        run_eval_batched "$GEN_MOTION_DIR" "$OUTPUT_DIR/eval_generated.json" "gen"
        ;;

    both)
        log "=== Evaluating both GT and generated ==="
        run_eval_batched "$GT_MOTION_DIR" "$OUTPUT_DIR/eval_gt.json" "gt"
        run_eval_batched "$GEN_MOTION_DIR" "$OUTPUT_DIR/eval_generated.json" "gen"
        
        log "=== Comparing results ==="
        python "$SCRIPT_DIR/compare_results.py" \
            --gt-json "$OUTPUT_DIR/eval_gt.json" \
            --gen-json "$OUTPUT_DIR/eval_generated.json" \
            --output-dir "$OUTPUT_DIR/comparison" 2>&1 | tee -a "$MASTER_LOG"
        ;;

    record)
        log "=== Recording videos from motions ==="
        MOTION_DIR="${2:-$TEST_MOTION_DIR}"
        NUM_RECORD="${3:-3}"
        RENDERER="${4:-skeleton}"
        
        log "  Motion dir: $MOTION_DIR"
        log "  Num videos: $NUM_RECORD"
        log "  Renderer:   $RENDERER"
        
        cd "$REPO_ROOT"
        python "$SCRIPT_DIR/record_video.py" \
            --motion-dir "$MOTION_DIR" \
            --output-dir "$OUTPUT_DIR/videos" \
            --num-motions "$NUM_RECORD" \
            --renderer "$RENDERER" 2>&1 | tee -a "$MASTER_LOG"
        
        log "Videos saved to $OUTPUT_DIR/videos/"
        ;;

    *)
        echo "Usage: $0 {test|gt|gen|both|record}"
        echo ""
        echo "  test   - Quick test with 6 motions"
        echo "  gt     - Evaluate all GT motions"
        echo "  gen    - Evaluate all generated motions"
        echo "  both   - Evaluate both and compare"
        echo "  record - Record MP4 videos from .motion files"
        echo "           record [motion_dir] [num_motions] [newton|skeleton|mujoco]"
        echo ""
        echo "Examples:"
        echo "  $0 record                                      # 3 test motions, skeleton renderer"
        echo "  $0 record prepare7/data/interhuman_gt_motions 5 newton"
        exit 1
        ;;
esac

log_separator
log "Done! Master log: $MASTER_LOG"
log_separator
