#!/bin/bash
# PP-Motion evaluation on RunAI GPU node.
#
# Usage (inside runai bash intintermask):
#   cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
#   bash prepare7/submit_eval.sh test    # quick test first
#   bash prepare7/submit_eval.sh gt      # evaluate GT motions
#   bash prepare7/submit_eval.sh gen     # evaluate generated motions
#   bash prepare7/submit_eval.sh both    # gt + gen + comparison
#
# Set CONDA_ENV to override the conda environment name:
#   CONDA_ENV=myenv bash prepare7/submit_eval.sh gt

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
# Persistent conda env on shared storage (survives container restarts)
CONDA_SH="/mnt/vita/scratch/vita-staff/users/rh/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="${CONDA_ENV:-protomotions}"

activate_env() {
    source "$CONDA_SH"
    conda activate "$ENV_NAME"

    # RunAI containers may lack a passwd entry for the current UID.
    export USER="${USER:-$(id -u)}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$$}"

    # EGL/GLVND setup for headless MuJoCo rendering (installed by setup_deps.sh)
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
        log "ERROR: Missing dependencies. Run first: bash prepare7/setup_deps.sh"
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
        --save-per-motion "$output_json" \
        2>&1 | tee "${output_json%.json}.log"
}

# --- Batched eval runner (splits large dirs to avoid OOM) ---
run_eval() {
    local motion_dir="$1"
    local output_json="$2"
    local label="$3"

    local num_motions
    num_motions=$(ls "$motion_dir"/*.motion 2>/dev/null | wc -l)

    log_separator
    log "  PP-Motion Eval: $label"
    log "  Motions: $motion_dir"
    log "  Output:  $output_json"
    log "  Envs:    $NUM_ENVS"
    log "  Batch:   $BATCH_SIZE"
    log "  Master log: $MASTER_LOG"
    log_separator
    log "Found $num_motions .motion files"

    if [ "$num_motions" -eq 0 ]; then
        log "ERROR: No .motion files found in $motion_dir"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    if [ "$num_motions" -le "$BATCH_SIZE" ]; then
        # Small enough to run in one shot
        run_eval_single "$motion_dir" "$output_json" "$label"
    else
        # Split into batches using symlink subdirectories
        log "Splitting $num_motions motions into batches of $BATCH_SIZE ..."
        local batch_dir="$OUTPUT_DIR/batches_$$"
        rm -rf "$batch_dir"
        mkdir -p "$batch_dir"

        local files=("$motion_dir"/*.motion)
        local total=${#files[@]}
        local batch_idx=0
        local i=0

        while [ "$i" -lt "$total" ]; do
            local bdir="$batch_dir/batch_$(printf '%03d' $batch_idx)"
            mkdir -p "$bdir"
            local end=$((i + BATCH_SIZE))
            if [ "$end" -gt "$total" ]; then end=$total; fi

            for (( j=i; j<end; j++ )); do
                ln -s "${files[$j]}" "$bdir/"
            done

            local bcount=$(ls "$bdir"/*.motion | wc -l)
            log "  Batch $batch_idx: $bcount motions"
            batch_idx=$((batch_idx + 1))
            i=$end
        done

        local num_batches=$batch_idx
        log "Created $num_batches batches"

        # Run each batch
        local batch_jsons=()
        for (( b=0; b<num_batches; b++ )); do
            local bdir="$batch_dir/batch_$(printf '%03d' $b)"
            local bjson="$batch_dir/batch_$(printf '%03d' $b).json"
            log ""
            log "======== Batch $((b+1))/$num_batches ========"
            run_eval_single "$bdir" "$bjson" "$label batch $((b+1))/$num_batches"
            batch_jsons+=("$bjson")
        done

        # Merge batch results
        log ""
        log "Merging $num_batches batch results ..."
        python "$SCRIPT_DIR/merge_eval_batches.py" "$output_json" "${batch_jsons[@]}"

        # Clean up batch symlink dirs (keep JSONs for debugging)
        for (( b=0; b<num_batches; b++ )); do
            rm -rf "$batch_dir/batch_$(printf '%03d' $b)"
        done

        log ""
        log "Done. Merged results: $output_json"
    fi
}

MODE="${1:-help}"

case "$MODE" in
    gt)
        activate_env
        run_eval \
            "$SCRIPT_DIR/data/interhuman_gt_motions" \
            "$OUTPUT_DIR/eval_gt.json" \
            "GT InterHuman (15616 motions)"
        ;;
    gen)
        activate_env
        run_eval \
            "$SCRIPT_DIR/data/interhuman_gen_motions" \
            "$OUTPUT_DIR/eval_generated.json" \
            "Generated InterMask (2196 motions)"
        ;;
    test)
        activate_env
        run_eval \
            "$SCRIPT_DIR/data/interhuman_test" \
            "$OUTPUT_DIR/eval_test.json" \
            "Quick test"
        ;;
    both)
        activate_env
        bash "$0" gt
        bash "$0" gen
        log "=== Running comparison ==="
        cd "$REPO_ROOT"
        python "$SCRIPT_DIR/compare_results.py" \
            --gt-json "$OUTPUT_DIR/eval_gt.json" \
            --gen-json "$OUTPUT_DIR/eval_generated.json" \
            --output-dir "$OUTPUT_DIR/comparison"
        ;;
    record)
        activate_env
        MOTION_DIR="${2:-$SCRIPT_DIR/data/interhuman_test}"
        NUM_RECORD="${3:-3}"
        RENDERER="${4:-skeleton}"
        log "Recording $NUM_RECORD motions from $MOTION_DIR (renderer=$RENDERER)"
        cd "$REPO_ROOT"
        python "$SCRIPT_DIR/record_video.py" \
            --motion-dir "$MOTION_DIR" \
            --output-dir "$OUTPUT_DIR/videos" \
            --num-motions "$NUM_RECORD" \
            --renderer "$RENDERER"
        log "Videos saved to $OUTPUT_DIR/videos/"
        ;;
    help|*)
        echo "Usage: bash prepare7/submit_eval.sh <gt|gen|test|both|record>"
        echo ""
        echo "  gt      Evaluate GT InterHuman motions"
        echo "  gen     Evaluate generated InterMask motions"
        echo "  test    Quick test on a few clips"
        echo "  both    Run gt + gen + comparison"
        echo "  record  Record MP4 videos (headless, no display needed)"
        echo "          record [motion_dir] [num_motions] [newton|skeleton|mujoco]"
        echo ""
        echo "Environment variables:"
        echo "  CONDA_ENV  Conda environment name (default: protomotions)"
        echo "  NUM_ENVS   Parallel envs for simulation (default: 64)"
        echo ""
        echo "Run inside: runai bash intintermask"
        ;;
esac
