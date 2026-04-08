#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Batch mesh-contact extraction for InterHuman (7810 clips).
#
# Usage (inside intintermask RunAI node):
#   cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
#   bash prepare_mesh_contact/run_interhuman_batch.sh            # all clips, 4 workers
#   bash prepare_mesh_contact/run_interhuman_batch.sh --workers 8
#   bash prepare_mesh_contact/run_interhuman_batch.sh --shard 0 --num-shards 3  # for multi-node
#
# Parallelism:
#   --workers N     Number of parallel processes on THIS invocation (default: 4)
#   --shard K       Which shard this invocation handles (0-indexed)
#   --num-shards M  Total number of shards (for running script multiple times)
#   --data-root P   InterHuman source root (GT default: data/InterHuman)
#   --output-dir P  Output JSON directory (GT default: output/mesh_contact/interhuman)
#   --betas-from-interhuman-root P
#                   Optional GT InterHuman root used to replace betas (for generated clips)
#
# If you run the script N times with --shard 0..N-1 --num-shards N, each
# invocation handles a disjoint slice of clips → safe concurrent execution.
#
# Idempotent: clips whose output JSON already exists are skipped.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
WORKERS=4
SHARD=0
NUM_SHARDS=1
DEVICE="cuda"
BATCH_SIZE=64
SELF_PEN_MODE="off"
SAVE_DETAILS=0   # set to 1 to also save per-frame .pkl details (large!)
DATA_ROOT="data/InterHuman"
OUTPUT_DIR="output/mesh_contact/interhuman"
BETAS_FROM_INTERHUMAN_ROOT=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)      WORKERS="$2";      shift 2 ;;
        --shard)        SHARD="$2";        shift 2 ;;
        --num-shards)   NUM_SHARDS="$2";   shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";   shift 2 ;;
        --data-root)    DATA_ROOT="$2";    shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --betas-from-interhuman-root) BETAS_FROM_INTERHUMAN_ROOT="$2"; shift 2 ;;
        --self-pen)     SELF_PEN_MODE="$2"; shift 2 ;;
        --save-details) SAVE_DETAILS=1;    shift ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
BODY_MODEL="data/body_model/smplx/SMPLX_NEUTRAL.npz"
LOG_DIR="output/mesh_contact/logs"
PIPELINE="prepare_mesh_contact/mesh_contact_pipeline.py"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Conda ─────────────────────────────────────────────────────────────────────
CONDA_BIN="${CONDA_BIN:-/mnt/vita/scratch/vita-staff/users/rh/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-intermask}"
CONDA_SH="${CONDA_SH:-$(dirname "$(dirname "$CONDA_BIN")")/etc/profile.d/conda.sh}"

if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
elif [ -f "$CONDA_BIN" ]; then
    export PATH="$(dirname "$CONDA_BIN"):$PATH"
fi

if ! command -v conda &>/dev/null; then
    echo "Conda not found. Set CONDA_BIN or CONDA_SH to a valid installation." >&2
    exit 1
fi

conda activate "$CONDA_ENV"
PYTHON="${PYTHON:-python}"

# ── Build clip list ───────────────────────────────────────────────────────────
ALL_CLIPS_FILE="$LOG_DIR/interhuman_all_clips.txt"
if [ -d "$DATA_ROOT/motions" ]; then
    find "$DATA_ROOT/motions" -maxdepth 1 -type f -name "*.pkl" -printf "%f\n" | sed 's/\.pkl$//' | sort -n > "$ALL_CLIPS_FILE"
else
    find "$DATA_ROOT" -maxdepth 1 -type f -name "*.pkl" -printf "%f\n" | sed 's/\.pkl$//' | sort -n > "$ALL_CLIPS_FILE"
fi
TOTAL=$(wc -l < "$ALL_CLIPS_FILE")
echo "InterHuman total clips: $TOTAL"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
if [ -n "$BETAS_FROM_INTERHUMAN_ROOT" ]; then
    echo "GT beta override root: $BETAS_FROM_INTERHUMAN_ROOT"
fi

# ── Shard ─────────────────────────────────────────────────────────────────────
SHARD_FILE="$LOG_DIR/interhuman_shard_${SHARD}_of_${NUM_SHARDS}.txt"
awk -v shard="$SHARD" -v nshards="$NUM_SHARDS" '(NR-1) % nshards == shard' \
    "$ALL_CLIPS_FILE" > "$SHARD_FILE"
SHARD_COUNT=$(wc -l < "$SHARD_FILE")
echo "Shard $SHARD/$NUM_SHARDS: $SHARD_COUNT clips"
echo "Workers: $WORKERS | Device: $DEVICE | Self-pen: $SELF_PEN_MODE"
if [[ "$DEVICE" == cuda* ]] && [ "$WORKERS" -gt 1 ]; then
    echo "WARNING: running $WORKERS workers on $DEVICE may cause GPU OOM. If failures spike, retry with --workers 1 or 2."
fi

# ── Lock directory (clean stale locks older than 30 min) ─────────────────────
LOCK_DIR="$LOG_DIR/locks_interhuman"
mkdir -p "$LOCK_DIR"
find "$LOCK_DIR" -maxdepth 1 -name "*.lock" -type d -mmin +30 -exec rmdir {} \; 2>/dev/null || true

# ── Worker function ───────────────────────────────────────────────────────────
process_clip() {
    local clip_id="$1"
    local json_out="$OUTPUT_DIR/${clip_id}.json"
    local lockfile="$LOCK_DIR/${clip_id}.lock"

    # Skip if already done
    if [ -f "$json_out" ]; then
        return 0
    fi

    # Atomic lock: mkdir is atomic on POSIX — if it fails, another worker owns it
    if ! mkdir "$lockfile" 2>/dev/null; then
        if [ -d "$lockfile" ]; then
            return 0
        fi
        echo "ERROR: failed to create lock dir for interhuman clip=$clip_id at $lockfile" >&2
        return 1
    fi
    # Clean up lock on exit (success or failure)
    trap "rmdir '$lockfile' 2>/dev/null" RETURN

    # Re-check after acquiring lock (another worker may have just finished)
    if [ -f "$json_out" ]; then
        return 0
    fi

    local -a extra_args=()
    local error_log="$LOG_DIR/interhuman_errors_shard${SHARD}.log"
    local detail_log="$LOG_DIR/interhuman_errors_shard${SHARD}.details.log"
    local clip_log="$LOG_DIR/interhuman_clip_${clip_id}_shard${SHARD}.tmp.log"
    if [ "$SAVE_DETAILS" = "1" ]; then
        extra_args+=(--output-details "$OUTPUT_DIR/${clip_id}_details.pkl")
    fi
    if [ -n "$BETAS_FROM_INTERHUMAN_ROOT" ]; then
        extra_args+=(--betas-from-interhuman-root "$BETAS_FROM_INTERHUMAN_ROOT")
    fi

    if ! $PYTHON "$PIPELINE" \
        --dataset interhuman \
        --clip "$clip_id" \
        --data-root "$DATA_ROOT" \
        --body-model-path "$BODY_MODEL" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --self-penetration-mode "$SELF_PEN_MODE" \
        --output-json "$json_out" \
        "${extra_args[@]}" \
        --quiet \
        > "$clip_log" 2>&1
    then
        echo "FAILED: interhuman clip=$clip_id" >> "$error_log"
        {
            echo "===== FAILED interhuman clip=$clip_id at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
            tail -n 80 "$clip_log"
            echo ""
        } >> "$detail_log"
    fi
    rm -f "$clip_log"
}
# Limit per-process cKDTree threads to avoid CPU oversubscription
TOTAL_CPUS=$(nproc 2>/dev/null || echo 16)
export CKDTREE_WORKERS=$(( TOTAL_CPUS / WORKERS ))
if [ "$CKDTREE_WORKERS" -lt 1 ]; then CKDTREE_WORKERS=1; fi
echo "cKDTree workers per process: $CKDTREE_WORKERS (${TOTAL_CPUS} CPUs / ${WORKERS} workers)"

export -f process_clip
export PYTHON PIPELINE DATA_ROOT BODY_MODEL OUTPUT_DIR LOG_DIR LOCK_DIR DEVICE BATCH_SIZE SELF_PEN_MODE SAVE_DETAILS SHARD CKDTREE_WORKERS BETAS_FROM_INTERHUMAN_ROOT

# ── Run ───────────────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
echo "Starting at $(date)"

if command -v parallel &>/dev/null; then
    parallel -j "$WORKERS" --bar --progress process_clip :::: "$SHARD_FILE"
else
    echo "(GNU parallel not found, falling back to bash worker pool)"
    RUNNING=0
    DISPATCHED=0
    while IFS= read -r clip_id; do
        process_clip "$clip_id" &
        RUNNING=$((RUNNING + 1))
        DISPATCHED=$((DISPATCHED + 1))
        if [ $((DISPATCHED % 200)) -eq 0 ]; then
            echo "Dispatched $DISPATCHED / $SHARD_COUNT clips..."
        fi
        if [ "$RUNNING" -ge "$WORKERS" ]; then
            wait -n || true
            RUNNING=$((RUNNING - 1))
        fi
    done < "$SHARD_FILE"

    while [ "$RUNNING" -gt 0 ]; do
        wait -n || true
        RUNNING=$((RUNNING - 1))
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ── Report ────────────────────────────────────────────────────────────────────
DONE_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" | wc -l)
ERROR_LOG="$LOG_DIR/interhuman_errors_shard${SHARD}.log"
ERROR_COUNT=0
if [ -f "$ERROR_LOG" ]; then
    ERROR_COUNT=$(wc -l < "$ERROR_LOG")
fi

echo ""
echo "════════════════════════════════════════════════"
echo "InterHuman batch complete (shard $SHARD/$NUM_SHARDS)"
echo "  Completed JSONs: $DONE_COUNT / $TOTAL"
echo "  Errors:          $ERROR_COUNT"
echo "  Elapsed:         ${ELAPSED}s"
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "  Error log:       $ERROR_LOG"
fi
echo "════════════════════════════════════════════════"
