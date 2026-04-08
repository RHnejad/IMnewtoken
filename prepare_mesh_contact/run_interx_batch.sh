#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Batch mesh-contact extraction for Inter-X Dataset.
#
# Usage (inside intintermask RunAI node):
#   cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
#   bash prepare_mesh_contact/run_interx_batch.sh               # all clips, 4 workers
#   bash prepare_mesh_contact/run_interx_batch.sh --workers 8
#   bash prepare_mesh_contact/run_interx_batch.sh --shard 0 --num-shards 3
#
# First run generates the clip list from the H5 file (needs h5py).
# Subsequent runs reuse the cached list.
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
SAVE_DETAILS=0
H5_FILE="data/Inter-X_Dataset/processed/inter-x.h5"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)      WORKERS="$2";      shift 2 ;;
        --shard)        SHARD="$2";        shift 2 ;;
        --num-shards)   NUM_SHARDS="$2";   shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";   shift 2 ;;
        --self-pen)     SELF_PEN_MODE="$2"; shift 2 ;;
        --save-details) SAVE_DETAILS=1;    shift ;;
        --h5-file)      H5_FILE="$2";     shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="data/Inter-X_Dataset"
BODY_MODEL="data/body_model/smplx/SMPLX_NEUTRAL.npz"
OUTPUT_DIR="output/mesh_contact/interx"
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

# ── Build clip list (cached) ─────────────────────────────────────────────────
ALL_CLIPS_FILE="$LOG_DIR/interx_all_clips.txt"
if [ ! -f "$ALL_CLIPS_FILE" ]; then
    echo "Building InterX clip list from H5 (one-time)..."
    $PYTHON prepare_mesh_contact/list_interx_clips.py \
        --h5 "$H5_FILE" \
        --output "$ALL_CLIPS_FILE"
fi

TOTAL=$(wc -l < "$ALL_CLIPS_FILE")
echo "InterX total clips: $TOTAL"

# ── Shard ─────────────────────────────────────────────────────────────────────
SHARD_FILE="$LOG_DIR/interx_shard_${SHARD}_of_${NUM_SHARDS}.txt"
awk -v shard="$SHARD" -v nshards="$NUM_SHARDS" '(NR-1) % nshards == shard' \
    "$ALL_CLIPS_FILE" > "$SHARD_FILE"
SHARD_COUNT=$(wc -l < "$SHARD_FILE")
echo "Shard $SHARD/$NUM_SHARDS: $SHARD_COUNT clips"
echo "Workers: $WORKERS | Device: $DEVICE | Self-pen: $SELF_PEN_MODE"
if [[ "$DEVICE" == cuda* ]] && [ "$WORKERS" -gt 1 ]; then
    echo "WARNING: running $WORKERS workers on $DEVICE may cause GPU OOM. If failures spike, retry with --workers 1 or 2."
fi

# ── Lock directory (clean stale locks older than 30 min) ─────────────────────
LOCK_DIR="$LOG_DIR/locks_interx"
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
        echo "ERROR: failed to create lock dir for interx clip=$clip_id at $lockfile" >&2
        return 1
    fi
    # Clean up lock on exit (success or failure)
    trap "rmdir '$lockfile' 2>/dev/null" RETURN

    # Re-check after acquiring lock (another worker may have just finished)
    if [ -f "$json_out" ]; then
        return 0
    fi

    local detail_args=""
    local error_log="$LOG_DIR/interx_errors_shard${SHARD}.log"
    local detail_log="$LOG_DIR/interx_errors_shard${SHARD}.details.log"
    local clip_log="$LOG_DIR/interx_clip_${clip_id}_shard${SHARD}.tmp.log"
    if [ "$SAVE_DETAILS" = "1" ]; then
        detail_args="--output-details $OUTPUT_DIR/${clip_id}_details.pkl"
    fi

    if ! $PYTHON "$PIPELINE" \
        --dataset interx \
        --clip "$clip_id" \
        --data-root "$DATA_ROOT" \
        --h5-file "$H5_FILE" \
        --body-model-path "$BODY_MODEL" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --self-penetration-mode "$SELF_PEN_MODE" \
        --output-json "$json_out" \
        $detail_args \
        --quiet \
        > "$clip_log" 2>&1
    then
        echo "FAILED: interx clip=$clip_id" >> "$error_log"
        {
            echo "===== FAILED interx clip=$clip_id at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
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
export PYTHON PIPELINE DATA_ROOT BODY_MODEL OUTPUT_DIR LOG_DIR LOCK_DIR DEVICE BATCH_SIZE SELF_PEN_MODE SAVE_DETAILS SHARD H5_FILE CKDTREE_WORKERS

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
ERROR_LOG="$LOG_DIR/interx_errors_shard${SHARD}.log"
ERROR_COUNT=0
if [ -f "$ERROR_LOG" ]; then
    ERROR_COUNT=$(wc -l < "$ERROR_LOG")
fi

echo ""
echo "════════════════════════════════════════════════"
echo "InterX batch complete (shard $SHARD/$NUM_SHARDS)"
echo "  Completed JSONs: $DONE_COUNT / $TOTAL"
echo "  Errors:          $ERROR_COUNT"
echo "  Elapsed:         ${ELAPSED}s"
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "  Error log:       $ERROR_LOG"
fi
echo "════════════════════════════════════════════════"
