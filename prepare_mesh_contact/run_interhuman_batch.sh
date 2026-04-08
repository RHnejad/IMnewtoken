#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/InterHuman}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/interhuman}"
BODY_MODEL_PATH="${BODY_MODEL_PATH:-$ROOT/data/body_model/smplx/SMPLX_NEUTRAL.npz}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-64}"
JOBS="${JOBS:-1}"
FAILURES_FILE="${FAILURES_FILE:-$OUTPUT_DIR/failed_clips.txt}"
CLIPS_FILE="${CLIPS_FILE:-}"
BETAS_FROM_INTERHUMAN_ROOT="${BETAS_FROM_INTERHUMAN_ROOT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"
: > "$FAILURES_FILE"

if [[ -n "$CLIPS_FILE" ]]; then
    mapfile -t CLIPS < <(grep -v '^[[:space:]]*$' "$CLIPS_FILE")
elif [[ -d "$DATA_ROOT/motions" ]]; then
    mapfile -t CLIPS < <(find "$DATA_ROOT/motions" -maxdepth 1 -type f -name '*.pkl' -printf '%f\n' | sed 's/\.pkl$//' | sort)
else
    mapfile -t CLIPS < <(find "$DATA_ROOT" -maxdepth 1 -type f -name '*.pkl' -printf '%f\n' | sed 's/\.pkl$//' | sort)
fi

run_clip() {
    local clip="$1"
    local json_path="$OUTPUT_DIR/${clip}.json"
    local details_path="$OUTPUT_DIR/${clip}_details.pkl"
    local extra_args=()
    if [[ -f "$json_path" ]]; then
        echo "[skip] $clip"
        return 0
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        extra_args=( ${EXTRA_ARGS} )
    fi
    local cmd=(
        "$PYTHON_BIN" prepare_mesh_contact/mesh_contact_pipeline.py
        --dataset interhuman
        --clip "$clip"
        --data-root "$DATA_ROOT"
        --body-model-path "$BODY_MODEL_PATH"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --output-json "$json_path"
        --output-details "$details_path"
    )
    if [[ -n "$BETAS_FROM_INTERHUMAN_ROOT" ]]; then
        cmd+=( --betas-from-interhuman-root "$BETAS_FROM_INTERHUMAN_ROOT" )
    fi
    cmd+=( "${extra_args[@]}" )
    if ! "${cmd[@]}"; then
        echo "$clip" >> "$FAILURES_FILE"
        echo "[fail] $clip" >&2
    else
        echo "[ok] $clip"
    fi
}
export -f run_clip
export ROOT PYTHON_BIN DATA_ROOT OUTPUT_DIR BODY_MODEL_PATH DEVICE BATCH_SIZE FAILURES_FILE BETAS_FROM_INTERHUMAN_ROOT EXTRA_ARGS
printf '%s\n' "${CLIPS[@]}" | xargs -I{} -P "$JOBS" bash -lc 'run_clip "$@"' _ {}
