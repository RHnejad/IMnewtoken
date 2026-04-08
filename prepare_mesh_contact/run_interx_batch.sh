#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/Inter-X_Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/interx}"
BODY_MODEL_PATH="${BODY_MODEL_PATH:-$ROOT/data/body_model/smplx/SMPLX_NEUTRAL.npz}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-64}"
JOBS="${JOBS:-1}"
FAILURES_FILE="${FAILURES_FILE:-$OUTPUT_DIR/failed_clips.txt}"
CLIPS_FILE="${CLIPS_FILE:-}"
H5_FILE="${H5_FILE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR"
: > "$FAILURES_FILE"

if [[ -n "$CLIPS_FILE" ]]; then
    mapfile -t CLIPS < <(grep -v '^[[:space:]]*$' "$CLIPS_FILE")
else
    list_cmd=( "$PYTHON_BIN" prepare_mesh_contact/list_interx_clips.py --data-root "$DATA_ROOT" )
    if [[ -n "$H5_FILE" ]]; then
        list_cmd+=( --h5-file "$H5_FILE" )
    fi
    mapfile -t CLIPS < <("${list_cmd[@]}")
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
        --dataset interx
        --clip "$clip"
        --data-root "$DATA_ROOT"
        --body-model-path "$BODY_MODEL_PATH"
        --device "$DEVICE"
        --batch-size "$BATCH_SIZE"
        --output-json "$json_path"
        --output-details "$details_path"
    )
    if [[ -n "$H5_FILE" ]]; then
        cmd+=( --h5-file "$H5_FILE" )
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
export ROOT PYTHON_BIN DATA_ROOT OUTPUT_DIR BODY_MODEL_PATH DEVICE BATCH_SIZE FAILURES_FILE H5_FILE EXTRA_ARGS
printf '%s\n' "${CLIPS[@]}" | xargs -I{} -P "$JOBS" bash -lc 'run_clip "$@"' _ {}
