#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <interhuman|interx> <clip>" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="$1"
CLIP="$2"
PYTHON_BIN="${PYTHON_BIN:-python}"
BODY_MODEL_PATH="${BODY_MODEL_PATH:-$ROOT/data/body_model/smplx/SMPLX_NEUTRAL.npz}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/diagnostics/$DATASET}"
mkdir -p "$OUTPUT_DIR"

if [[ "$DATASET" == "interhuman" ]]; then
    DATA_ROOT="${DATA_ROOT:-$ROOT/data/InterHuman}"
    EXTRA_DATASET_ARGS=()
    if [[ -n "${BETAS_FROM_INTERHUMAN_ROOT:-}" ]]; then
        EXTRA_DATASET_ARGS+=( --betas-from-interhuman-root "$BETAS_FROM_INTERHUMAN_ROOT" )
    fi
else
    DATA_ROOT="${DATA_ROOT:-$ROOT/data/Inter-X_Dataset}"
    EXTRA_DATASET_ARGS=()
    if [[ -n "${H5_FILE:-}" ]]; then
        EXTRA_DATASET_ARGS+=( --h5-file "$H5_FILE" )
    fi
fi

JSON_PATH="$OUTPUT_DIR/${CLIP}.json"
DETAILS_PATH="$OUTPUT_DIR/${CLIP}_details.pkl"
PLY_DIR="$OUTPUT_DIR/${CLIP}_ply"
RENDER_DIR="$OUTPUT_DIR/render"

"$PYTHON_BIN" prepare_mesh_contact/mesh_contact_pipeline.py \
    --dataset "$DATASET" \
    --clip "$CLIP" \
    --data-root "$DATA_ROOT" \
    --body-model-path "$BODY_MODEL_PATH" \
    --device "$DEVICE" \
    --output-json "$JSON_PATH" \
    --output-details "$DETAILS_PATH" \
    --export-ply-frame 0 \
    --export-ply-dir "$PLY_DIR" \
    "${EXTRA_DATASET_ARGS[@]}"

"$PYTHON_BIN" prepare_mesh_contact/render_contact_headless.py \
    --dataset "$DATASET" \
    --clip "$CLIP" \
    --data-root "$DATA_ROOT" \
    --body-model-path "$BODY_MODEL_PATH" \
    --device "$DEVICE" \
    --frames-per-clip 1 \
    --frame-policy representative \
    --out-dir "$RENDER_DIR" \
    --show-caption \
    "${EXTRA_DATASET_ARGS[@]}"

"$PYTHON_BIN" prepare_mesh_contact/visualize_contact_newton.py \
    --dataset "$DATASET" \
    --clip "$CLIP" \
    --data-root "$DATA_ROOT" \
    --body-model-path "$BODY_MODEL_PATH" \
    --device "$DEVICE" \
    --dry-run \
    "${EXTRA_DATASET_ARGS[@]}"
