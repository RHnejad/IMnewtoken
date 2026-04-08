#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
BODY_MODEL_PATH="${BODY_MODEL_PATH:-$ROOT/data/body_model/smplx/SMPLX_NEUTRAL.npz}"
DEVICE="${DEVICE:-cpu}"

"$PYTHON_BIN" prepare_mesh_contact/mesh_contact_pipeline.py --self-test
"$PYTHON_BIN" prepare_mesh_contact/mesh_contact_pipeline.py \
    --dataset interhuman \
    --clip 7605 \
    --data-root "$ROOT/data/InterHuman" \
    --body-model-path "$BODY_MODEL_PATH" \
    --device "$DEVICE" \
    --frame-end 10 \
    --output-json "$ROOT/output/mesh_contact/smoke_interhuman_7605.json" \
    --output-details "$ROOT/output/mesh_contact/smoke_interhuman_7605_details.pkl"
"$PYTHON_BIN" prepare_mesh_contact/mesh_contact_pipeline.py \
    --dataset interx \
    --clip G039T007A025R000 \
    --data-root "$ROOT/data/Inter-X_Dataset" \
    --body-model-path "$BODY_MODEL_PATH" \
    --device "$DEVICE" \
    --frame-end 10 \
    --output-json "$ROOT/output/mesh_contact/smoke_interx_G039T007A025R000.json" \
    --output-details "$ROOT/output/mesh_contact/smoke_interx_G039T007A025R000_details.pkl"
