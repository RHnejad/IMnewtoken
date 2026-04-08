#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_ROOT="${GENERATED_ROOT:-${1:-}}"
if [[ -z "$GENERATED_ROOT" ]]; then
    echo "Usage: GENERATED_ROOT=/path/to/generated_smplx_root $0" >&2
    exit 1
fi
GT_ROOT="${GT_ROOT:-$ROOT/data/InterHuman}"
OUT_DIR="${OUT_DIR:-$ROOT/output/mesh_contact/generated_vs_gt_renders}"
BODY_MODEL_PATH="${BODY_MODEL_PATH:-$ROOT/data/body_model/smplx/SMPLX_NEUTRAL.npz}"
DEVICE="${DEVICE:-cpu}"
MAX_CLIPS="${MAX_CLIPS:-0}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-1}"
FRAME_POLICY="${FRAME_POLICY:-representative}"
SHOW_CAPTION="${SHOW_CAPTION:-1}"

cmd=(
    "$PYTHON_BIN" prepare_mesh_contact/render_interhuman_generated_vs_gt.py
    --generated-root "$GENERATED_ROOT"
    --gt-root "$GT_ROOT"
    --out-dir "$OUT_DIR"
    --body-model-path "$BODY_MODEL_PATH"
    --device "$DEVICE"
    --frames-per-clip "$FRAMES_PER_CLIP"
    --frame-policy "$FRAME_POLICY"
)
if [[ "$MAX_CLIPS" -gt 0 ]]; then
    cmd+=( --max-clips "$MAX_CLIPS" )
fi
if [[ "$SHOW_CAPTION" == "1" ]]; then
    cmd+=( --show-caption )
fi
if [[ -n "${CLIP:-}" ]]; then
    cmd+=( --clip "$CLIP" )
fi
if [[ -n "${CLIPS_FILE:-}" ]]; then
    cmd+=( --clips-file "$CLIPS_FILE" )
fi
"${cmd[@]}"
