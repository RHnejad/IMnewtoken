#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-${DATASET:-interhuman}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$DATASET" == "interhuman" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/interhuman}"
    BATCH_SCRIPT="prepare_mesh_contact/run_interhuman_batch.sh"
else
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/interx}"
    BATCH_SCRIPT="prepare_mesh_contact/run_interx_batch.sh"
fi
FAILURES_FILE="${FAILURES_FILE:-$OUTPUT_DIR/failed_clips.txt}"

if [[ ! -f "$FAILURES_FILE" ]]; then
    echo "No failures file found at $FAILURES_FILE" >&2
    exit 1
fi
TMP_CLIPS="$(mktemp)"
sort -u "$FAILURES_FILE" > "$TMP_CLIPS"
CLIPS_FILE="$TMP_CLIPS" "$BATCH_SCRIPT"
rm -f "$TMP_CLIPS"
