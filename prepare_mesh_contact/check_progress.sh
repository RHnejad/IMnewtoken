#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-${DATASET:-interhuman}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$DATASET" == "interhuman" ]]; then
    DATA_ROOT="${DATA_ROOT:-$ROOT/data/InterHuman}"
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/interhuman}"
    if [[ -d "$DATA_ROOT/motions" ]]; then
        TOTAL=$(find "$DATA_ROOT/motions" -maxdepth 1 -type f -name '*.pkl' | wc -l)
    else
        TOTAL=$(find "$DATA_ROOT" -maxdepth 1 -type f -name '*.pkl' | wc -l)
    fi
else
    DATA_ROOT="${DATA_ROOT:-$ROOT/data/Inter-X_Dataset}"
    OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/mesh_contact/interx}"
    TOTAL=$("$PYTHON_BIN" prepare_mesh_contact/list_interx_clips.py --data-root "$DATA_ROOT" --count-only)
fi
COMPLETED=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)
DETAILS=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*_details.pkl' | wc -l)
FAILURES=0
if [[ -f "$OUTPUT_DIR/failed_clips.txt" ]]; then
    FAILURES=$(grep -v '^[[:space:]]*$' "$OUTPUT_DIR/failed_clips.txt" | sort -u | wc -l)
fi
PCT=0
if [[ "$TOTAL" -gt 0 ]]; then
    PCT=$(( 100 * COMPLETED / TOTAL ))
fi
printf 'dataset=%s\n' "$DATASET"
printf 'data_root=%s\n' "$DATA_ROOT"
printf 'output_dir=%s\n' "$OUTPUT_DIR"
printf 'completed=%s/%s (%s%%)\n' "$COMPLETED" "$TOTAL" "$PCT"
printf 'details=%s\n' "$DETAILS"
printf 'unique_failures=%s\n' "$FAILURES"
