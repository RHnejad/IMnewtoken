#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-${DATASET:-interhuman}}"
INTERVAL="${INTERVAL:-30}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

while true; do
    clear || true
    date
    echo
    prepare_mesh_contact/check_progress.sh "$DATASET"
    echo
    sleep "$INTERVAL"
done
