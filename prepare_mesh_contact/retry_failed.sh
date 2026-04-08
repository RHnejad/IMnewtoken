#!/usr/bin/env bash
# Retry all FAILED clips recorded in interhuman/interx error logs.
# Run inside the intintermask node (or any RunAI job with the intermask env).
#
# Usage:
#   bash prepare_mesh_contact/retry_failed.sh
#   bash prepare_mesh_contact/retry_failed.sh --dataset interhuman
#   bash prepare_mesh_contact/retry_failed.sh --dataset interx
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="all"
DEVICE="cuda"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --device)  DEVICE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Conda ─────────────────────────────────────────────────────────────────────
CONDA_BIN="${CONDA_BIN:-/mnt/vita/scratch/vita-staff/users/rh/miniconda3/bin/conda}"
CONDA_SH="${CONDA_SH:-$(dirname "$(dirname "$CONDA_BIN")")/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" || export PATH="$(dirname "$CONDA_BIN"):$PATH"
conda activate "${CONDA_ENV:-intermask}"
PYTHON="python"

BODY_MODEL="data/body_model/smplx/SMPLX_NEUTRAL.npz"
PIPELINE="prepare_mesh_contact/mesh_contact_pipeline.py"
LOG_DIR="output/mesh_contact/logs"

retry_interhuman() {
    local failed_clips
    # Extract clip IDs from error logs
    failed_clips=$(grep -h "FAILED: interhuman clip=" "$LOG_DIR"/interhuman_errors_shard*.log 2>/dev/null \
        | sed 's/FAILED: interhuman clip=//' | sort -u)

    if [ -z "$failed_clips" ]; then
        echo "No InterHuman failures found."
        return
    fi

    echo "InterHuman failed clips: $(echo "$failed_clips" | wc -w)"
    echo "$failed_clips"
    echo ""

    local ok=0 skip=0 fail=0
    while IFS= read -r clip_id; do
        [ -z "$clip_id" ] && continue
        local json_out="output/mesh_contact/interhuman/${clip_id}.json"
        if [ -f "$json_out" ]; then
            echo "  [skip] $clip_id (already done)"
            (( skip++ )) || true
            continue
        fi
        echo -n "  [retry] interhuman clip=$clip_id ... "
        if $PYTHON "$PIPELINE" \
            --dataset interhuman \
            --clip "$clip_id" \
            --data-root data/InterHuman \
            --body-model-path "$BODY_MODEL" \
            --device "$DEVICE" \
            --self-penetration-mode off \
            --output-json "$json_out" \
            --quiet 2>&1; then
            echo "OK"
            (( ok++ )) || true
        else
            echo "FAILED again"
            (( fail++ )) || true
        fi
    done <<< "$failed_clips"

    echo ""
    echo "InterHuman retry: ok=$ok  skip=$skip  still_failing=$fail"
}

retry_interx() {
    local failed_clips
    failed_clips=$(grep -h "FAILED: interx clip=" "$LOG_DIR"/interx_errors_shard*.log 2>/dev/null \
        | sed 's/FAILED: interx clip=//' | sort -u)

    if [ -z "$failed_clips" ]; then
        echo "No InterX failures found."
        return
    fi

    local H5_FILE="data/Inter-X_Dataset/processed/inter-x.h5"
    echo "InterX failed clips: $(echo "$failed_clips" | wc -w)"
    echo "$failed_clips"
    echo ""

    local ok=0 skip=0 fail=0
    while IFS= read -r clip_id; do
        [ -z "$clip_id" ] && continue
        local json_out="output/mesh_contact/interx/${clip_id}.json"
        if [ -f "$json_out" ]; then
            echo "  [skip] $clip_id (already done)"
            (( skip++ )) || true
            continue
        fi
        echo -n "  [retry] interx clip=$clip_id ... "
        if $PYTHON "$PIPELINE" \
            --dataset interx \
            --clip "$clip_id" \
            --data-root data/Inter-X_Dataset \
            --h5-file "$H5_FILE" \
            --body-model-path "$BODY_MODEL" \
            --device "$DEVICE" \
            --self-penetration-mode off \
            --output-json "$json_out" \
            --quiet 2>&1; then
            echo "OK"
            (( ok++ )) || true
        else
            echo "FAILED again"
            (( fail++ )) || true
        fi
    done <<< "$failed_clips"

    echo ""
    echo "InterX retry: ok=$ok  skip=$skip  still_failing=$fail"
}

case "$DATASET" in
    interhuman) retry_interhuman ;;
    interx)     retry_interx ;;
    all)        retry_interhuman; echo ""; retry_interx ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac
