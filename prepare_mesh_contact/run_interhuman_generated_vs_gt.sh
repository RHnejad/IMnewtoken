#!/usr/bin/env bash
# InterHuman generated-vs-GT contact analysis orchestration.
#
# Runs:
#   1. GT InterHuman split top-up for missing JSONs
#   2. Generated InterHuman contact extraction with GT betas
#   3. GT-vs-generated comparison renders
#   4. CSV/Markdown summary reports
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKERS=4
DEVICE="cuda"
BATCH_SIZE=64
SHARD=0
NUM_SHARDS=1
GT_DATA_ROOT="data/InterHuman"
GENERATED_DATA_ROOT="/mnt/vita/scratch/vita-staff/users/rh/codes/2026/default_intermask/data/generated/interhuman"
GT_JSON_DIR="output/mesh_contact/interhuman"
GENERATED_JSON_DIR="output/mesh_contact/generated_interhuman"
COMPARISON_DIR="output/renders/interhuman_gt_vs_generated_contact"
REPORT_DIR="output/reports/interhuman_generated_vs_gt"
BODY_MODEL="data/body_model/smplx/SMPLX_NEUTRAL.npz"
RENDER_MAX_CLIPS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --shard) SHARD="$2"; shift 2 ;;
        --num-shards) NUM_SHARDS="$2"; shift 2 ;;
        --gt-data-root) GT_DATA_ROOT="$2"; shift 2 ;;
        --generated-data-root) GENERATED_DATA_ROOT="$2"; shift 2 ;;
        --gt-json-dir) GT_JSON_DIR="$2"; shift 2 ;;
        --generated-json-dir) GENERATED_JSON_DIR="$2"; shift 2 ;;
        --comparison-dir) COMPARISON_DIR="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --render-max-clips) RENDER_MAX_CLIPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

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

mkdir -p "$GT_JSON_DIR" "$GENERATED_JSON_DIR" "$COMPARISON_DIR" "$REPORT_DIR"

echo "Step 1/4: detecting missing GT InterHuman split JSONs"
mapfile -t MISSING_GT_CLIPS < <(
    GT_JSON_DIR="$GT_JSON_DIR" GT_DATA_ROOT="$GT_DATA_ROOT" "$PYTHON" - <<'PY'
import os

gt_json_dir = os.environ["GT_JSON_DIR"]
split_root = os.path.join(os.environ["GT_DATA_ROOT"], "split")

def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

train = set(read_ids(os.path.join(split_root, "train.txt")))
test = set(read_ids(os.path.join(split_root, "test.txt")))
covered = {
    os.path.splitext(name)[0]
    for name in os.listdir(gt_json_dir)
    if name.endswith(".json")
}
missing = sorted((train | test) - covered, key=lambda x: (len(x), x))
for clip_id in missing:
    print(clip_id)
PY
)

if [ "${#MISSING_GT_CLIPS[@]}" -eq 0 ]; then
    echo "GT InterHuman train/test coverage is already complete."
else
    echo "Missing GT InterHuman split clips: ${MISSING_GT_CLIPS[*]}"
    for clip_id in "${MISSING_GT_CLIPS[@]}"; do
        json_out="$GT_JSON_DIR/${clip_id}.json"
        if [ -f "$json_out" ]; then
            continue
        fi
        echo "  topping up GT clip $clip_id"
        "$PYTHON" prepare_mesh_contact/mesh_contact_pipeline.py \
            --dataset interhuman \
            --clip "$clip_id" \
            --data-root "$GT_DATA_ROOT" \
            --body-model-path "$BODY_MODEL" \
            --device "$DEVICE" \
            --batch-size "$BATCH_SIZE" \
            --self-penetration-mode off \
            --output-json "$json_out" \
            --quiet
    done
fi

echo "Step 2/4: generated InterHuman contact extraction with GT betas"
bash prepare_mesh_contact/run_interhuman_batch.sh \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --data-root "$GENERATED_DATA_ROOT" \
    --output-dir "$GENERATED_JSON_DIR" \
    --betas-from-interhuman-root "$GT_DATA_ROOT"

echo "Step 3/4: GT-vs-generated comparison renders"
RENDER_ARGS=()
if [ "$RENDER_MAX_CLIPS" -gt 0 ]; then
    RENDER_ARGS+=(--max-clips "$RENDER_MAX_CLIPS")
fi
"$PYTHON" prepare_mesh_contact/render_interhuman_generated_vs_gt.py \
    --data-root "$GT_DATA_ROOT" \
    --generated-data-root "$GENERATED_DATA_ROOT" \
    --gt-json-dir "$GT_JSON_DIR" \
    --generated-json-dir "$GENERATED_JSON_DIR" \
    --body-model-path "$BODY_MODEL" \
    --out-dir "$COMPARISON_DIR" \
    --device "$DEVICE" \
    --batch-size 1 \
    "${RENDER_ARGS[@]}"

echo "Step 4/4: summary tables and markdown report"
"$PYTHON" prepare_mesh_contact/summarize_interhuman_generated_vs_gt.py \
    --data-root "$GT_DATA_ROOT" \
    --gt-json-dir "$GT_JSON_DIR" \
    --generated-json-dir "$GENERATED_JSON_DIR" \
    --comparison-dir "$COMPARISON_DIR" \
    --out-dir "$REPORT_DIR"

echo ""
echo "InterHuman generated-vs-GT analysis complete."
echo "  GT JSON dir:         $GT_JSON_DIR"
echo "  Generated JSON dir:  $GENERATED_JSON_DIR"
echo "  Comparison PNG dir:  $COMPARISON_DIR"
echo "  Summary report dir:  $REPORT_DIR"
echo ""
echo "InterX GT completion command:"
echo "  cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken"
echo "  bash prepare_mesh_contact/run_interx_batch.sh --workers 4 --device cuda --batch-size 64"
