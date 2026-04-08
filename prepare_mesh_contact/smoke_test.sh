#!/usr/bin/env bash
# Smoke-test the mesh_contact_pipeline in the 'intermask' conda environment.
# Run this INSIDE the intintermask RunAI interactive node:
#
#   runai bash intintermask
#   cd /mnt/vita/scratch/vita-staff/users/rh/codes/2026/IMnewtoken
#   bash prepare_mesh_contact/smoke_test.sh
#
# The script:
#   A) Verifies imports (torch, scipy, h5py, numpy)
#   B) Runs a 20-frame smoke test on InterHuman clip 7605

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="intermask"
CONDA_BIN="${CONDA_BIN:-/mnt/vita/scratch/vita-staff/users/rh/miniconda3/bin/conda}"
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
PYTHON="python"

# ── A) Dependency check ────────────────────────────────────────────────────────
echo "=== A) Checking dependencies in env: ${CONDA_ENV} ==="
cat > /tmp/_smoke_depcheck.py <<'PYCHECK'
import importlib, sys
ok = True
for pkg in ("torch", "scipy", "h5py", "numpy"):
    try:
        m = importlib.import_module(pkg)
        print(f"  OK  {pkg:10s}  {m.__version__}")
    except ImportError as e:
        print(f"  FAIL {pkg}: {e}")
        ok = False
if not ok:
    sys.exit(1)
print("All imports OK.")
PYCHECK
${PYTHON} /tmp/_smoke_depcheck.py

# ── B) Verify data paths ───────────────────────────────────────────────────────
echo ""
echo "=== B) Verifying data paths ==="
for path in \
    "data/body_model/smplx/SMPLX_NEUTRAL.npz" \
    "data/InterHuman" \
    "data/Inter-X_Dataset"; do
    if [ -e "$path" ]; then
        echo "  OK  $path"
    else
        echo "  MISSING  $path"
        exit 1
    fi
done

# ── C) InterHuman smoke test (clip 7605, frames 0-20) ─────────────────────────
echo ""
echo "=== C) InterHuman smoke test: clip=7605, frames 0-20 ==="
mkdir -p output/mesh_contact

${PYTHON} prepare_mesh_contact/mesh_contact_pipeline.py \
  --dataset interhuman \
  --clip 7605 \
  --data-root data/InterHuman \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz \
  --frame-start 0 --frame-end 20 \
  --self-penetration-mode off \
  --output-json output/mesh_contact/interhuman_7605_smoke.json \
  --output-details output/mesh_contact/interhuman_7605_smoke_details.pkl

echo ""
echo "=== Smoke test PASSED ==="
echo "Output JSON:    output/mesh_contact/interhuman_7605_smoke.json"
echo "Output details: output/mesh_contact/interhuman_7605_smoke_details.pkl"
echo ""
echo "Quick summary:"
cat > /tmp/_smoke_print.py <<'PYPRINT'
import json, sys
with open("output/mesh_contact/interhuman_7605_smoke.json") as f:
    d = json.load(f)
print(f"  dataset : {d['dataset']}")
print(f"  clip    : {d['clip']}")
print(f"  frames  : {d['frame_range']['num_frames']}")
print(f"  summary : {json.dumps(d['summary'], indent=4)}")
PYPRINT
${PYTHON} /tmp/_smoke_print.py
