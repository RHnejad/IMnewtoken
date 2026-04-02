#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data2/rh/conda_envs/mimickit/bin/python}"
VIEWER="${VIEWER:-gl}"
INTERHUMAN_CLIP="${INTERHUMAN_CLIP:-7605}"
INTERX_CLIP="${INTERX_CLIP:-G039T007A025R000}"

echo "Launching InterHuman contact visualization (clip ${INTERHUMAN_CLIP})..."
"${PYTHON_BIN}" prepare_mesh_contact/visualize_contact_newton.py \
  --viewer "${VIEWER}" \
  --dataset interhuman \
  --clip "${INTERHUMAN_CLIP}" \
  --data-root data/InterHuman \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz

echo "Launching InterX contact visualization (clip ${INTERX_CLIP})..."
"${PYTHON_BIN}" prepare_mesh_contact/visualize_contact_newton.py \
  --viewer "${VIEWER}" \
  --dataset interx \
  --clip "${INTERX_CLIP}" \
  --data-root data/Inter-X_Dataset \
  --body-model-path data/body_model/smplx/SMPLX_NEUTRAL.npz
