#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Run mesh-contact extraction for BOTH datasets (InterHuman + InterX).
#
# Usage:
#   bash prepare_mesh_contact/run_all_batch.sh                    # defaults
#   bash prepare_mesh_contact/run_all_batch.sh --workers 6
#   bash prepare_mesh_contact/run_all_batch.sh --shard 0 --num-shards 4
#
# All arguments are forwarded to both run_interhuman_batch.sh and
# run_interx_batch.sh.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════╗"
echo "║  Mesh-Contact Batch: InterHuman + InterX     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

echo "──── Phase 1: InterHuman ────"
bash "$DIR/run_interhuman_batch.sh" "$@"
echo ""

echo "──── Phase 2: InterX ────"
bash "$DIR/run_interx_batch.sh" "$@"
echo ""

echo "All done."
