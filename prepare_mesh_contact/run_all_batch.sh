#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_INTERHUMAN="${RUN_INTERHUMAN:-1}"
RUN_INTERX="${RUN_INTERX:-1}"

if [[ "$RUN_INTERHUMAN" == "1" ]]; then
    prepare_mesh_contact/run_interhuman_batch.sh
fi
if [[ "$RUN_INTERX" == "1" ]]; then
    prepare_mesh_contact/run_interx_batch.sh
fi
