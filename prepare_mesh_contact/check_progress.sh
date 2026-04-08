#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Check progress of mesh-contact batch extraction.
#
# Usage:
#   bash prepare_mesh_contact/check_progress.sh
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "═══════════════════════════════════════════════"
echo "  Mesh-Contact Progress Report"
echo "═══════════════════════════════════════════════"
echo ""

# ── InterHuman ────────────────────────────────────────────────────────────────
IH_TOTAL=$(find data/InterHuman/motions -maxdepth 1 -name "*.pkl" | wc -l)
IH_DONE=$(find output/mesh_contact/interhuman -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
IH_PCT=0
if [ "$IH_TOTAL" -gt 0 ]; then
    IH_PCT=$((IH_DONE * 100 / IH_TOTAL))
fi
echo "InterHuman:  $IH_DONE / $IH_TOTAL  ($IH_PCT%)"

IH_ERR_FILES=$(ls output/mesh_contact/logs/interhuman_errors_shard*.log 2>/dev/null)
IH_ERRORS=0
if [ -n "$IH_ERR_FILES" ]; then
    IH_ERRORS=$(cat $IH_ERR_FILES 2>/dev/null | wc -l)
fi
echo "  Errors:    $IH_ERRORS"

# ── InterX ────────────────────────────────────────────────────────────────────
IX_LIST="output/mesh_contact/logs/interx_all_clips.txt"
if [ -f "$IX_LIST" ]; then
    IX_TOTAL=$(wc -l < "$IX_LIST")
else
    IX_TOTAL="(clip list not yet generated)"
fi
IX_DONE=$(find output/mesh_contact/interx -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
if [[ "$IX_TOTAL" =~ ^[0-9]+$ ]] && [ "$IX_TOTAL" -gt 0 ]; then
    IX_PCT=$((IX_DONE * 100 / IX_TOTAL))
    echo ""
    echo "InterX:      $IX_DONE / $IX_TOTAL  ($IX_PCT%)"
else
    echo ""
    echo "InterX:      $IX_DONE / $IX_TOTAL"
fi

IX_ERR_FILES=$(ls output/mesh_contact/logs/interx_errors_shard*.log 2>/dev/null)
IX_ERRORS=0
if [ -n "$IX_ERR_FILES" ]; then
    IX_ERRORS=$(cat $IX_ERR_FILES 2>/dev/null | wc -l)
fi
echo "  Errors:    $IX_ERRORS"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "───────────────────────────────────────────────"
TOTAL_DONE=$((IH_DONE + IX_DONE))
echo "Total completed JSONs: $TOTAL_DONE"

# Show recent error samples
for logfile in output/mesh_contact/logs/*_errors_*.log; do
    if [ -f "$logfile" ]; then
        echo ""
        echo "Recent errors in $(basename $logfile):"
        tail -5 "$logfile" | sed 's/^/  /'
    fi
done
