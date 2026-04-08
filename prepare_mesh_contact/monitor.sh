#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Live monitor for mesh-contact batch extraction.
#
# Usage:
#   bash prepare_mesh_contact/monitor.sh           # refresh every 30s
#   bash prepare_mesh_contact/monitor.sh --interval 10
#   bash prepare_mesh_contact/monitor.sh --once    # single snapshot, no loop
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INTERVAL=30
ONCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval|-i) INTERVAL="$2"; shift 2 ;;
        --once)        ONCE=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# State file to track previous count for throughput
STATE_FILE="/tmp/.mesh_contact_monitor_state"

snapshot() {
    local NOW
    NOW=$(date +%s)

    # ── Counts ────────────────────────────────────────────────────────────────
    local IH_TOTAL IH_DONE IX_TOTAL IX_DONE
    IH_TOTAL=$(find data/InterHuman/motions -maxdepth 1 -name "*.pkl" 2>/dev/null | wc -l)
    IH_DONE=$(find output/mesh_contact/interhuman -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)

    local IX_LIST="output/mesh_contact/logs/interx_all_clips.txt"
    if [ -f "$IX_LIST" ]; then
        IX_TOTAL=$(wc -l < "$IX_LIST")
    else
        IX_TOTAL=0
    fi
    IX_DONE=$(find output/mesh_contact/interx -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)

    local TOTAL_DONE=$(( IH_DONE + IX_DONE ))
    local TOTAL=$(( IH_TOTAL + IX_TOTAL ))

    # ── Throughput + ETA ──────────────────────────────────────────────────────
    local RATE_STR="--"
    local ETA_STR="--"
    if [ -f "$STATE_FILE" ]; then
        local PREV_TIME PREV_DONE
        read -r PREV_TIME PREV_DONE < "$STATE_FILE"
        local DELTA_T=$(( NOW - PREV_TIME ))
        local DELTA_D=$(( TOTAL_DONE - PREV_DONE ))
        if [ "$DELTA_T" -gt 0 ] && [ "$DELTA_D" -ge 0 ]; then
            # clips per minute (scaled)
            local RATE_X10=$(( DELTA_D * 600 / DELTA_T ))   # ×10 to keep integer
            RATE_STR="${RATE_X10:0:-1}.${RATE_X10: -1} clips/min"
            local REMAINING=$(( TOTAL - TOTAL_DONE ))
            if [ "$DELTA_D" -gt 0 ] && [ "$REMAINING" -gt 0 ]; then
                local ETA_SECS=$(( REMAINING * DELTA_T / DELTA_D ))
                local ETA_H=$(( ETA_SECS / 3600 ))
                local ETA_M=$(( (ETA_SECS % 3600) / 60 ))
                ETA_STR="${ETA_H}h ${ETA_M}m"
            elif [ "$REMAINING" -eq 0 ]; then
                ETA_STR="done"
            fi
        fi
    fi
    echo "$NOW $TOTAL_DONE" > "$STATE_FILE"

    # ── Active locks (in-flight clips) ────────────────────────────────────────
    local IH_ACTIVE IX_ACTIVE
    IH_ACTIVE=$(find output/mesh_contact/logs/locks_interhuman -maxdepth 1 -name "*.lock" -type d 2>/dev/null | wc -l)
    IX_ACTIVE=$(find output/mesh_contact/logs/locks_interx -maxdepth 1 -name "*.lock" -type d 2>/dev/null | wc -l)

    # ── Errors ────────────────────────────────────────────────────────────────
    local IH_ERRORS=0 IX_ERRORS=0
    for f in output/mesh_contact/logs/interhuman_errors_shard*.log; do
        [ -f "$f" ] && IH_ERRORS=$(( IH_ERRORS + $(wc -l < "$f") ))
    done
    for f in output/mesh_contact/logs/interx_errors_shard*.log; do
        [ -f "$f" ] && IX_ERRORS=$(( IX_ERRORS + $(wc -l < "$f") ))
    done

    # ── Progress bars ─────────────────────────────────────────────────────────
    bar() {
        local done=$1 total=$2 width=30
        [ "$total" -eq 0 ] && { printf "[%-${width}s]" ""; return; }
        local filled=$(( done * width / total ))
        printf "[%-${width}s]" "$(printf '#%.0s' $(seq 1 $filled))"
    }

    local IH_PCT=0 IX_PCT=0 TOT_PCT=0
    [ "$IH_TOTAL" -gt 0 ] && IH_PCT=$(( IH_DONE * 100 / IH_TOTAL ))
    [ "$IX_TOTAL" -gt 0 ] && IX_PCT=$(( IX_DONE * 100 / IX_TOTAL ))
    [ "$TOTAL"    -gt 0 ] && TOT_PCT=$(( TOTAL_DONE * 100 / TOTAL ))

    # ── Render ────────────────────────────────────────────────────────────────
    clear
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║         Mesh-Contact Batch Monitor                   ║"
    printf "║  %-52s  ║\n" "$(date '+%Y-%m-%d %H:%M:%S')  [refresh: ${INTERVAL}s]"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
    printf "  InterHuman  $(bar $IH_DONE $IH_TOTAL) %4d / %4d  (%3d%%)  err=%-3d  active=%d\n" \
        "$IH_DONE" "$IH_TOTAL" "$IH_PCT" "$IH_ERRORS" "$IH_ACTIVE"
    printf "  InterX      $(bar $IX_DONE $IX_TOTAL) %4d / %4d  (%3d%%)  err=%-3d  active=%d\n" \
        "$IX_DONE" "$IX_TOTAL" "$IX_PCT" "$IX_ERRORS" "$IX_ACTIVE"
    echo ""
    printf "  Total       $(bar $TOTAL_DONE $TOTAL) %4d / %4d  (%3d%%)\n" \
        "$TOTAL_DONE" "$TOTAL" "$TOT_PCT"
    echo ""
    echo "  Throughput : $RATE_STR"
    echo "  ETA        : $ETA_STR"

    # Recent errors
    local ANY_ERRORS=$(( IH_ERRORS + IX_ERRORS ))
    if [ "$ANY_ERRORS" -gt 0 ]; then
        echo ""
        echo "  ── Recent errors ──────────────────────────────────────"
        for f in output/mesh_contact/logs/*_errors_shard*.log; do
            [ -f "$f" ] || continue
            local n
            n=$(wc -l < "$f")
            [ "$n" -eq 0 ] && continue
            printf "  %-50s  (%d lines)\n" "$(basename "$f")" "$n"
            tail -3 "$f" | sed 's/^/    /'
        done
    fi

    echo ""
    echo "  Press Ctrl+C to exit."
}

if [ "$ONCE" -eq 1 ]; then
    snapshot
else
    while true; do
        snapshot
        sleep "$INTERVAL"
    done
fi
