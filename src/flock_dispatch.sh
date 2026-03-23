#!/usr/bin/env bash
#
# flock_dispatch.sh — Dynamic flock-based work-stealing dispatcher.
#
# Multiple nodes can run this concurrently against the same shared task file.
# Uses flock + a cursor file for atomic job claiming.
#
# Usage:
#   flock_dispatch.sh <task_file> [-j workers]
#
# The cursor file is <task_file>.cursor (must be initialized to 1 before first use).
#

set -euo pipefail

TASK_FILE=""
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs) JOBS="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,/^$/{ s/^# \{0,1\}//; p }' "$0"
            exit 0 ;;
        *)
            if [[ -z "$TASK_FILE" ]]; then
                TASK_FILE="$1"; shift
            else
                echo "Unknown argument: $1" >&2; exit 1
            fi
            ;;
    esac
done

if [[ -z "$TASK_FILE" || ! -f "$TASK_FILE" ]]; then
    echo "ERROR: task file not found: ${TASK_FILE:-<none>}" >&2
    exit 1
fi

CURSOR_FILE="${TASK_FILE}.cursor"
LOCK_FILE="${TASK_FILE}.lock"
TOTAL=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [[ ! -f "$CURSOR_FILE" ]]; then
    echo "ERROR: cursor file not found: $CURSOR_FILE (initialize with: echo 1 > $CURSOR_FILE)" >&2
    exit 1
fi

echo "[flock_dispatch] task_file=$TASK_FILE  total=$TOTAL  workers=$JOBS  node=${SLURM_NODEID:-local}"

# claim_next: atomically read and increment cursor, return the line number or -1 if done
claim_next() {
    local line_num
    (
        flock 9
        line_num=$(cat "$CURSOR_FILE")
        if (( line_num > TOTAL )); then
            echo -1
        else
            echo "$line_num"
            echo $(( line_num + 1 )) > "$CURSOR_FILE"
        fi
    ) 9>"$LOCK_FILE"
}

RUNNING=0
COMPLETED=0
FAILED=0
PIDS=()
LINE_NUMS=()

reap_one() {
    # Wait for any one child to finish
    local pid status
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null && status=0 || status=$?
            COMPLETED=$((COMPLETED + 1))
            (( status != 0 )) && FAILED=$((FAILED + 1))
            (( status != 0 )) && echo "[flock_dispatch] FAILED (exit=$status) line=${LINE_NUMS[$i]}"
            unset 'PIDS[$i]'
            unset 'LINE_NUMS[$i]'
            RUNNING=$((RUNNING - 1))
            return
        fi
    done
    # All still running — brief sleep and retry
    sleep 0.1
    reap_one
}

while true; do
    # Fill worker slots
    while (( RUNNING < JOBS )); do
        LINE_NUM=$(claim_next)
        if (( LINE_NUM < 0 )); then
            break 2  # no more tasks
        fi
        CMD=$(sed -n "${LINE_NUM}p" "$TASK_FILE")
        if [[ -z "$CMD" ]]; then
            continue
        fi
        bash -c "$CMD" &
        PIDS+=($!)
        LINE_NUMS+=($LINE_NUM)
        RUNNING=$((RUNNING + 1))
    done

    # Wait for one to finish before claiming more
    if (( RUNNING > 0 )); then
        reap_one
    else
        break
    fi
done

# Drain remaining
while (( RUNNING > 0 )); do
    reap_one
done

echo "[flock_dispatch] Done. completed=$COMPLETED  failed=$FAILED  node=${SLURM_NODEID:-local}"
(( FAILED > 0 )) && exit 1
exit 0
