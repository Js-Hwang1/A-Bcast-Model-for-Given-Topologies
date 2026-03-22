#!/bin/bash
# dispatch.sh — Run jobs from a file with bounded parallelism.
# Usage: dispatch.sh <jobfile> <max_workers> <proj_dir>
set -uo pipefail
JOBFILE=$1; W=$2; cd "$3"
RUNNING=0
while IFS= read -r CMD; do
    eval "$CMD" &
    RUNNING=$((RUNNING + 1))
    if (( RUNNING >= W )); then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done < "$JOBFILE"
wait
