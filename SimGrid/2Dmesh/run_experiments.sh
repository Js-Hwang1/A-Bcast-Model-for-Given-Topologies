#!/usr/bin/env bash
#
# run_experiments.sh — Sweep across message sizes and (optionally) root nodes.
#
# Usage:
#   ./run_experiments.sh                    # default root=0
#   ./run_experiments.sh 0                  # single root
#   ./run_experiments.sh all                # all roots (0..N-1)
#   ./run_experiments.sh 0 5 10 15          # specific roots
#
# Output: results/bbs_greedy_mpi.csv  (or results/bbs_greedy_mpi_rootR.csv)

set -euo pipefail

PLATFORM="platform_2dmesh_4x4.xml"
HOSTFILE="hostfile"
NP=16
BINARY="./bcast_2dmesh"
HOST_SPEED="--cfg=smpi/host-speed:2000Gf"
QUIET="--log=root.thres:warning"

OUTDIR="results"
mkdir -p "$OUTDIR"

# ---- Parse root nodes ----
if [ $# -eq 0 ]; then
    ROOTS=(0)
elif [ "$1" = "all" ]; then
    ROOTS=($(seq 0 $((NP - 1))))
else
    ROOTS=("$@")
fi

# ---- Message sizes (bytes) ----
MSG_SIZES=(
    256
    1024
    4096
    16384
    65536
    262144
    1048576
    4194304
    16777216
    67108864
)

choose_chunks() {
    local msg=$1
    local target_chunk=16384
    local nc=$(( msg / target_chunk ))
    if [ "$nc" -lt 4 ]; then
        nc=4
    fi
    echo "$nc"
}

for ROOT in "${ROOTS[@]}"; do

    if [ ${#ROOTS[@]} -gt 1 ]; then
        OUTFILE="$OUTDIR/bbs_greedy_mpi_root${ROOT}.csv"
    else
        OUTFILE="$OUTDIR/bbs_greedy_mpi.csv"
    fi

    echo "msg_bytes,nchunks,algorithm,root,time_sec,rounds,correct" > "$OUTFILE"

    echo "=============================================="
    echo "  Tree vs BBS vs Greedy vs MPI_Bcast  —  root=$ROOT"
    echo "=============================================="

    for MSG in "${MSG_SIZES[@]}"; do
        NC=$(choose_chunks "$MSG")

        if   [ "$MSG" -ge 1048576 ]; then
            HR="$(( MSG / 1048576 )) MB"
        elif [ "$MSG" -ge 1024 ]; then
            HR="$(( MSG / 1024 )) KB"
        else
            HR="${MSG} B"
        fi

        echo ""
        echo "--- $HR ($MSG bytes, $NC chunks, root=$ROOT) ---"

        # ---- MPI_Bcast ----
        OUTPUT=$(smpirun -np $NP -platform "$PLATFORM" -hostfile "$HOSTFILE" \
            $HOST_SPEED $QUIET "$BINARY" mpi "$MSG" 0 "$ROOT" 2>/dev/null)

        MPI_TIME=$(echo "$OUTPUT" | grep '^time_sec' | awk '{print $3}')
        MPI_OK=$(echo "$OUTPUT"   | grep '^correct'  | awk '{print $3}')
        echo "  MPI_Bcast : ${MPI_TIME} s  (correct: ${MPI_OK})"
        echo "${MSG},0,mpi,${ROOT},${MPI_TIME},0,${MPI_OK}" >> "$OUTFILE"

        # ---- Tree Pipeline ----
        OUTPUT=$(smpirun -np $NP -platform "$PLATFORM" -hostfile "$HOSTFILE" \
            $HOST_SPEED $QUIET "$BINARY" tree "$MSG" "$NC" "$ROOT" 2>/dev/null)

        TR_TIME=$(echo "$OUTPUT"  | grep '^time_sec' | awk '{print $3}')
        TR_OK=$(echo "$OUTPUT"    | grep '^correct'  | awk '{print $3}')
        echo "  Tree      : ${TR_TIME} s  (correct: ${TR_OK})"
        echo "${MSG},${NC},tree,${ROOT},${TR_TIME},0,${TR_OK}" >> "$OUTFILE"

        # ---- Greedy ----
        OUTPUT=$(smpirun -np $NP -platform "$PLATFORM" -hostfile "$HOSTFILE" \
            $HOST_SPEED $QUIET "$BINARY" greedy "$MSG" "$NC" "$ROOT" 2>/dev/null)

        GR_TIME=$(echo "$OUTPUT"  | grep '^time_sec' | awk '{print $3}')
        GR_OK=$(echo "$OUTPUT"    | grep '^correct'  | awk '{print $3}')
        GR_RND=$(echo "$OUTPUT"   | grep '^rounds'   | awk '{print $3}')
        echo "  Greedy    : ${GR_TIME} s  (rounds: ${GR_RND}, correct: ${GR_OK})"
        echo "${MSG},${NC},greedy,${ROOT},${GR_TIME},${GR_RND},${GR_OK}" >> "$OUTFILE"

        # ---- BBS ----
        OUTPUT=$(smpirun -np $NP -platform "$PLATFORM" -hostfile "$HOSTFILE" \
            $HOST_SPEED $QUIET "$BINARY" bbs "$MSG" "$NC" "$ROOT" 2>/dev/null || true)

        if echo "$OUTPUT" | grep -q '^time_sec'; then
            BBS_TIME=$(echo "$OUTPUT" | grep '^time_sec' | awk '{print $3}')
            BBS_OK=$(echo "$OUTPUT"   | grep '^correct'  | awk '{print $3}')
            BBS_RND=$(echo "$OUTPUT"  | grep '^rounds'   | awk '{print $3}')
            echo "  BBS       : ${BBS_TIME} s  (rounds: ${BBS_RND}, correct: ${BBS_OK})"
            echo "${MSG},${NC},bbs,${ROOT},${BBS_TIME},${BBS_RND},${BBS_OK}" >> "$OUTFILE"
        else
            echo "  BBS       : [frames not yet populated]"
        fi
    done

    echo ""
    echo "Results written to: $OUTFILE"

done

echo ""
echo "=============================================="
echo "  Done."
echo "=============================================="
