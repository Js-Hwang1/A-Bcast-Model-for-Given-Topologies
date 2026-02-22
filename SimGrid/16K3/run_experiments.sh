#!/usr/bin/env bash
#
# run_experiments.sh — Greedy vs MPI_Bcast sweep on the 16K3 topology.
#
# Message sizes: 256B → 64MB  (powers of 4, covering small/medium/large)
# Chunk sizes:   tuned per message size so chunk ≈ 16 KB when possible,
#                with a minimum of 4 chunks to allow pipelining.
#
# Output: results/greedy_vs_mpi.csv

set -euo pipefail

PLATFORM="platform_16K3.xml"
HOSTFILE="hostfile"
NP=16
BINARY="./bcast_16K3"
HOST_SPEED="--cfg=smpi/host-speed:2000Gf"
QUIET="--log=root.thres:warning"       # suppress SimGrid info spam

OUTDIR="results"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/greedy_vs_mpi.csv"

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

# Choose nchunks so each chunk ≈ 16 KB, minimum 4 chunks.
choose_chunks() {
    local msg=$1
    local target_chunk=16384   # 16 KB
    local nc=$(( msg / target_chunk ))
    if [ "$nc" -lt 4 ]; then
        nc=4
    fi
    echo "$nc"
}

# ---- Header ----
echo "msg_bytes,nchunks,algorithm,time_sec,correct" > "$OUTFILE"

echo "=============================================="
echo "  Greedy vs MPI_Bcast  —  16K3 topology"
echo "=============================================="

for MSG in "${MSG_SIZES[@]}"; do
    NC=$(choose_chunks "$MSG")

    # Human-readable size
    if   [ "$MSG" -ge 1048576 ]; then
        HR="$(( MSG / 1048576 )) MB"
    elif [ "$MSG" -ge 1024 ]; then
        HR="$(( MSG / 1024 )) KB"
    else
        HR="${MSG} B"
    fi

    echo ""
    echo "--- Message size: $HR  ($MSG bytes, $NC chunks) ---"

    # ---- MPI_Bcast ----
    OUTPUT=$(smpirun -np $NP -platform "$PLATFORM" -hostfile "$HOSTFILE" \
        $HOST_SPEED $QUIET "$BINARY" mpi "$MSG" 2>/dev/null)

    MPI_TIME=$(echo "$OUTPUT" | grep '^time_sec' | awk '{print $3}')
    MPI_OK=$(echo "$OUTPUT"   | grep '^correct'  | awk '{print $3}')
    echo "  MPI_Bcast : ${MPI_TIME} s  (correct: ${MPI_OK})"
    echo "${MSG},0,mpi,${MPI_TIME},${MPI_OK}" >> "$OUTFILE"

    # ---- Greedy ----
    OUTPUT=$(smpirun -np $NP -platform "$PLATFORM" -hostfile "$HOSTFILE" \
        $HOST_SPEED $QUIET "$BINARY" greedy "$MSG" "$NC" 2>/dev/null)

    GR_TIME=$(echo "$OUTPUT"  | grep '^time_sec' | awk '{print $3}')
    GR_OK=$(echo "$OUTPUT"    | grep '^correct'  | awk '{print $3}')
    GR_RND=$(echo "$OUTPUT"   | grep '^rounds'   | awk '{print $3}')
    echo "  Greedy    : ${GR_TIME} s  (rounds: ${GR_RND}, correct: ${GR_OK})"
    echo "${MSG},${NC},greedy,${GR_TIME},${GR_OK}" >> "$OUTFILE"
done

echo ""
echo "=============================================="
echo "Results written to: $OUTFILE"
echo "=============================================="
