#!/usr/bin/env bash
#
# run_experiments.sh — Sweep across topologies, node counts, algorithms, and
#                      message sizes.
#
# Usage:
#   ./run_experiments.sh              # full sweep
#   ./run_experiments.sh mpi srda     # only specific algorithms
#
# Output: ../results/<Topo>_<N>.csv
#
# Columns: topology,nodes,algorithm,msg_bytes,nchunks,root,time_sec,correct

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOPO_DIR="$SCRIPT_DIR/../topo"
RES_DIR="$SCRIPT_DIR/../results"
BINARY="$SCRIPT_DIR/../bin/runner"
HOST_SPEED="--cfg=smpi/host-speed:2000Gf"
QUIET="--log=root.thres:warning"

mkdir -p "$RES_DIR"

# ---- Topologies and sizes ----
TOPOS=(2Dmesh Butterfly Dragonfly FatTree)
SIZES=(128 256 512 1024)
ROOT=0

# ---- Algorithms (override with command-line args) ----
if [ $# -gt 0 ]; then
    ALGOS=("$@")
else
    ALGOS=(mpi srda)
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
    local nc=$(( msg / 16384 )); (( nc < 4 )) && nc=4; echo "$nc"
}

# ---- 2Dmesh dimension lookup ----
mesh_dims() {
    case "$1" in
        128)  echo "8x16" ;;
        256)  echo "16x16" ;;
        512)  echo "16x32" ;;
        1024) echo "32x32" ;;
    esac
}

# ---- Resolve platform XML path ----
find_platform() {
    local topo=$1
    local n=$2
    case "$topo" in
        2Dmesh)
            local dims
            dims=$(mesh_dims "$n")
            echo "$TOPO_DIR/2Dmesh/platform_2dmesh_${dims}.xml"
            ;;
        Butterfly)
            echo "$TOPO_DIR/Butterfly/platform_butterfly_${n}.xml"
            ;;
        Dragonfly)
            echo "$TOPO_DIR/Dragonfly/platform_dragonfly_${n}.xml"
            ;;
        FatTree)
            echo "$TOPO_DIR/FatTree/platform_fattree_${n}.xml"
            ;;
    esac
}

# ---- Check binary ----
if [ ! -x "$BINARY" ]; then
    echo "Error: runner binary not found. Run 'make' first." >&2
    exit 1
fi

echo "=============================================="
echo "  Broadcast Experiment Sweep"
echo "  Topologies: ${TOPOS[*]}"
echo "  Sizes:      ${SIZES[*]}"
echo "  Algorithms: ${ALGOS[*]}"
echo "=============================================="

for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        PLATFORM=$(find_platform "$TOPO" "$N")
        HOSTFILE="$TOPO_DIR/$TOPO/hostfile_$N"

        if [ ! -f "$PLATFORM" ] || [ ! -f "$HOSTFILE" ]; then
            echo "  SKIP $TOPO N=$N (platform/hostfile not found)"
            continue
        fi

        OUTFILE="$RES_DIR/${TOPO}_${N}.csv"
        echo "topology,nodes,algorithm,msg_bytes,nchunks,root,time_sec,correct" > "$OUTFILE"

        echo ""
        echo "--- $TOPO  N=$N ---"

        for MSG in "${MSG_SIZES[@]}"; do
            NC=$(choose_chunks "$MSG")

            if   [ "$MSG" -ge 1048576 ]; then
                HR="$(( MSG / 1048576 )) MB"
            elif [ "$MSG" -ge 1024 ]; then
                HR="$(( MSG / 1024 )) KB"
            else
                HR="${MSG} B"
            fi

            for ALGO in "${ALGOS[@]}"; do
                TOPO_CFG=""
                if [[ "$ALGO" == "test" ]]; then
                    TOPO_CFG="$TOPO_DIR/$TOPO/topo_${N}.cfg"
                fi
                OUTPUT=$(smpirun -np "$N" -platform "$PLATFORM" \
                    -hostfile "$HOSTFILE" \
                    $HOST_SPEED $QUIET \
                    "$BINARY" "$ALGO" "$MSG" "$NC" "$ROOT" _ $TOPO_CFG \
                    2>/dev/null || true)

                if echo "$OUTPUT" | grep -q '^time_sec'; then
                    TIME=$(echo "$OUTPUT" | grep '^time_sec' | awk '{print $3}')
                    OK=$(echo "$OUTPUT"   | grep '^correct'  | awk '{print $3}')
                    echo "  $HR  $ALGO : ${TIME} s  (correct: ${OK})"
                    echo "${TOPO},${N},${ALGO},${MSG},${NC},${ROOT},${TIME},${OK}" >> "$OUTFILE"
                else
                    echo "  $HR  $ALGO : [failed]"
                fi
            done
        done

        echo "  -> $OUTFILE"
    done
done

echo ""
echo "=============================================="
echo "  Done. Results in: $RES_DIR/"
echo "=============================================="
