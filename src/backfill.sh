#!/usr/bin/env bash
#
# backfill.sh — Re-run only missing experiments with low concurrency.
#
# Scans the data/ directory for expected JSON files that don't exist
# and re-dispatches them.  Uses low -j to avoid OOM on large messages.
#
# Usage:
#   ./backfill.sh                    # default: -j 4
#   ./backfill.sh -j 2              # even lower concurrency
#   ./backfill.sh --sif bcast.sif   # run inside container
#   ./backfill.sh --dry-run         # list missing, don't execute

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOPO_DIR="$SCRIPT_DIR/../topo"
DATA_DIR="$SCRIPT_DIR/../data"
BINARY="$SCRIPT_DIR/runner"
HOST_SPEED="2000Gf"

TOPOS=(2Dmesh Butterfly Dragonfly FatTree)
SIZES=(128 256 512 1024)
ALGOS=(mpi srda)
MSG_SIZES=(16777216 67108864)

JOBS=4
DRY_RUN=0
SIF=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)    JOBS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --sif)        SIF="$2"; shift 2 ;;
        --msgs)       IFS=',' read -ra MSG_SIZES <<< "$2"; shift 2 ;;
        --topos)      IFS=',' read -ra TOPOS <<< "$2"; shift 2 ;;
        --sizes)      IFS=',' read -ra SIZES <<< "$2"; shift 2 ;;
        -h|--help)    sed -n '3,/^$/{ s/^# \{0,1\}//; p }' "$0"; exit 0 ;;
        *)            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ---- Container prefix ----
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMPI_PREFIX=""
if [[ -n "$SIF" ]]; then
    [[ ! -f "$SIF" ]] && { echo "ERROR: Container not found: $SIF" >&2; exit 1; }
    SMPI_PREFIX="singularity exec --bind $PROJ_ROOT $SIF"
fi

# ---- Helpers (same as sweep.sh) ----
mesh_dims() {
    case "$1" in
        128)  echo "8x16"  ;;
        256)  echo "16x16" ;;
        512)  echo "16x32" ;;
        1024) echo "32x32" ;;
    esac
}

platform_path() {
    local topo=$1 n=$2
    case "$topo" in
        2Dmesh)    echo "$TOPO_DIR/2Dmesh/platform_2dmesh_$(mesh_dims "$n").xml" ;;
        Butterfly) echo "$TOPO_DIR/Butterfly/platform_butterfly_${n}.xml" ;;
        Dragonfly) echo "$TOPO_DIR/Dragonfly/platform_dragonfly_${n}.xml" ;;
        FatTree)   echo "$TOPO_DIR/FatTree/platform_fattree_${n}.xml" ;;
    esac
}

choose_chunks() {
    local nc=$(( $1 / 16384 ))
    (( nc < 4 )) && nc=4
    echo "$nc"
}

# ---- Build runner if needed ----
if [[ ! -x "$BINARY" ]]; then
    echo "Building runner..."
    if [[ -n "$SMPI_PREFIX" ]]; then
        $SMPI_PREFIX smpicc -O2 -Wall -Wextra -o "$BINARY" "$SCRIPT_DIR/runner.c" -lm
    else
        make -C "$SCRIPT_DIR" -s
    fi
fi

# ---- Find missing experiments ----
JOBFILE=$(mktemp "${TMPDIR:-/tmp}/backfill_jobs.XXXXXX")
trap 'rm -f "$JOBFILE"' EXIT

NMISSING=0

for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        PLATFORM=$(platform_path "$TOPO" "$N")
        HOSTFILE="$TOPO_DIR/$TOPO/hostfile_$N"
        [[ ! -f "$PLATFORM" || ! -f "$HOSTFILE" ]] && continue

        for ALGO in "${ALGOS[@]}"; do
            for MSG in "${MSG_SIZES[@]}"; do
                NC=$(choose_chunks "$MSG")

                for ROOT in $(seq 0 $(( N - 1 ))); do
                    OUTJSON="$DATA_DIR/$TOPO/$ALGO/N${N}_MSG${MSG}_R${ROOT}.json"

                    # Skip if already exists
                    [[ -f "$OUTJSON" ]] && continue

                    OUTDIR=$(dirname "$OUTJSON")
                    CMD="mkdir -p $OUTDIR"
                    CMD+=" && $SMPI_PREFIX smpirun -np $N"
                    CMD+=" -platform $PLATFORM"
                    CMD+=" -hostfile $HOSTFILE"
                    CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                    CMD+=" --cfg=smpi/display-timing:yes"
                    CMD+=" --log=root.thres:warning"
                    CMD+=" $BINARY $ALGO $MSG $NC $ROOT _ $OUTJSON"

                    echo "$CMD" >> "$JOBFILE"
                    NMISSING=$((NMISSING + 1))
                done
            done
        done
    done
done

echo "=============================================="
echo "  Backfill Sweep"
echo "  Topologies : ${TOPOS[*]}"
echo "  Msg sizes  : ${MSG_SIZES[*]}"
echo "  Missing    : $NMISSING jobs"
echo "  Workers    : $JOBS"
echo "=============================================="

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "--- Dry run: first 20 missing jobs ---"
    head -20 "$JOBFILE" | nl -ba
    echo "..."
    exit 0
fi

if [[ $NMISSING -eq 0 ]]; then
    echo "No missing jobs. All experiments complete!"
    exit 0
fi

# ---- Dispatch with low concurrency ----
mkdir -p "$DATA_DIR"
JOBLOG="$DATA_DIR/backfill.log"

COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

if command -v parallel &>/dev/null; then
    echo ""
    echo "Dispatching via GNU parallel ($JOBS workers)..."
    echo ""
    parallel \
        --bar \
        --jobs "$JOBS" \
        --joblog "$JOBLOG" \
        --halt soon,fail=10% \
        < "$JOBFILE" \
    && true

    if [[ -f "$JOBLOG" ]]; then
        COMPLETED=$(awk 'NR>1' "$JOBLOG" | wc -l | tr -d ' ')
        FAILED=$(awk 'NR>1 && $7!=0' "$JOBLOG" | wc -l | tr -d ' ')
    fi
else
    echo ""
    echo "Dispatching via bash worker pool ($JOBS workers)..."
    echo ""

    RUNNING=0
    while IFS= read -r CMD; do
        bash -c "$CMD" > /dev/null 2>&1 &
        RUNNING=$((RUNNING + 1))

        if (( RUNNING >= JOBS )); then
            if wait -n 2>/dev/null; then
                COMPLETED=$((COMPLETED + 1))
            else
                COMPLETED=$((COMPLETED + 1))
                FAILED=$((FAILED + 1))
            fi
            RUNNING=$((RUNNING - 1))
            printf "\r  [%d/%d] completed (%d failed)" \
                "$COMPLETED" "$NMISSING" "$FAILED"
        fi
    done < "$JOBFILE"

    while (( RUNNING > 0 )); do
        if wait -n 2>/dev/null; then
            COMPLETED=$((COMPLETED + 1))
        else
            COMPLETED=$((COMPLETED + 1))
            FAILED=$((FAILED + 1))
        fi
        RUNNING=$((RUNNING - 1))
        printf "\r  [%d/%d] completed (%d failed)" \
            "$COMPLETED" "$NMISSING" "$FAILED"
    done
    echo ""
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
JSON_COUNT=$(find "$DATA_DIR" -name '*_R*.json' 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=============================================="
echo "  Backfill complete"
echo "  Completed  : $COMPLETED / $NMISSING jobs"
[[ $FAILED -gt 0 ]] && \
echo "  Failed     : $FAILED jobs"
echo "  Wall time  : ${ELAPSED}s"
echo "  JSON files : $JSON_COUNT / 153600"
echo "=============================================="
