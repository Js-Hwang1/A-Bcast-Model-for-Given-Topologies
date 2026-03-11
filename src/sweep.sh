#!/usr/bin/env bash
#
# sweep.sh — Dynamic-dispatch parallel experiment sweep.
#
# Enumerates all (topology, N, algorithm, msg_bytes, root) experiments
# and dispatches them to a pool of CPU cores.  When a core finishes
# early, it immediately picks up the next pending job (work-stealing).
#
# Uses GNU parallel (preferred) or falls back to bash wait -n.
#
# Usage:
#   ./sweep.sh                        # root=0 only (default)
#   ./sweep.sh --roots all            # every root 0..N-1
#   ./sweep.sh --roots lo             # roots 0..N/2-1
#   ./sweep.sh --roots hi             # roots N/2..N-1
#   ./sweep.sh --roots 5              # just root 5
#   ./sweep.sh -j 8                   # limit to 8 workers
#   ./sweep.sh --dry-run              # list commands, don't execute
#   ./sweep.sh --algos mpi,srda       # specific algorithms
#   ./sweep.sh --topos FatTree,2Dmesh # specific topologies
#   ./sweep.sh --sizes 128,256        # specific node counts
#   ./sweep.sh --msgs 1024,1048576    # specific message sizes
#   ./sweep.sh --sif bcast.sif        # run smpirun inside container
#   TRACE=1 ./sweep.sh               # enable Paje tracing
#
# Output:  JSON files in ../data/<Topo>/<algo>/N<N>_MSG<msg>_R<root>.json
#          Job log in     ../data/sweep_<roots>.log  (GNU parallel only)
#

set -euo pipefail

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOPO_DIR="$SCRIPT_DIR/../topo"
DATA_DIR="$SCRIPT_DIR/../data"
BINARY="$SCRIPT_DIR/../bin/runner"
HOST_SPEED="2000Gf"

# ---- Default parameter space ----
TOPOS=(2Dmesh Butterfly Dragonfly FatTree)
SIZES=(128 256 512 1024)
ALGOS=(mpi srda pipe bine glf ffgb)
MSG_SIZES=(256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864)

# ---- Defaults ----
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DRY_RUN=0
SIF=""
ROOT_MODE="single"   # single | all | lo | hi
ROOT_SINGLE=0

# ---- Parse CLI ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)    JOBS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --algos)      IFS=',' read -ra ALGOS <<< "$2"; shift 2 ;;
        --topos)      IFS=',' read -ra TOPOS <<< "$2"; shift 2 ;;
        --sizes)      IFS=',' read -ra SIZES <<< "$2"; shift 2 ;;
        --msgs)       IFS=',' read -ra MSG_SIZES <<< "$2"; shift 2 ;;
        --sif)        SIF="$2"; shift 2 ;;
        --roots)
            case "$2" in
                all) ROOT_MODE="all" ;;
                lo)  ROOT_MODE="lo" ;;
                hi)  ROOT_MODE="hi" ;;
                *)   ROOT_MODE="single"; ROOT_SINGLE="$2" ;;
            esac
            shift 2 ;;
        -h|--help)
            sed -n '3,/^$/{ s/^# \{0,1\}//; p }' "$0"
            exit 0 ;;
        *)  echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ---- Root range generator (called per N) ----
# Outputs space-separated list of root values.
root_range() {
    local n=$1
    case "$ROOT_MODE" in
        single) echo "$ROOT_SINGLE" ;;
        all)    seq 0 $(( n - 1 )) ;;
        lo)     seq 0 $(( n / 2 - 1 )) ;;
        hi)     seq $(( n / 2 )) $(( n - 1 )) ;;
    esac
}

# ---- Container prefix (empty when running natively) ----
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMPI_PREFIX=""
if [[ -n "$SIF" ]]; then
    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: Container not found: $SIF" >&2
        exit 1
    fi
    SMPI_PREFIX="singularity exec --bind $PROJ_ROOT $SIF"
    echo "Container  : $SIF"
fi

# ---- Helpers ----
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
    local msg=$1
    local nc=$(( msg / 16384 )); (( nc < 4 )) && nc=4; echo "$nc"
}

topo_cfg_path() {
    local topo=$1 n=$2
    echo "$TOPO_DIR/$topo/topo_${n}.cfg"
}

topo_data_path() {
    local topo=$1 n=$2
    local xml
    xml=$(platform_path "$topo" "$n")
    echo "${xml%.xml}.tdat"
}

# ---- Build runner if needed ----
if [[ ! -x "$BINARY" ]]; then
    echo "Building runner..."
    if [[ -n "$SMPI_PREFIX" ]]; then
        mkdir -p "$(dirname "$BINARY")"
        $SMPI_PREFIX smpicc -O2 -Wall -Wextra -o "$BINARY" "$SCRIPT_DIR/runner.c" -lm
    else
        make -C "$SCRIPT_DIR" -s
    fi
fi

# ---- Preprocess XML -> .tdat for test algorithm ----
needs_tdat=0
for _algo in "${ALGOS[@]}"; do
    [[ "$_algo" == "test" || "$_algo" == "ffgb" ]] && needs_tdat=1
done
if [[ $needs_tdat -eq 1 ]]; then
    PREPROCESS="$SCRIPT_DIR/topo_preprocess.py"
    for _topo in "${TOPOS[@]}"; do
        for _n in "${SIZES[@]}"; do
            _xml=$(platform_path "$_topo" "$_n")
            _tdat=$(topo_data_path "$_topo" "$_n")
            [[ ! -f "$_xml" ]] && continue
            if [[ ! -f "$_tdat" || "$_xml" -nt "$_tdat" ]]; then
                echo "Preprocessing: $_xml -> $_tdat"
                python3 "$PREPROCESS" "$_xml" "$_tdat"
            fi
        done
    done
fi

# ---- Enumerate all jobs ----
JOBFILE=$(mktemp "${TMPDIR:-/tmp}/sweep_jobs.XXXXXX")
trap 'rm -f "$JOBFILE"' EXIT

NJOBS=0

for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        PLATFORM=$(platform_path "$TOPO" "$N")
        HOSTFILE="$TOPO_DIR/$TOPO/hostfile_$N"

        # Skip if topology files missing
        [[ ! -f "$PLATFORM" || ! -f "$HOSTFILE" ]] && continue

        for ALGO in "${ALGOS[@]}"; do
            for MSG in "${MSG_SIZES[@]}"; do
                NC=$(choose_chunks "$MSG")

                if [[ "$ROOT_MODE" == "all" ]]; then
                    # Bulk mode: single smpirun with root=all
                    # Avoids per-root MPI_Init overhead (128x fewer launches)
                    OUTJSON="$DATA_DIR/$TOPO/$ALGO/N${N}_MSG${MSG}.json"
                    OUTDIR=$(dirname "$OUTJSON")

                    CMD="mkdir -p $OUTDIR"
                    CMD+=" && $SMPI_PREFIX smpirun -np $N"
                    CMD+=" -platform $PLATFORM"
                    CMD+=" -hostfile $HOSTFILE"
                    CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                    CMD+=" --cfg=smpi/display-timing:yes"
                    CMD+=" --log=root.thres:warning"
                    CMD+=" $BINARY $ALGO $MSG $NC all $OUTJSON"
                    if [[ "$ALGO" == "test" || "$ALGO" == "ffgb" ]]; then
                        CMD+=" $(topo_data_path "$TOPO" "$N")"
                    elif [[ "$ALGO" == "glf" ]]; then
                        CMD+=" $(topo_cfg_path "$TOPO" "$N")"
                    fi

                    echo "$CMD" >> "$JOBFILE"
                    NJOBS=$((NJOBS + 1))
                else
                    # Per-root mode: one smpirun per root value
                    ROOTS=$(root_range "$N")
                    for ROOT in $ROOTS; do
                        OUTJSON="$DATA_DIR/$TOPO/$ALGO/N${N}_MSG${MSG}_R${ROOT}.json"
                        OUTDIR=$(dirname "$OUTJSON")

                        CMD="mkdir -p $OUTDIR"
                        CMD+=" && $SMPI_PREFIX smpirun -np $N"
                        CMD+=" -platform $PLATFORM"
                        CMD+=" -hostfile $HOSTFILE"
                        CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                        CMD+=" --cfg=smpi/display-timing:yes"
                        CMD+=" --log=root.thres:warning"

                        if [[ "${TRACE:-}" == "1" ]]; then
                            TRACEFILE="${OUTJSON%.json}.trace"
                            CMD+=" -trace"
                            CMD+=" --cfg=tracing/filename:$TRACEFILE"
                            CMD+=" --cfg=tracing/smpi:yes"
                            CMD+=" --cfg=tracing/smpi/internals:yes"
                        fi

                        CMD+=" $BINARY $ALGO $MSG $NC $ROOT $OUTJSON"
                        if [[ "$ALGO" == "test" ]]; then
                            CMD+=" $(topo_data_path "$TOPO" "$N")"
                        elif [[ "$ALGO" == "glf" ]]; then
                            CMD+=" $(topo_cfg_path "$TOPO" "$N")"
                        fi

                        echo "$CMD" >> "$JOBFILE"
                        NJOBS=$((NJOBS + 1))
                    done
                fi
            done
        done
    done
done

# ---- Job log name (unique per root mode) ----
JOBLOG="$DATA_DIR/sweep_${ROOT_MODE}.log"

# ---- Print summary ----
echo "=============================================="
echo "  Broadcast Experiment Sweep"
echo "  Topologies : ${TOPOS[*]}"
echo "  Node counts: ${SIZES[*]}"
echo "  Algorithms : ${ALGOS[*]}"
echo "  Msg sizes  : ${#MSG_SIZES[@]}"
echo "  Roots      : $ROOT_MODE"
echo "  Total jobs : $NJOBS"
echo "  Workers    : $JOBS"
echo "  Output     : $DATA_DIR/"
echo "=============================================="

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "--- Dry run: commands that would execute ---"
    echo "(showing first 20 of $NJOBS)"
    head -20 "$JOBFILE" | nl -ba
    echo "..."
    exit 0
fi

if [[ $NJOBS -eq 0 ]]; then
    echo "No jobs to run. Check topology/plan files."
    exit 0
fi

# ---- Dynamic dispatch ----
mkdir -p "$DATA_DIR"

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
        --halt soon,fail=20% \
        < "$JOBFILE" \
    && true

    if [[ -f "$JOBLOG" ]]; then
        COMPLETED=$(awk 'NR>1' "$JOBLOG" | wc -l | tr -d ' ')
        FAILED=$(awk 'NR>1 && $7!=0' "$JOBLOG" | wc -l | tr -d ' ')
    fi
else
    echo ""
    echo "Dispatching via bash worker pool ($JOBS workers)..."
    echo "(Install GNU parallel for progress bars and job logging)"
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
                "$COMPLETED" "$NJOBS" "$FAILED"
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
            "$COMPLETED" "$NJOBS" "$FAILED"
    done
    echo ""
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
JSON_COUNT=$(find "$DATA_DIR" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=============================================="
echo "  Sweep complete"
echo "  Completed  : $COMPLETED / $NJOBS jobs"
[[ $FAILED -gt 0 ]] && \
echo "  Failed     : $FAILED jobs"
echo "  Wall time  : ${ELAPSED}s"
echo "  JSON files : $JSON_COUNT"
echo "  Output     : $DATA_DIR/"
[[ -f "$JOBLOG" ]] && \
echo "  Job log    : $JOBLOG"
echo "=============================================="
