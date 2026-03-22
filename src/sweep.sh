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
#   ./sweep.sh --roots 0-127          # roots 0 through 127
#   ./sweep.sh --roots 128-255        # roots 128 through 255
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
ALGOS=(mpi srda pipe bine glf obfs bbs)
MSG_SIZES=(65536 262144 1048576 4194304 16777216 67108864)

# ---- Defaults ----
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DRY_RUN=0
SIF=""
ROOT_MODE="single"   # single | all | lo | hi | range
ROOT_SINGLE=0
ROOT_LO=0
ROOT_HI=0
CHUNKS_OVERRIDE=""

# ---- Parse CLI ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)    JOBS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --algos)      IFS=',' read -ra ALGOS <<< "$2"; shift 2 ;;
        --topos)      IFS=',' read -ra TOPOS <<< "$2"; shift 2 ;;
        --sizes)      IFS=',' read -ra SIZES <<< "$2"; shift 2 ;;
        --msgs)       IFS=',' read -ra MSG_SIZES <<< "$2"; shift 2 ;;
        --chunks)     CHUNKS_OVERRIDE="$2"; shift 2 ;;
        --sif)        SIF="$2"; shift 2 ;;
        --roots)
            case "$2" in
                all) ROOT_MODE="all" ;;
                lo)  ROOT_MODE="lo" ;;
                hi)  ROOT_MODE="hi" ;;
                *-*) ROOT_MODE="range"; ROOT_LO="${2%%-*}"; ROOT_HI="${2##*-}" ;;
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
        range)  seq "$ROOT_LO" "$ROOT_HI" ;;
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
        32)   echo "4x8"   ;;
        36)   echo "6x6"   ;;
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
    local nc=$(( msg / 8192 )); (( nc < 4 )) && nc=4; echo "$nc"
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
    [[ "$_algo" == "test" || "$_algo" == "ffgb" || "$_algo" == "obfs" || "$_algo" == "bbs" ]] && needs_tdat=1
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
                if [[ -n "$SIF" ]]; then
                    singularity exec --bind "$PROJ_ROOT" "$SIF" python3 "$PREPROCESS" "$_xml" "$_tdat"
                else
                    python3 "$PREPROCESS" "$_xml" "$_tdat"
                fi
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
            NP=$N
            HF=$HOSTFILE
            ROOT_ARG="all"
            if [[ "$ALGO" == "bbs" && "$TOPO" == "FatTree" ]]; then
                CFG_FILE="$TOPO_DIR/$TOPO/topo_${N}.cfg"
                if [[ -f "$CFG_FILE" ]]; then
                    NPL=$(awk '/^fattree/ { print $2 }' "$CFG_FILE")
                    if [[ -n "$NPL" && "$NPL" -gt 0 ]]; then
                        NL=$((N / NPL))
                        NS=$((2 * (NL - 1)))  # one spine rank per inter-leaf edge
                        NP=$((N + NL + NS))    # Nc + Nl + 2*(Nl-1)
                        HF="$TOPO_DIR/$TOPO/hostfile_bbs_$N"
                        ROOT_ARG="all:$N"      # limit roots to compute nodes
                    fi
                fi
            fi

            for MSG in "${MSG_SIZES[@]}"; do
                NC=${CHUNKS_OVERRIDE:-$(choose_chunks "$MSG")}

                if [[ "$ROOT_MODE" == "all" ]]; then
                    # Bulk mode: single smpirun with root=all
                    # Much faster — one MPI_Init for all roots
                    OUTJSON="$DATA_DIR/$TOPO/$ALGO/N${N}_MSG${MSG}.json"
                    OUTDIR=$(dirname "$OUTJSON")

                    CMD="mkdir -p $OUTDIR"
                    CMD+=" && $SMPI_PREFIX smpirun -np $NP"
                    CMD+=" -platform $PLATFORM"
                    CMD+=" -hostfile $HF"
                    CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                    CMD+=" --cfg=smpi/simulate-computation:no"
                    CMD+=" --cfg=smpi/display-timing:yes"
                    CMD+=" --log=root.thres:warning"
                    CMD+=" $BINARY $ALGO $MSG $NC $ROOT_ARG $OUTJSON"
                    if [[ "$ALGO" == "test" || "$ALGO" == "ffgb" || "$ALGO" == "obfs" || "$ALGO" == "bbs" ]]; then
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
                        CMD+=" && $SMPI_PREFIX smpirun -np $NP"
                        CMD+=" -platform $PLATFORM"
                        CMD+=" -hostfile $HF"
                        CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                        CMD+=" --cfg=smpi/simulate-computation:no"
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
                        if [[ "$ALGO" == "test" || "$ALGO" == "ffgb" || "$ALGO" == "obfs" || "$ALGO" == "bbs" ]]; then
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

    # Collect PIDs and job labels
    PIDS=()
    LABELS=()
    RUNNING=0

    # Helper: format elapsed / ETA as human-readable
    fmt_time() {
        local s=$1
        if (( s >= 3600 )); then
            printf "%dh%02dm" $((s/3600)) $(((s%3600)/60))
        elif (( s >= 60 )); then
            printf "%dm%02ds" $((s/60)) $((s%60))
        else
            printf "%ds" "$s"
        fi
    }

    # Helper: print completion line with ETA
    print_progress() {
        local NOW ELAPSED ETA_S ETA_STR
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        if (( COMPLETED > 0 && ELAPSED > 0 )); then
            ETA_S=$(( (NJOBS - COMPLETED) * ELAPSED / COMPLETED ))
            ETA_STR=$(fmt_time $ETA_S)
        else
            ETA_STR="?"
        fi
        local STATUS_TAG=""
        (( FAILED > 0 )) && STATUS_TAG="  ${FAILED} failed"
        printf "  [%d/%d] %-50s  elapsed=%-8s ETA=%s%s\n" \
            "$COMPLETED" "$NJOBS" "$1" "$(fmt_time $ELAPSED)" "$ETA_STR" "$STATUS_TAG"
    }

    while IFS= read -r CMD; do
        # Extract a short label from the command (algo + msg + root info)
        LABEL=$(echo "$CMD" | sed -n 's/.*runner \([a-z]*\) \([0-9]*\) [0-9]* \([^ ]*\) .*/\1 MSG=\2 root=\3/p')
        [[ -z "$LABEL" ]] && LABEL="job $((${#PIDS[@]}+1))"

        bash -c "$CMD" > /dev/null 2>&1 &
        PIDS+=($!)
        LABELS+=("$LABEL")
        RUNNING=$((RUNNING + 1))

        if (( RUNNING >= JOBS )); then
            wait "${PIDS[COMPLETED]}" 2>/dev/null
            STATUS=$?
            COMPLETED=$((COMPLETED + 1))
            RUNNING=$((RUNNING - 1))
            (( STATUS != 0 )) && FAILED=$((FAILED + 1))
            print_progress "${LABELS[$((COMPLETED-1))]}"
        fi
    done < "$JOBFILE"

    # Drain remaining jobs
    while (( COMPLETED < NJOBS )); do
        wait "${PIDS[COMPLETED]}" 2>/dev/null
        STATUS=$?
        COMPLETED=$((COMPLETED + 1))
        RUNNING=$((RUNNING - 1))
        (( STATUS != 0 )) && FAILED=$((FAILED + 1))
        print_progress "${LABELS[$((COMPLETED-1))]}"
    done
    echo ""
fi

# ---- Aggregate per-root JSONs into bulk JSONs for plotting ----
echo "Aggregating per-root JSONs..."
for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        for ALGO in "${ALGOS[@]}"; do
            for MSG in "${MSG_SIZES[@]}"; do
                OUTDIR="$DATA_DIR/$TOPO/$ALGO"
                BULK="$OUTDIR/N${N}_MSG${MSG}.json"
                [[ -f "$BULK" ]] && continue

                # Check if all per-root JSONs exist
                ALL_EXIST=1
                for ((r=0; r<N; r++)); do
                    [[ ! -f "$OUTDIR/N${N}_MSG${MSG}_R${r}.json" ]] && ALL_EXIST=0 && break
                done
                (( ALL_EXIST == 0 )) && continue

                NC=${CHUNKS_OVERRIDE:-$(choose_chunks "$MSG")}
                python3 -c "
import json, math
N, outdir, msg, algo, nc = $N, '$OUTDIR', $MSG, '$ALGO', $NC
times, ok = [], True
for r in range(N):
    with open(f'{outdir}/N{N}_MSG{msg}_R{r}.json') as f:
        rec = json.load(f)
    times.append(rec['time_sec'])
    if not rec.get('correct', True): ok = False
mean = sum(times)/N
stdev = math.sqrt(sum((t-mean)**2 for t in times)/N)
bulk = {'algorithm':algo,'nodes':N,'msg_bytes':msg,'nchunks':nc,
        'n_roots':N,'mean_sec':mean,'stdev_sec':stdev,
        'min_sec':min(times),'max_sec':max(times),
        'correct':ok,'per_root_sec':times}
with open('$BULK','w') as f: json.dump(bulk,f,indent=2)
print(f'  {algo} N={N} MSG={msg}: mean={mean*1e6:.1f}us  min={min(times)*1e6:.1f}us  max={max(times)*1e6:.1f}us')
" 2>/dev/null && \
                # Clean up per-root files
                for ((r=0; r<N; r++)); do
                    rm -f "$OUTDIR/N${N}_MSG${MSG}_R${r}.json"
                done
            done
        done
    done
done

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
