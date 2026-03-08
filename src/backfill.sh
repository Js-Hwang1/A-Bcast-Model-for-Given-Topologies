#!/usr/bin/env bash
#
# backfill.sh — Smart backfill for missing experiments.
#
# Scans the data/ directory for expected JSON files that don't exist
# and re-dispatches them.  Matches the sweep.sh conventions:
#
#   - Main algos (mpi,srda,pipe,bine,glf): bulk mode (--roots all),
#     output files named  N<N>_MSG<msg>.json  (no _R suffix).
#   - Spec algo: per-root mode (root=0 only),
#     output files named  N<N>_MSG<msg>_R0.json.
#
# OOM-aware:  automatically lowers concurrency for large N×MSG
# combinations that previously OOM-killed (signal 137).
#
# Usage:
#   ./backfill.sh                        # detect & re-run all missing
#   ./backfill.sh --dry-run              # list missing, don't execute
#   ./backfill.sh -j 8                   # override max workers
#   ./backfill.sh --sif bcast.sif        # run inside container
#   ./backfill.sh --algos mpi,spec       # specific algorithms
#   ./backfill.sh --topos FatTree        # specific topologies
#   ./backfill.sh --sizes 1024           # specific node counts
#   ./backfill.sh --msgs 67108864        # specific message sizes

set -euo pipefail

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOPO_DIR="$SCRIPT_DIR/../topo"
DATA_DIR="$SCRIPT_DIR/../data"
BINARY="$SCRIPT_DIR/../bin/runner"
HOST_SPEED="2000Gf"

# ---- Default parameter space (must match sweep.sh) ----
TOPOS=(2Dmesh Butterfly Dragonfly FatTree)
SIZES=(128 256 512 1024)
ALGOS=(mpi srda pipe bine glf)
MSG_SIZES=(256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864)

# ---- Defaults ----
MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DRY_RUN=0
SIF=""

# ---- Parse CLI ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)    MAX_JOBS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --sif)        SIF="$2"; shift 2 ;;
        --algos)      IFS=',' read -ra ALGOS <<< "$2"; shift 2 ;;
        --topos)      IFS=',' read -ra TOPOS <<< "$2"; shift 2 ;;
        --sizes)      IFS=',' read -ra SIZES <<< "$2"; shift 2 ;;
        --msgs)       IFS=',' read -ra MSG_SIZES <<< "$2"; shift 2 ;;
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

spec_data_path() {
    local topo=$1 n=$2
    local xml
    xml=$(platform_path "$topo" "$n")
    echo "${xml%.xml}.sdat"
}

# ---- OOM-aware concurrency ----
# Returns the number of workers to use for a given (N, MSG) pair.
# Large N × large MSG combos previously OOM-killed; throttle them.
choose_jobs() {
    local n=$1 msg=$2
    local mem_pressure=$(( n * msg ))

    if (( mem_pressure >= 68719476736 )); then
        # N>=1024 × MSG>=64MB  — the known OOM zone
        echo 1
    elif (( mem_pressure >= 17179869184 )); then
        # N>=1024 × MSG>=16MB  or  N>=512 × MSG>=32MB
        echo 2
    elif (( mem_pressure >= 4294967296 )); then
        # moderately large
        local j=$(( MAX_JOBS / 2 ))
        (( j < 2 )) && j=2
        echo "$j"
    else
        echo "$MAX_JOBS"
    fi
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


# ---- Find missing experiments ----
# We group jobs by concurrency tier so large jobs don't compete with
# small ones for memory.  Each tier runs sequentially; within a tier,
# jobs run in parallel at that tier's concurrency limit.
#
# Tiers:  1  = very heavy (N×MSG >= 64G, known OOM zone)
#         2  = heavy      (N×MSG >= 16G)
#         half = moderate (N×MSG >= 4G)
#         max  = everything else

TIER_1_FILE=$(mktemp "${TMPDIR:-/tmp}/backfill_t1.XXXXXX")
TIER_2_FILE=$(mktemp "${TMPDIR:-/tmp}/backfill_t2.XXXXXX")
TIER_HALF_FILE=$(mktemp "${TMPDIR:-/tmp}/backfill_thalf.XXXXXX")
TIER_MAX_FILE=$(mktemp "${TMPDIR:-/tmp}/backfill_tmax.XXXXXX")
trap 'rm -f "$TIER_1_FILE" "$TIER_2_FILE" "$TIER_HALF_FILE" "$TIER_MAX_FILE"' EXIT

TOTAL_MISSING=0
MISSING_SUMMARY=""

for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        PLATFORM=$(platform_path "$TOPO" "$N")
        HOSTFILE="$TOPO_DIR/$TOPO/hostfile_$N"
        [[ ! -f "$PLATFORM" || ! -f "$HOSTFILE" ]] && continue

        for ALGO in "${ALGOS[@]}"; do
            for MSG in "${MSG_SIZES[@]}"; do
                NC=$(choose_chunks "$MSG")

                if [[ "$ALGO" == "spec" ]]; then
                    # Spec: per-root file with _R0 suffix
                    OUTJSON="$DATA_DIR/$TOPO/$ALGO/N${N}_MSG${MSG}_R0.json"
                else
                    # Main algos: bulk file (no _R suffix)
                    OUTJSON="$DATA_DIR/$TOPO/$ALGO/N${N}_MSG${MSG}.json"
                fi

                # Skip if already exists and is non-empty
                [[ -s "$OUTJSON" ]] && continue

                OUTDIR=$(dirname "$OUTJSON")
                CMD="mkdir -p $OUTDIR"
                CMD+=" && $SMPI_PREFIX smpirun -np $N"
                CMD+=" -platform $PLATFORM"
                CMD+=" -hostfile $HOSTFILE"
                CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                CMD+=" --cfg=smpi/display-timing:yes"
                CMD+=" --log=root.thres:warning"

                if [[ "$ALGO" == "spec" ]]; then
                    CMD+=" $BINARY $ALGO $MSG $NC 0 $OUTJSON"
                    CMD+=" $(spec_data_path "$TOPO" "$N")"
                else
                    CMD+=" $BINARY $ALGO $MSG $NC all $OUTJSON"
                    if [[ "$ALGO" == "glf" ]]; then
                        CMD+=" $(topo_cfg_path "$TOPO" "$N")"
                    fi
                fi

                # Route to concurrency tier based on memory pressure
                MEM_PRESSURE=$(( N * MSG ))
                if (( MEM_PRESSURE >= 68719476736 )); then
                    echo "$CMD" >> "$TIER_1_FILE"
                elif (( MEM_PRESSURE >= 17179869184 )); then
                    echo "$CMD" >> "$TIER_2_FILE"
                elif (( MEM_PRESSURE >= 4294967296 )); then
                    echo "$CMD" >> "$TIER_HALF_FILE"
                else
                    echo "$CMD" >> "$TIER_MAX_FILE"
                fi
                TOTAL_MISSING=$((TOTAL_MISSING + 1))

                # Track for summary
                MISSING_SUMMARY="${MISSING_SUMMARY}  $(basename "$OUTJSON")  ($TOPO/$ALGO, N=$N, MSG=$MSG)
"
            done
        done
    done
done

# ---- Compute expected totals ----
EXPECTED=$(( ${#TOPOS[@]} * ${#SIZES[@]} * ${#ALGOS[@]} * ${#MSG_SIZES[@]} ))

# Count per tier
T1_COUNT=0; T2_COUNT=0; THALF_COUNT=0; TMAX_COUNT=0
[[ -s "$TIER_1_FILE" ]]    && T1_COUNT=$(wc -l < "$TIER_1_FILE" | tr -d ' ')
[[ -s "$TIER_2_FILE" ]]    && T2_COUNT=$(wc -l < "$TIER_2_FILE" | tr -d ' ')
[[ -s "$TIER_HALF_FILE" ]] && THALF_COUNT=$(wc -l < "$TIER_HALF_FILE" | tr -d ' ')
[[ -s "$TIER_MAX_FILE" ]]  && TMAX_COUNT=$(wc -l < "$TIER_MAX_FILE" | tr -d ' ')

HALF_JOBS=$(( MAX_JOBS / 2 ))
(( HALF_JOBS < 2 )) && HALF_JOBS=2

# ---- Print summary ----
echo "=============================================="
echo "  Backfill Sweep"
echo "  Topologies : ${TOPOS[*]}"
echo "  Node counts: ${SIZES[*]}"
echo "  Algorithms : ${ALGOS[*]}"
echo "  Msg sizes  : ${#MSG_SIZES[@]}"
echo "  Missing    : $TOTAL_MISSING / $EXPECTED"
echo "=============================================="

if [[ $TOTAL_MISSING -gt 0 ]]; then
    echo ""
    echo "Missing experiments:"
    echo "$MISSING_SUMMARY"

    echo "Concurrency plan:"
    [[ $TMAX_COUNT  -gt 0 ]] && echo "  -j $MAX_JOBS  : $TMAX_COUNT jobs (light)"
    [[ $THALF_COUNT -gt 0 ]] && echo "  -j $HALF_JOBS  : $THALF_COUNT jobs (moderate)"
    [[ $T2_COUNT    -gt 0 ]] && echo "  -j 2  : $T2_COUNT jobs (heavy)"
    [[ $T1_COUNT    -gt 0 ]] && echo "  -j 1  : $T1_COUNT jobs (very heavy / OOM-prone)"
    echo ""
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "--- Dry run: commands that would execute ---"
    for label_file in "max:$TIER_MAX_FILE" "half:$TIER_HALF_FILE" "2:$TIER_2_FILE" "1:$TIER_1_FILE"; do
        label="${label_file%%:*}"
        file="${label_file#*:}"
        if [[ -s "$file" ]]; then
            echo ""
            echo "=== Tier -j $label ==="
            nl -ba "$file"
        fi
    done
    exit 0
fi

if [[ $TOTAL_MISSING -eq 0 ]]; then
    echo ""
    echo "No missing jobs. All experiments complete!"
    exit 0
fi

# ---- Dispatch helper ----
run_tier() {
    local tier_jobs=$1 tier_file=$2 tier_label=$3
    [[ ! -s "$tier_file" ]] && return 0

    local tier_count
    tier_count=$(wc -l < "$tier_file" | tr -d ' ')
    echo "--- Running $tier_count $tier_label jobs at -j $tier_jobs ---"

    if command -v parallel &>/dev/null; then
        parallel \
            --bar \
            --jobs "$tier_jobs" \
            --joblog "$JOBLOG.${tier_label}" \
            --halt soon,fail=50% \
            < "$tier_file" \
        && true

        if [[ -f "$JOBLOG.${tier_label}" ]]; then
            local tok tfail
            tok=$(awk 'NR>1 && $7==0' "$JOBLOG.${tier_label}" | wc -l | tr -d ' ')
            tfail=$(awk 'NR>1 && $7!=0' "$JOBLOG.${tier_label}" | wc -l | tr -d ' ')
            COMPLETED=$((COMPLETED + tok))
            FAILED=$((FAILED + tfail))
        fi
    else
        local running=0
        while IFS= read -r CMD; do
            bash -c "$CMD" > /dev/null 2>&1 &
            running=$((running + 1))

            if (( running >= tier_jobs )); then
                if wait -n 2>/dev/null; then
                    COMPLETED=$((COMPLETED + 1))
                else
                    COMPLETED=$((COMPLETED + 1))
                    FAILED=$((FAILED + 1))
                fi
                running=$((running - 1))
                printf "\r  [%d/%d] completed (%d failed)" \
                    "$COMPLETED" "$TOTAL_MISSING" "$FAILED"
            fi
        done < "$tier_file"

        while (( running > 0 )); do
            if wait -n 2>/dev/null; then
                COMPLETED=$((COMPLETED + 1))
            else
                COMPLETED=$((COMPLETED + 1))
                FAILED=$((FAILED + 1))
            fi
            running=$((running - 1))
            printf "\r  [%d/%d] completed (%d failed)" \
                "$COMPLETED" "$TOTAL_MISSING" "$FAILED"
        done
        echo ""
    fi
}

# ---- Dispatch tiers (lightest first, heaviest last) ----
mkdir -p "$DATA_DIR"
JOBLOG="$DATA_DIR/backfill.log"

COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

run_tier "$MAX_JOBS"  "$TIER_MAX_FILE"  "light"
run_tier "$HALF_JOBS" "$TIER_HALF_FILE" "moderate"
run_tier 2            "$TIER_2_FILE"    "heavy"
run_tier 1            "$TIER_1_FILE"    "very_heavy"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "  Backfill complete"
echo "  Completed  : $COMPLETED / $TOTAL_MISSING jobs"
[[ $FAILED -gt 0 ]] && \
echo "  Failed     : $FAILED jobs"
echo "  Wall time  : ${ELAPSED}s"
echo "=============================================="
