#!/usr/bin/env bash
#
# Generate a Paje trace and a states CSV for one broadcast run.
#
# Usage:
#   ./run_with_trace.sh [algo] [topo] [N] [msg] [root] [nchunks]
#
# Examples:
#   ./run_with_trace.sh              # bine, 2Dmesh, 128, 512, 0
#   ./run_with_trace.sh bine
#   ./run_with_trace.sh bine_chunk 2Dmesh 128 512 0 8
#   ./run_with_trace.sh mpi 2Dmesh 128 512 0
#   ./run_with_trace.sh glf 2Dmesh 128 512 0

#   ./run_with_trace.sh pipe Dragonfly 128 4194304 0 512
#   ./run_with_trace.sh srda Dragonfly 128 4194304 0 128
#   ./run_with_trace.sh mpi Dragonfly 128 4194304 0 128
#   ./run_with_trace.sh glf Dragonfly 128 4194304 0 128
#   ./run_with_trace.sh bine Dragonfly 128 4194304 0 128
#   ./run_with_trace.sh bbs Dragonfly 128 4194304 0 512

#
# Output in traces/:
#   <TOPO>_N<N>_<ALGO>_MSG<MSG>_R<ROOT>.trace
#   <TOPO>_N<N>_<ALGO>_MSG<MSG>_R<ROOT>.states.csv      (if pj_dump is available)
#   <TOPO>_N<N>_<ALGO>_MSG<MSG>_R<ROOT>.pjdump.csv      (if pj_dump is available)
#
# Requires: SimGrid (smpirun). If smpirun is not in PATH, uses Docker:
#   docker run --rm -v $(pwd):/work -w /work simgrid/stable:latest ...
# Optional: pj_dump (Paje) for .states.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
TOPO_DIR="$PROJ/topo"
BINARY="$PROJ/bin/runner"
TRACES_DIR="$SCRIPT_DIR"

ALGO="${1:-bine}"
TOPO="${2:-2Dmesh}"
N="${3:-128}"
MSG="${4:-512}"
ROOT="${5:-0}"
NC="${6:-64}"

mesh_dims() {
    case "$1" in
        128)  echo "8x16"  ;;
        256)  echo "16x16" ;;
        512)  echo "16x32" ;;
        1024) echo "32x32" ;;
        *)    echo "8x16"  ;;
    esac
}

case "$TOPO" in
    2Dmesh)    PLATFORM="$TOPO_DIR/2Dmesh/platform_2dmesh_$(mesh_dims "$N").xml" ;;
    Butterfly) PLATFORM="$TOPO_DIR/Butterfly/platform_butterfly_${N}.xml" ;;
    Dragonfly) PLATFORM="$TOPO_DIR/Dragonfly/platform_dragonfly_${N}.xml" ;;
    FatTree)   PLATFORM="$TOPO_DIR/FatTree/platform_fattree_${N}.xml" ;;
    *)         PLATFORM="$TOPO_DIR/2Dmesh/platform_2dmesh_8x16.xml" ;;
esac
HOSTFILE="$TOPO_DIR/$TOPO/hostfile_$N"
NP="$N"

TOPO_CFG="$TOPO_DIR/$TOPO/topo_${N}.cfg"
TDAT="${PLATFORM%.xml}.tdat"
SDAT="${PLATFORM%.xml}.sdat"
EXTRA_ARG=""

BASENAME="${TOPO}_N${N}_${ALGO}_MSG${MSG}_R${ROOT}"
TRACEFILE="$TRACES_DIR/${BASENAME}.trace"
CSVFILE="$TRACES_DIR/${BASENAME}.states.csv"
PJDUMPFILE="$TRACES_DIR/${BASENAME}.pjdump.csv"

if [[ ! -f "$PLATFORM" ]]; then
    echo "Platform not found: $PLATFORM" >&2
    exit 1
fi
if [[ ! -x "$BINARY" ]]; then
    echo "Runner not found: $BINARY" >&2
    echo "  With Docker, build first: docker run --rm -v $PROJ:/work -w /work/src simgrid/stable:latest make" >&2
    echo "  Or install SimGrid and: make -C src" >&2
    exit 1
fi
ensure_tdat() {
    if [[ -f "$TDAT" && "$PLATFORM" -ot "$TDAT" ]]; then
        return
    fi
    echo "Generating topology data: $TDAT"
    if command -v python3 &>/dev/null; then
        if python3 "$PROJ/src/topo_preprocess.py" "$PLATFORM" "$TDAT"; then
            return
        fi
    fi
    if command -v docker &>/dev/null; then
        docker run --rm -v "$PROJ:/work" -w /work simgrid/stable:latest \
            python3 /work/src/topo_preprocess.py \
            "/work/${PLATFORM#$PROJ/}" "/work/${TDAT#$PROJ/}"
        return
    fi
    echo "Could not generate $TDAT (need local python3 deps or docker)." >&2
    exit 1
}

ensure_sdat() {
    if [[ -f "$SDAT" && "$PLATFORM" -ot "$SDAT" ]]; then
        return
    fi
    echo "Generating spectral data: $SDAT"
    if command -v python3 &>/dev/null; then
        if python3 "$PROJ/src/spectral_preprocess.py" "$PLATFORM" "$SDAT"; then
            return
        fi
    fi
    if command -v docker &>/dev/null; then
        docker run --rm -v "$PROJ:/work" -w /work simgrid/stable:latest \
            python3 /work/src/spectral_preprocess.py \
            "/work/${PLATFORM#$PROJ/}" "/work/${SDAT#$PROJ/}"
        return
    fi
    echo "Could not generate $SDAT (need local python3 deps or docker)." >&2
    exit 1
}

case "$ALGO" in
    glf|bbs)
        EXTRA_ARG="$TOPO_CFG"
        if [[ ! -f "$EXTRA_ARG" ]]; then
            echo "Topology config not found: $EXTRA_ARG" >&2
            exit 1
        fi
        ;;
    proto|test)
        ensure_tdat
        EXTRA_ARG="$TDAT"
        ;;
    spec|specbin)
        ensure_sdat
        EXTRA_ARG="$SDAT"
        ;;
esac

# BBS + FatTree: extend NP to include switch relay ranks
if [[ "$ALGO" == "bbs" && "$TOPO" == "FatTree" ]]; then
    if [[ -f "$TOPO_CFG" ]]; then
        NPL=$(awk '/^fattree/ { print $2 }' "$TOPO_CFG")
        if [[ -n "$NPL" && "$NPL" -gt 0 ]]; then
            NL=$((N / NPL))
            NS=$NPL
            NP=$((N + NL + NS))
            HOSTFILE="$TOPO_DIR/$TOPO/hostfile_$NP"
        fi
    fi
fi

if [[ ! -f "$HOSTFILE" ]]; then
    echo "Hostfile not found: $HOSTFILE" >&2
    exit 1
fi

# Use Docker when smpirun is not in PATH
if command -v smpirun &>/dev/null; then
    RUN_CMD="smpirun"
    RUN_PLATFORM="$PLATFORM"
    RUN_HOSTFILE="$HOSTFILE"
    RUN_BINARY="$BINARY"
    RUN_TRACEFILE="$TRACEFILE"
    RUN_EXTRA_ARG="$EXTRA_ARG"
else
    if ! command -v docker &>/dev/null; then
        echo "smpirun not found and docker not available." >&2
        echo "  Install SimGrid, or use Docker: docker run --rm -v \$PROJ:/work -w /work simgrid/stable:latest ./traces/run_with_trace.sh $ALGO $TOPO $N $MSG $ROOT" >&2
        exit 1
    fi
    RUN_CMD="docker run --rm -v $PROJ:/work -w /work simgrid/stable:latest smpirun"
    RUN_PLATFORM="/work/${PLATFORM#$PROJ/}"
    RUN_HOSTFILE="/work/${HOSTFILE#$PROJ/}"
    RUN_BINARY="/work/bin/runner"
    RUN_TRACEFILE="/work/traces/${BASENAME}.trace"
    RUN_EXTRA_ARG=""
    if [[ -n "$EXTRA_ARG" ]]; then
        RUN_EXTRA_ARG="/work/${EXTRA_ARG#$PROJ/}"
    fi
fi

TRACE_SOURCE_FLAG=""
if command -v smpirun &>/dev/null; then
    if smpirun --help 2>/dev/null | sed -n '/-trace-source/p' | sed -n '1p' >/dev/null && \
       [[ -n "$(smpirun --help 2>/dev/null | sed -n '/-trace-source/p')" ]]; then
        TRACE_SOURCE_FLAG="-trace-source"
    fi
elif command -v docker &>/dev/null; then
    if docker run --rm simgrid/stable:latest sh -lc 'smpirun --help 2>/dev/null | sed -n "/-trace-source/p"' \
        | sed -n '1p' >/dev/null && \
       [[ -n "$(docker run --rm simgrid/stable:latest sh -lc 'smpirun --help 2>/dev/null | sed -n "/-trace-source/p"')" ]]; then
        TRACE_SOURCE_FLAG="-trace-source"
    fi
fi

echo "run_with_trace: $ALGO TOPO=$TOPO N=$N MSG=$MSG ROOT=$ROOT"
echo "  -> $TRACEFILE"

$RUN_CMD -np "$NP" -platform "$RUN_PLATFORM" -hostfile "$RUN_HOSTFILE" \
    --cfg=smpi/host-speed:2000Gf \
    --cfg=smpi/simulate-computation:no \
    --cfg=smpi/display-timing:yes \
    --log=root.thres:warning \
    -trace \
    $TRACE_SOURCE_FLAG \
    --cfg=tracing/filename:"$RUN_TRACEFILE" \
    --cfg=tracing/smpi:yes \
    --cfg=tracing/smpi/internals:yes \
    "$RUN_BINARY" "$ALGO" "$MSG" "$NC" "$ROOT" _ "${RUN_EXTRA_ARG:-_}"

if [[ ! -f "$TRACEFILE" ]]; then
    echo "Trace file was not created." >&2
    exit 1
fi

if command -v pj_dump &>/dev/null; then
    DYLD_FALLBACK_LIBRARY_PATH="/usr/local/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}" \
        pj_dump -z "$TRACEFILE" 2>/dev/null > "$PJDUMPFILE"

    echo "  -> $PJDUMPFILE"
    echo "  -> $CSVFILE"
    python3 - "$PJDUMPFILE" "$CSVFILE" <<'PY'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="") as inp, open(dst, "w", newline="") as out:
    reader = csv.reader(inp, skipinitialspace=True)
    writer = csv.writer(out)
    for row in reader:
        if row and row[0] == "State":
            writer.writerow(row)
PY
    [[ -s "$CSVFILE" ]] && echo "     $(wc -l < "$CSVFILE") State rows"

else
    echo "  (install pj_dump to generate trace-derived CSV outputs)"
fi

echo "Done."
