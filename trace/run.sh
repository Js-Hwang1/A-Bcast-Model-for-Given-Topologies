#!/usr/bin/env bash
#
# Run one broadcast experiment quickly, without generating a trace.
#
# Usage:
#   ./run.sh [algo] [topo] [N] [msg] [root] [nchunks]
#
# Examples:
#   ./run.sh
#   ./run.sh bine
#   ./run.sh bine_chunk 2Dmesh 128 512 0 8
#   ./run.sh mpi 2Dmesh 128 512 0
#   ./run.sh glf 2Dmesh 128 512 0
#   ./run.sh proto 2Dmesh 128 512 0 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/.." && pwd)"
TOPO_DIR="$PROJ/topo"
BINARY="$PROJ/bin/runner"

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

TOPO_CFG="$TOPO_DIR/$TOPO/topo_${N}.cfg"
TDAT="${PLATFORM%.xml}.tdat"
SDAT="${PLATFORM%.xml}.sdat"
EXTRA_ARG=""

if [[ ! -f "$PLATFORM" ]]; then
    echo "Platform not found: $PLATFORM" >&2
    exit 1
fi
if [[ ! -f "$HOSTFILE" ]]; then
    echo "Hostfile not found: $HOSTFILE" >&2
    exit 1
fi
if [[ ! -x "$BINARY" ]]; then
    echo "Runner not found: $BINARY" >&2
    echo "  With Docker, build first: docker run --rm -v $PROJ:/work -w /work simgrid/stable:latest smpicc -O2 -Wall -Wextra -o /work/bin/runner /work/src/runner.c -lm" >&2
    echo "  Or install SimGrid and build locally." >&2
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
    glf)
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

if command -v smpirun &>/dev/null; then
    RUN_CMD="smpirun"
    RUN_PLATFORM="$PLATFORM"
    RUN_HOSTFILE="$HOSTFILE"
    RUN_BINARY="$BINARY"
    RUN_EXTRA_ARG="$EXTRA_ARG"
else
    if ! command -v docker &>/dev/null; then
        echo "smpirun not found and docker not available." >&2
        exit 1
    fi
    RUN_CMD="docker run --rm -v $PROJ:/work -w /work simgrid/stable:latest smpirun"
    RUN_PLATFORM="/work/${PLATFORM#$PROJ/}"
    RUN_HOSTFILE="/work/${HOSTFILE#$PROJ/}"
    RUN_BINARY="/work/bin/runner"
    RUN_EXTRA_ARG=""
    if [[ -n "$EXTRA_ARG" ]]; then
        RUN_EXTRA_ARG="/work/${EXTRA_ARG#$PROJ/}"
    fi
fi

echo "run: $ALGO TOPO=$TOPO N=$N MSG=$MSG ROOT=$ROOT NCHUNKS=$NC"

$RUN_CMD -np "$N" -platform "$RUN_PLATFORM" -hostfile "$RUN_HOSTFILE" \
    --cfg=smpi/host-speed:2000Gf \
    --cfg=smpi/display-timing:yes \
    --log=root.thres:warning \
    "$RUN_BINARY" "$ALGO" "$MSG" "$NC" "$ROOT" _ "${RUN_EXTRA_ARG:-_}"
