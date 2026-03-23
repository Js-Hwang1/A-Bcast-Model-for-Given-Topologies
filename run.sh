#!/usr/bin/env bash
#
# run.sh — Compile and run broadcast experiments (Docker/Singularity/native).
#
# Usage:
#   ./run.sh <algo> <topo> <N> <msg_bytes>     # single experiment, all roots
#   ./run.sh bbs Dragonfly 128 65536            # example
#   ./run.sh mpi Butterfly 256 1048576          # example
#   ./run.sh all                                # run everything
#   ./run.sh --dry-run bbs Dragonfly 128 65536  # show command only
#

set -euo pipefail

# ============================================================
#  OPTIONS
# ============================================================
DRY_RUN=0
HOST_SPEED="2000Gf"
DOCKER_IMG="bcast"

# Parse flags (before positional args)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] <algo> <topo> <N> <msg_bytes>"
            echo "       $0 [--dry-run] all"
            echo ""
            echo "Arguments:"
            echo "  algo       : mpi, srda, pipe, bine, bbs, glf, obfs, ffgb, test"
            echo "  topo       : Butterfly, Dragonfly, FatTree, 2Dmesh"
            echo "  N          : 128, 256, 512, 1024"
            echo "  msg_bytes  : e.g. 65536, 1048576, 67108864"
            echo ""
            echo "Examples:"
            echo "  $0 bbs Dragonfly 128 65536"
            echo "  $0 mpi Butterfly 256 1048576"
            echo "  $0 all                          # run all combos"
            echo "  $0 --dry-run bbs FatTree 512 4194304"
            exit 0 ;;
        -*) echo "Unknown flag: $1" >&2; exit 1 ;;
        *)  break ;;
    esac
done

# Parse positional args
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [--dry-run] <algo> <topo> <N> <msg_bytes>" >&2
    echo "       $0 [--dry-run] all" >&2
    exit 1
elif [[ $# -eq 1 && "$1" == "all" ]]; then
    ALGOS=(mpi srda pipe bine bbs)
    TOPOS=(Butterfly Dragonfly FatTree 2Dmesh)
    SIZES=(128 256 512 1024)
    MSG_SIZES=(65536 262144 1048576 4194304 16777216 67108864 134217728)
elif [[ $# -eq 4 ]]; then
    ALGOS=("$1")
    TOPOS=("$2")
    SIZES=("$3")
    MSG_SIZES=("$4")
else
    echo "ERROR: expected 4 arguments: <algo> <topo> <N> <msg_bytes>" >&2
    echo "       or: all" >&2
    exit 1
fi

# ============================================================
#  PATHS
# ============================================================
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
TOPO_DIR="$PROJ_DIR/topo"
DATA_DIR="$PROJ_DIR/data"
BIN="$PROJ_DIR/bin/runner"

# ============================================================
#  CONTAINER DETECTION: Docker > Singularity > native
# ============================================================
RUNTIME=""
SMPI_PREFIX=""
DOCKER_RUN=""

if command -v docker &>/dev/null; then
    if ! docker image inspect "$DOCKER_IMG" &>/dev/null; then
        echo "=== Building Docker image '$DOCKER_IMG' ==="
        docker build -t "$DOCKER_IMG" "$PROJ_DIR"
    fi
    RUNTIME="docker"
    DOCKER_RUN="docker run --rm -v $PROJ_DIR:/workspace -w /workspace $DOCKER_IMG"
    SMPI_PREFIX="$DOCKER_RUN"
    echo "Using: Docker ($DOCKER_IMG)"
elif command -v singularity &>/dev/null && [[ -f "$PROJ_DIR/bcast.sif" ]]; then
    RUNTIME="singularity"
    SMPI_PREFIX="singularity exec --bind $PROJ_DIR $PROJ_DIR/bcast.sif"
    echo "Using: Singularity (bcast.sif)"
elif command -v smpirun &>/dev/null; then
    RUNTIME="native"
    echo "Using: native smpirun"
else
    echo "ERROR: No container runtime (docker/singularity) or native smpirun found." >&2
    echo "Install Docker and run: docker build -t bcast ." >&2
    exit 1
fi

# ============================================================
#  HELPERS
# ============================================================
mesh_dims() {
    case "$1" in
        32)   echo "4x8"   ;; 36)   echo "6x6"   ;;
        128)  echo "8x16"  ;; 256)  echo "16x16" ;;
        512)  echo "16x32" ;; 1024) echo "32x32" ;;
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

topo_data_path() {
    local xml; xml=$(platform_path "$1" "$2")
    echo "${xml%.xml}.tdat"
}

choose_chunks() {
    local nc=$(( $1 / 8192 )); (( nc < 4 )) && nc=4; echo "$nc"
}

relpath() { echo "${1#$PROJ_DIR/}"; }

# ============================================================
#  BUILD
# ============================================================
echo "=== Building runner ==="
mkdir -p "$(dirname "$BIN")"
if [[ "$RUNTIME" == "docker" ]]; then
    $DOCKER_RUN smpicc -O2 -Wall -Wextra -o bin/runner src/runner.c -lm
elif [[ "$RUNTIME" == "singularity" ]]; then
    $SMPI_PREFIX smpicc -O2 -Wall -Wextra -o "$BIN" "$PROJ_DIR/src/runner.c" -lm
else
    make -C "$PROJ_DIR/src" -s
fi
echo "  Built: $BIN"

# ============================================================
#  PREPROCESS .tdat FILES (only for algos that need them)
# ============================================================
needs_tdat=0
for a in "${ALGOS[@]}"; do
    [[ "$a" == "bbs" || "$a" == "test" || "$a" == "ffgb" || "$a" == "obfs" ]] && needs_tdat=1
done
if [[ $needs_tdat -eq 1 ]]; then
    echo "=== Preprocessing topology data ==="
    PREPROCESS="$PROJ_DIR/src/topo_preprocess.py"
    for topo in "${TOPOS[@]}"; do
        for n in "${SIZES[@]}"; do
            xml=$(platform_path "$topo" "$n")
            tdat=$(topo_data_path "$topo" "$n")
            [[ ! -f "$xml" ]] && continue
            if [[ ! -f "$tdat" || "$xml" -nt "$tdat" ]]; then
                echo "  $xml -> $tdat"
                if [[ "$RUNTIME" == "docker" ]]; then
                    $DOCKER_RUN python3 src/topo_preprocess.py "$(relpath "$xml")" "$(relpath "$tdat")"
                elif [[ "$RUNTIME" == "singularity" ]]; then
                    singularity exec --bind "$PROJ_DIR" "$PROJ_DIR/bcast.sif" python3 "$PREPROCESS" "$xml" "$tdat"
                else
                    python3 "$PREPROCESS" "$xml" "$tdat"
                fi
            fi
        done
    done
fi

# ============================================================
#  RUN EXPERIMENTS
# ============================================================
echo ""
echo "=============================================="
echo "  Broadcast Experiments"
echo "  Topologies : ${TOPOS[*]}"
echo "  Sizes      : ${SIZES[*]}"
echo "  Algorithms : ${ALGOS[*]}"
echo "  Msg sizes  : ${MSG_SIZES[*]}"
echo "=============================================="

for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        PLAT=$(platform_path "$TOPO" "$N")
        HF="$TOPO_DIR/$TOPO/hostfile_$N"
        [[ ! -f "$PLAT" || ! -f "$HF" ]] && continue

        for ALGO in "${ALGOS[@]}"; do
            NP=$N
            ROOT_SUFFIX=""
            TOPO_ARG=""

            if [[ "$ALGO" == "bbs" || "$ALGO" == "test" || "$ALGO" == "ffgb" || "$ALGO" == "obfs" ]]; then
                TOPO_ARG=$(topo_data_path "$TOPO" "$N")
                [[ ! -f "$TOPO_ARG" ]] && { echo "SKIP $ALGO $TOPO N=$N: missing tdat"; continue; }
            elif [[ "$ALGO" == "glf" ]]; then
                TOPO_ARG="$TOPO_DIR/$TOPO/topo_${N}.cfg"
                [[ ! -f "$TOPO_ARG" ]] && { echo "SKIP glf $TOPO N=$N: missing cfg"; continue; }
            fi

            if [[ "$ALGO" == "bbs" && "$TOPO" == "FatTree" ]]; then
                CFG="$TOPO_DIR/$TOPO/topo_${N}.cfg"
                if [[ -f "$CFG" ]]; then
                    NPL=$(awk '/^fattree/ { print $2 }' "$CFG")
                    if [[ -n "$NPL" && "$NPL" -gt 0 ]]; then
                        NL=$((N / NPL))
                        NS=$((2 * (NL - 1)))
                        NP=$((N + NL + NS))
                        HF="$TOPO_DIR/$TOPO/hostfile_bbs_$N"
                        ROOT_SUFFIX=":$N"
                    fi
                fi
            elif [[ "$ALGO" == "bbs" && "$TOPO" == "Dragonfly" ]]; then
                CFG="$TOPO_DIR/$TOPO/topo_${N}.cfg"
                if [[ -f "$CFG" ]]; then
                    read -r _ DG DC DR DP < "$CFG"
                    NR=$((DG * DC * DR))
                    NP=$((N + NR))
                    HF="$TOPO_DIR/$TOPO/hostfile_bbs_$N"
                    ROOT_SUFFIX=":$N"
                fi
            fi

            for MSG in "${MSG_SIZES[@]}"; do
                NC=$(choose_chunks "$MSG")
                OUTDIR="$DATA_DIR/$TOPO/$ALGO"
                BULK="$OUTDIR/N${N}_MSG${MSG}.json"
                mkdir -p "$OUTDIR"

                [[ -f "$BULK" ]] && { echo "SKIP $ALGO $TOPO N=$N MSG=$MSG (exists)"; continue; }

                echo ""
                echo "--- $ALGO $TOPO N=$N MSG=$MSG NC=$NC NP=$NP ---"

                if [[ "$RUNTIME" == "docker" ]]; then
                    CMD="$DOCKER_RUN smpirun -np $NP"
                    CMD+=" -platform $(relpath "$PLAT") -hostfile $(relpath "$HF")"
                    CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                    CMD+=" --cfg=smpi/simulate-computation:no"
                    CMD+=" --cfg=smpi/display-timing:yes"
                    CMD+=" --log=root.thres:warning"
                    CMD+=" bin/runner $ALGO $MSG $NC all${ROOT_SUFFIX} $(relpath "$BULK")"
                    [[ -n "$TOPO_ARG" ]] && CMD+=" $(relpath "$TOPO_ARG")"
                else
                    CMD="$SMPI_PREFIX smpirun -np $NP -platform $PLAT -hostfile $HF"
                    CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                    CMD+=" --cfg=smpi/simulate-computation:no"
                    CMD+=" --cfg=smpi/display-timing:yes"
                    CMD+=" --log=root.thres:warning"
                    CMD+=" $BIN $ALGO $MSG $NC all${ROOT_SUFFIX} $BULK"
                    [[ -n "$TOPO_ARG" ]] && CMD+=" $TOPO_ARG"
                fi

                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "  $CMD"
                else
                    eval "$CMD"
                fi
            done
        done
    done
done

echo ""
echo "=== Done: $(date) ==="
