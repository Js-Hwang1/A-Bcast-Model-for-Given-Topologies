#!/usr/bin/env bash
#
# run.sh — Compile and run broadcast experiments locally (no SLURM).
#
# Hard-coded config below. Edit the arrays to choose what to run.
# Handles BBS special cases (extra ranks for FatTree/Dragonfly).
#
# Usage:
#   chmod +x run.sh && ./run.sh
#   ./run.sh --dry-run        # show commands without executing
#   ./run.sh --jobs 8         # limit parallelism
#

set -euo pipefail

# ============================================================
#  CONFIGURATION — edit these
# ============================================================
TOPOS=(Butterfly Dragonfly FatTree 2Dmesh)
SIZES=(128 256 512 1024)
ALGOS=(mpi srda pipe bine bbs)
MSG_SIZES=(65536 262144 1048576 4194304 16777216 67108864 134217728)
HOST_SPEED="2000Gf"

# ============================================================
#  OPTIONS
# ============================================================
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
DRY_RUN=0
SIF=""    # set to container path if using singularity, e.g. SIF="bcast.sif"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --jobs|-j)  JOBS="$2"; shift 2 ;;
        --sif)      SIF="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--jobs N] [--sif container.sif]"
            echo "Edit the TOPOS/SIZES/ALGOS/MSG_SIZES arrays at the top of this script."
            exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ============================================================
#  PATHS
# ============================================================
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
TOPO_DIR="$PROJ_DIR/topo"
DATA_DIR="$PROJ_DIR/data"
BIN="$PROJ_DIR/bin/runner"

SMPI_PREFIX=""
if [[ -n "$SIF" ]]; then
    [[ ! -f "$SIF" ]] && { echo "ERROR: SIF not found: $SIF" >&2; exit 1; }
    SMPI_PREFIX="singularity exec --bind $PROJ_DIR $SIF"
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

# ============================================================
#  BUILD
# ============================================================
echo "=== Building runner ==="
mkdir -p "$(dirname "$BIN")"
if [[ -n "$SMPI_PREFIX" ]]; then
    $SMPI_PREFIX smpicc -O2 -Wall -Wextra -o "$BIN" "$PROJ_DIR/src/runner.c" -lm
else
    make -C "$PROJ_DIR/src" -s
fi
echo "  Built: $BIN"

# ============================================================
#  PREPROCESS .tdat FILES
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
                if [[ -n "$SMPI_PREFIX" ]]; then
                    singularity exec --bind "$PROJ_DIR" "$SIF" python3 "$PREPROCESS" "$xml" "$tdat"
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
echo "  Msg sizes  : ${#MSG_SIZES[@]}"
echo "  Workers    : $JOBS"
echo "=============================================="

SMPI_RUN="$SMPI_PREFIX smpirun"

for TOPO in "${TOPOS[@]}"; do
    for N in "${SIZES[@]}"; do
        PLAT=$(platform_path "$TOPO" "$N")
        HF="$TOPO_DIR/$TOPO/hostfile_$N"
        [[ ! -f "$PLAT" || ! -f "$HF" ]] && continue

        for ALGO in "${ALGOS[@]}"; do
            # --- Compute NP, hostfile, root suffix for BBS special topologies ---
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
                # Skip known OOM cases
                MEM_EST=$(( NP * MSG / 1048576 ))
                if (( MEM_EST > 120000 )); then
                    echo "SKIP $ALGO $TOPO N=$N MSG=$MSG (est ${MEM_EST}MB > 120GB)"
                    continue
                fi

                NC=$(choose_chunks "$MSG")
                OUTDIR="$DATA_DIR/$TOPO/$ALGO"
                BULK="$OUTDIR/N${N}_MSG${MSG}.json"
                mkdir -p "$OUTDIR"

                [[ -f "$BULK" ]] && { echo "SKIP $ALGO $TOPO N=$N MSG=$MSG (exists)"; continue; }

                echo ""
                echo "--- $ALGO $TOPO N=$N MSG=$MSG NC=$NC NP=$NP ---"

                CMD="$SMPI_RUN -np $NP -platform $PLAT -hostfile $HF"
                CMD+=" --cfg=smpi/host-speed:$HOST_SPEED"
                CMD+=" --cfg=smpi/simulate-computation:no"
                CMD+=" --cfg=smpi/display-timing:yes"
                CMD+=" --log=root.thres:warning"
                CMD+=" $BIN $ALGO $MSG $NC all${ROOT_SUFFIX} $BULK"
                [[ -n "$TOPO_ARG" ]] && CMD+=" $TOPO_ARG"

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
