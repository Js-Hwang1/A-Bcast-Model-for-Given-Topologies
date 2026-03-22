#!/usr/bin/env bash
# pool_worker.sh — One worker in a 1120-wide SLURM pool.
#
# Each of the 1120 workers (identified by $SLURM_PROCID 0..1119) reads
# every 1120-th line from the shared task file (round-robin interleaving),
# skips tasks whose output JSON already exists, and runs the smpirun.
#
# Args: <algo> <sif> <proj_dir> <tasks_file> <nworkers>

set -uo pipefail

ALGO="$1"
SIF="$2"
PROJ_DIR="$3"
TASKS_FILE="$4"
NWORKERS="$5"

WORKER_ID="${SLURM_PROCID:-0}"
TOPO_DIR="$PROJ_DIR/topo"
DATA_DIR="$PROJ_DIR/data"
BINARY="$PROJ_DIR/bin/runner"
HOST_SPEED="2000Gf"

mesh_dims() {
    case "$1" in
        128)  echo "8x16"  ;;
        256)  echo "16x16" ;;
        512)  echo "16x32" ;;
        1024) echo "32x32" ;;
        *)    echo "unknown" ;;
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

# Process every NWORKERS-th line starting at WORKER_ID+1 (1-indexed)
awk -v w="$WORKER_ID" -v n="$NWORKERS" '(NR - 1) % n == w' "$TASKS_FILE" | \
while IFS=' ' read -r topo N msg nc root; do
    outdir="$DATA_DIR/$topo/$ALGO"
    outfile="$outdir/N${N}_MSG${msg}_R${root}.json"

    [[ -f "$outfile" ]] && continue   # resume: skip already-done

    platform=$(platform_path "$topo" "$N")
    hostfile="$TOPO_DIR/$topo/hostfile_$N"

    [[ ! -f "$platform" || ! -f "$hostfile" ]] && continue

    mkdir -p "$outdir"

    extra=""
    [[ "$ALGO" == "glf" ]] && extra="$TOPO_DIR/$topo/topo_${N}.cfg"

    singularity exec --bind "$PROJ_DIR" "$SIF" \
        smpirun -np "$N" \
        -platform "$platform" \
        -hostfile "$hostfile" \
        --cfg=smpi/host-speed:$HOST_SPEED \
        --cfg=smpi/simulate-computation:no \
        --cfg=smpi/display-timing:yes \
        --cfg=smpi/shared-malloc-blocksize:4194304 \
        --log=root.thres:warning \
        "$BINARY" "$ALGO" "$msg" "$nc" "$root" "$outfile" $extra \
        > /dev/null 2>&1 || true   # don't kill worker on single-task failure
done
