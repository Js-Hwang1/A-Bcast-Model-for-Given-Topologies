#!/bin/bash
# Sweep all 32 roots x all msg sizes for one algorithm on 4x8 2Dmesh
# Usage: ./sweep_n32.sh <algo> [nchunks]
# Example: ./sweep_n32.sh mpi
#          ./sweep_n32.sh bbs 128

ALGO="$1"
NCHUNKS="${2:-1}"
PLAT="topo/2Dmesh/platform_2dmesh_4x8.xml"
HOST="topo/2Dmesh/hostfile_32"
TDAT="topo/2Dmesh/platform_2dmesh_4x8.tdat"
OUTDIR="data/2Dmesh/${ALGO}"

MSGS="65536 262144 1048576 4194304 16777216 67108864 134217728"

mkdir -p "$OUTDIR"

export SWEEP_SMPI="smpirun -np 32 -platform $PLAT -hostfile $HOST --cfg=smpi/simulate-computation:no"
export SWEEP_JOBS=4

for MSG in $MSGS; do
    echo "=== $ALGO N=32 MSG=$MSG nchunks=$NCHUNKS ==="
    smpirun -np 32 -platform "$PLAT" -hostfile "$HOST" \
        --cfg=smpi/simulate-computation:no \
        ./src/runner "$ALGO" "$MSG" "$NCHUNKS" sweep "$OUTDIR" "$TDAT"
done
