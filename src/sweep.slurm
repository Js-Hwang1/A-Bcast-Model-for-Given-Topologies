#!/bin/bash
#SBATCH --job-name=bcast-sweep
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err
#SBATCH --mail-type=END,FAIL
#
# sweep.slurm — Full experiment sweep using the Singularity container.
#
# Partition: h200x4-long (CPU-only, no GPUs requested)
# GNU parallel runs on the host and dispatches each smpirun
# through the container for dynamic load balancing.
#
# Prerequisites:
#   sbatch src/build.slurm   # wait for bcast.sif to appear
#
# Usage:
#   module load slurm
#   cd /path/to/A-Bcast-Model-for-Given-Topologies
#   sbatch src/sweep.slurm
#   sbatch src/sweep.slurm --algos mpi,srda

set -euo pipefail

PROJ_DIR="$SLURM_SUBMIT_DIR"
SIF="$PROJ_DIR/bcast.sif"
SRC_DIR="$PROJ_DIR/src"

echo "=== Broadcast Sweep ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Container : $SIF"
echo "Start     : $(date)"
echo ""

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container not found at $SIF" >&2
    echo "  Build it first:  sbatch src/build.slurm" >&2
    exit 1
fi

# Compile runner inside the container (uses smpicc from container)
echo "Building runner..."
singularity exec --bind "$PROJ_DIR" "$SIF" \
    smpicc -O2 -Wall -Wextra -o "$SRC_DIR/runner" "$SRC_DIR/runner.c" -lm
echo ""

# Run the sweep: parallel dispatches on host, each smpirun goes through container
"$SRC_DIR/sweep.sh" \
    -j "${SLURM_CPUS_PER_TASK:-64}" \
    --sif "$SIF" \
    "$@"

echo ""
echo "End time  : $(date)"
