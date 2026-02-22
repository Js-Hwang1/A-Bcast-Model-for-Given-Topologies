#!/bin/bash
#SBATCH --job-name=bcast-backfill
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --output=backfill_%j.out
#SBATCH --error=backfill_%j.err
#SBATCH --mail-type=END,FAIL
#
# backfill.slurm — Re-run only missing experiments with low concurrency.
#
# The main sweep OOM-killed large-message jobs (16MB, 64MB) when running
# 32 concurrently.  This job re-runs only the missing ones with -j 4
# so at most 4 large smpirun processes share 300G RAM.
#
# Usage:
#   module load slurm
#   sbatch src/backfill.slurm

set -euo pipefail

PROJ_DIR="$SLURM_SUBMIT_DIR"
SIF="$PROJ_DIR/bcast.sif"
SRC_DIR="$PROJ_DIR/src"

echo "=== Broadcast Backfill ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Container : $SIF"
echo "Start     : $(date)"
echo ""

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container not found at $SIF" >&2
    exit 1
fi

# Compile runner inside the container
echo "Building runner..."
singularity exec --bind "$PROJ_DIR" "$SIF" \
    smpicc -O2 -Wall -Wextra -o "$SRC_DIR/runner" "$SRC_DIR/runner.c" -lm
echo ""

# Run backfill with low concurrency (4 workers to avoid OOM)
# Extra args (e.g. --topos 2Dmesh) passed via sbatch src/backfill.slurm --topos X
"$SRC_DIR/backfill.sh" \
    -j 4 \
    --sif "$SIF" \
    "$@"

echo ""
echo "End time  : $(date)"
