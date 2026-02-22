#!/bin/bash
#SBATCH --job-name=bcast-build
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err
#
# build.slurm — Pull the SimGrid container on a compute node.
#
# Uses singularity pull (no root/fakeroot needed).
# The bcast.def is kept for reference but is not used here.
#
# Usage (from login node):
#   module load slurm
#   cd /path/to/A-Bcast-Model-for-Given-Topologies
#   sbatch slurm/build.slurm

set -euo pipefail

PROJ_DIR="$SLURM_SUBMIT_DIR"
SIF="$PROJ_DIR/bcast.sif"
SRC_DIR="$PROJ_DIR/src"

echo "=== Container Build ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Output  : $SIF"
echo "Start   : $(date)"
echo ""

# Pull SimGrid stable image from Docker Hub (no fakeroot needed)
echo "Pulling simgrid/stable from Docker Hub..."
singularity pull --force "$SIF" docker://simgrid/stable:latest

echo ""
echo "=== Pull complete ==="
echo "Image   : $SIF"
echo "Size    : $(du -h "$SIF" | cut -f1)"
echo ""

# Smoke test
echo "=== Smoke test ==="
singularity exec --bind "$PROJ_DIR" "$SIF" smpicc --version  || true
singularity exec --bind "$PROJ_DIR" "$SIF" smpirun --version || true
echo ""

# Compile the runner inside the container (no make in image, use smpicc directly)
echo "=== Compiling runner ==="
mkdir -p "$PROJ_DIR/bin"
rm -f "$PROJ_DIR/bin/runner"
singularity exec --bind "$PROJ_DIR" "$SIF" \
    smpicc -O2 -Wall -Wextra -o "$PROJ_DIR/bin/runner" "$SRC_DIR/runner.c" -lm
echo ""

echo "=== Done ==="
echo "Container : $SIF"
echo "Runner    : $PROJ_DIR/bin/runner"
echo "End       : $(date)"
echo ""
echo "Next step: sbatch slurm/sweep.slurm"
