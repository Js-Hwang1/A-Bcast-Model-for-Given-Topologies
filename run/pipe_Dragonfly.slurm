#!/bin/bash
#SBATCH --job-name=pipe-Dragonfly
#SBATCH --partition=long-28core
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=run/pipe_Dragonfly_%j.out
#SBATCH --error=run/pipe_Dragonfly_%j.err
#SBATCH --mail-type=END,FAIL
#
# Pipe sweep: Dragonfly, N=128..1024, MSG=64KB..128MB (skip N=1024/128MB)
# O(N*msg) memory: SimGrid 4.1 privatizes globals by default.
# nchunks = MSG / 8192
#
# Usage:
#   source /etc/profile && module load slurm
#   cd /gpfs/scratch/$USER/A-Bcast-Model-for-Given-Topologies
#   sbatch run/pipe_Dragonfly.slurm

set -euo pipefail

PROJ_DIR="$SLURM_SUBMIT_DIR"
SIF="$PROJ_DIR/bcast.sif"
BIN="$PROJ_DIR/bin/runner"
TOPO_DIR="$PROJ_DIR/topo"
DATA="$PROJ_DIR/data"
ALGO=pipe
TOPO=Dragonfly

echo "=== Pipe sweep: $TOPO — N=128..1024, MSG=64KB..128MB ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Nodes  : $SLURM_JOB_NODELIST"
echo "Start  : $(date)"

[[ ! -f "$SIF" ]] && { echo "ERROR: Container not found: $SIF" >&2; exit 1; }

if [[ ! -x "$BIN" ]]; then
    echo "Building runner..."
    mkdir -p "$(dirname "$BIN")"
    singularity exec --bind "$PROJ_DIR" "$SIF" \
        smpicc -O2 -Wall -Wextra -o "$BIN" "$PROJ_DIR/src/runner.c" -lm
fi

NODES=( $(scontrol show hostnames "$SLURM_NODELIST") )
NNODES=${#NODES[@]}

SMPI="singularity exec --bind $PROJ_DIR $SIF smpirun"

# Delete old pipe data for this topology (we are re-running)
echo "Cleaning old pipe data for $TOPO..."
rm -f "$DATA/$TOPO/$ALGO"/N*_MSG*.json

for N in 128 256 512 1024; do
    PLAT="$TOPO_DIR/$TOPO/platform_dragonfly_${N}.xml"
    HF="$TOPO_DIR/$TOPO/hostfile_$N"

    [[ ! -f "$PLAT" || ! -f "$HF" ]] && { echo "SKIP N=$N: missing platform/hostfile"; continue; }

    for MSG in 65536 262144 1048576 4194304 16777216 67108864 134217728; do
        [[ $N -eq 1024 && $MSG -eq 134217728 ]] && continue

        NC=$(( MSG / 8192 ))
        (( NC < 4 )) && NC=4

        OUTDIR="$DATA/$TOPO/$ALGO"
        BULK="$OUTDIR/N${N}_MSG${MSG}.json"
        mkdir -p "$OUTDIR"

        if [[ -f "$BULK" ]]; then
            echo "--- N=$N MSG=$MSG SKIP (exists) ---"
            continue
        fi

        MEM_MB=$(( (N * MSG) / 1048576 + N * 2 + 256 ))
        (( MEM_MB < 512 )) && MEM_MB=512
        W=$(( 110000 / MEM_MB ))
        (( W > 27 )) && W=27
        (( W < 1 )) && W=1

        echo ""
        echo "--- N=$N MSG=$MSG NC=$NC mem/worker=${MEM_MB}MB workers/node=$W ---"

        COMMON="-np $N -platform $PLAT -hostfile $HF"
        COMMON+=" --cfg=smpi/simulate-computation:no"
        COMMON+=" --cfg=smpi/display-timing:yes"
        COMMON+=" --log=root.thres:warning"

        JOBFILE="$OUTDIR/.jobs_${SLURM_JOB_ID}_N${N}_M${MSG}"
        > "$JOBFILE"

        for ROOT in $(seq 0 $((N - 1))); do
            RJSON="$OUTDIR/N${N}_MSG${MSG}_R${ROOT}.json"
            [[ -f "$RJSON" ]] && continue
            echo "$SMPI $COMMON $BIN $ALGO $MSG $NC $ROOT $RJSON" >> "$JOBFILE"
        done

        NJOBS=$(wc -l < "$JOBFILE" | tr -d ' ')
        if [[ $NJOBS -eq 0 ]]; then
            echo "  All roots already done, aggregating..."
        else
            echo "  Dispatching $NJOBS roots across $NNODES nodes ($W workers each)..."

            JOBS_PER_NODE=$(( (NJOBS + NNODES - 1) / NNODES ))
            split -l "$JOBS_PER_NODE" -d "$JOBFILE" "${JOBFILE}_"

            i=0
            for chunk in "${JOBFILE}_"*; do
                [[ $i -ge $NNODES ]] && break
                ssh "${NODES[$i]}" "bash $PROJ_DIR/run/dispatch.sh $chunk $W $PROJ_DIR" &
                i=$((i + 1))
            done
            wait

            rm -f "${JOBFILE}_"*
        fi
        rm -f "$JOBFILE"

        MISSING=0
        for ROOT in $(seq 0 $((N - 1))); do
            [[ ! -f "$OUTDIR/N${N}_MSG${MSG}_R${ROOT}.json" ]] && { MISSING=1; break; }
        done

        if [[ $MISSING -eq 1 ]]; then
            echo "  WARNING: some roots missing, skipping aggregation"
            continue
        fi

        python3 - "$OUTDIR" "$N" "$MSG" "$ALGO" "$BULK" "$NC" << 'PYEOF'
import json, sys, math, os
outdir, n, msg, algo, bp, nc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
times, ok = [], True
for r in range(n):
    with open(os.path.join(outdir, f"N{n}_MSG{msg}_R{r}.json")) as f:
        d = json.load(f)
    times.append(d["time_sec"])
    if not d.get("correct", True): ok = False
mean = sum(times)/len(times)
std = math.sqrt(sum((t-mean)**2 for t in times)/len(times))
with open(bp, "w") as f:
    json.dump({"algorithm":algo,"nodes":n,"msg_bytes":msg,"nchunks":nc,
        "n_roots":n,"mean_sec":mean,"stdev_sec":std,
        "min_sec":min(times),"max_sec":max(times),
        "correct":ok,"per_root_sec":times}, f, indent=2)
print(f"  N={n} MSG={msg}: mean={mean:.6e} max={max(times):.6e} {'OK' if ok else 'FAIL'}")
PYEOF

        for ROOT in $(seq 0 $((N - 1))); do
            rm -f "$OUTDIR/N${N}_MSG${MSG}_R${ROOT}.json"
        done
    done
done

echo ""
echo "=== All done: $(date) ==="
