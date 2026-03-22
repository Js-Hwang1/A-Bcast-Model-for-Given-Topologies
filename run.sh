#!/bin/bash
#SBATCH --job-name=bbs64mb2dmesh
#SBATCH --partition=hbm-large-96core
#SBATCH --nodes=31
#SBATCH --ntasks=31
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --output=hello.out
#SBATCH --error=hello.err
#SBATCH --mail-type=END,FAIL
# pipe / 2Dmesh / N=1024 / MSG=64MB / all roots
# 31 nodes x 2 workers, 384GB RAM per node

set -euo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
PROJ_DIR="/gpfs/scratch/jungshwang/A-Bcast-Model-for-Given-Topologies"
SIF="$PROJ_DIR/bcast.sif"
N=1024; NNODES=31; WORKERS=1; ALGO=bbs; TOPO=2Dmesh
MSG=67108864 # 64MB
NC=$(( MSG / 8192 ))
OUTDIR="$PROJ_DIR/data/$TOPO/$ALGO"

echo "=== $TOPO $ALGO N=$N MSG=$MSG NC=$NC ==="
echo "Job ID: $SLURM_JOB_ID | Start: $(date) | Nodes: $SLURM_JOB_NODELIST"
[[ ! -f "$SIF" ]] && { echo "ERROR: SIF not found" >&2; exit 1; }
cd "$PROJ_DIR"; mkdir -p "$OUTDIR"

ROOTS_PER_NODE=$(( N / NNODES ))
REMAINDER=$(( N % NNODES ))

for (( i=0; i<NNODES; i++ )); do
    ROOT_LO=$(( i * ROOTS_PER_NODE + (i < REMAINDER ? i : REMAINDER) ))
    ROOT_HI=$(( ROOT_LO + ROOTS_PER_NODE + (i < REMAINDER ? 1 : 0) - 1 ))
    [[ $ROOT_LO -gt $ROOT_HI ]] && continue
    srun --nodes=1 --ntasks=1 --exclusive \
        bash src/sweep.sh --algos $ALGO --topos $TOPO --sizes $N \
            --msgs $MSG --chunks $NC --roots ${ROOT_LO}-${ROOT_HI} \
            --sif "$SIF" -j $WORKERS &
done
wait

BULK="$OUTDIR/N${N}_MSG${MSG}.json"
if [[ -f "$BULK" ]]; then
    echo "Bulk exists (sweep.sh aggregated): $BULK"
    for ((r=0; r<N; r++)); do rm -f "$OUTDIR/N${N}_MSG${MSG}_R${r}.json"; done
else
    python3 -c "
import json, math
N,outdir,msg,algo,nc=$N,'$OUTDIR',$MSG,'$ALGO',$NC
times,ok=[],True
for r in range(N):
    with open(f'{outdir}/N{N}_MSG{msg}_R{r}.json') as f: rec=json.load(f)
    times.append(rec['time_sec'])
    if not rec.get('correct',True): ok=False
mean=sum(times)/N; stdev=math.sqrt(sum((t-mean)**2 for t in times)/N)
bulk={'algorithm':algo,'nodes':N,'msg_bytes':msg,'nchunks':nc,'n_roots':N,
      'mean_sec':mean,'stdev_sec':stdev,'min_sec':min(times),'max_sec':max(times),
      'correct':ok,'per_root_sec':times}
with open('$BULK','w') as f: json.dump(bulk,f,indent=2)
print(f'  Aggregated: mean={mean*1e3:.2f}ms  min={min(times)*1e3:.2f}ms  max={max(times)*1e3:.2f}ms')
"
    for ((r=0; r<N; r++)); do rm -f "$OUTDIR/N${N}_MSG${MSG}_R${r}.json"; done
fi
echo "Done: $(date)"
