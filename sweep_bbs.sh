#!/bin/bash
# Usage: ./sweep_bbs.sh <platform.xml> <hostfile> <topo.tdat> <N> [nchunks] [msg_bytes] [workers] [outdir]
# Sweeps all N roots with $workers parallel smpirun processes.
# Produces per-root JSON files, then aggregates into a bulk JSON for plot_results.py.
#
# Example:
#   ./sweep_bbs.sh topo/2Dmesh/platform_2dmesh_8x16.xml topo/2Dmesh/hostfile_128 \
#       topo/2Dmesh/platform_2dmesh_8x16.tdat 128 128 1048576 4 data/2Dmesh/bbs

PLAT="$1"
HOST="$2"
TDAT="$3"
N="$4"
K="${5:-128}"
MSG="${6:-1048576}"
W="${7:-4}"
OUTDIR="${8:-data/2Dmesh/bbs}"

mkdir -p "$OUTDIR"

NP=$(wc -l < "$HOST" | tr -d ' ')
BULK="$OUTDIR/N${N}_MSG${MSG}.json"

# Skip if bulk JSON already exists
if [ -f "$BULK" ]; then
    echo "Already done: $BULK"
    exit 0
fi

echo "=== BBS N=$N MSG=$MSG K=$K workers=$W ==="

# ── Sweep all roots ──
DONE=0
START=$(date +%s)

for ((base=0; base<N; base+=W)); do
    for ((j=0; j<W && base+j<N; j++)); do
        r=$((base+j))
        RJSON="$OUTDIR/N${N}_MSG${MSG}_R${r}.json"
        # Skip if per-root JSON exists
        if [ -f "$RJSON" ]; then
            DONE=$((DONE+1))
            continue
        fi
        (
            result=$(smpirun -np "$NP" -platform "$PLAT" -hostfile "$HOST" \
                --cfg=smpi/simulate-computation:no \
                ./src/runner bbs "$MSG" "$K" "$r" _ "$TDAT" 2>&1)
            t=$(echo "$result" | awk '/time_sec/{print $3}')
            ok_raw=$(echo "$result" | awk '/^correct/{print $3}')
            if [ "$ok_raw" = "yes" ]; then ok="true"; else ok="false"; fi
            # Write per-root JSON
            cat > "$RJSON" <<EOF
{
  "algorithm": "bbs",
  "nodes": $N,
  "msg_bytes": $MSG,
  "nchunks": $K,
  "root": $r,
  "time_sec": $t,
  "correct": $ok
}
EOF
        ) &
    done
    wait

    # Progress report
    for ((j=0; j<W && base+j<N; j++)); do
        r=$((base+j))
        RJSON="$OUTDIR/N${N}_MSG${MSG}_R${r}.json"
        if [ -f "$RJSON" ]; then
            t=$(awk -F': ' '/time_sec/{gsub(/,/,"",$2); print $2}' "$RJSON")
            DONE=$((DONE+1))
            NOW=$(date +%s)
            ELAPSED=$((NOW - START))
            if [ $DONE -gt 0 ] && [ $ELAPSED -gt 0 ]; then
                ETA=$(( (N - DONE) * ELAPSED / DONE ))
            else
                ETA="?"
            fi
            printf "[%3d/%d] root=%-3d  t=%-14s  elapsed=%ds  ETA=%ss\n" \
                "$DONE" "$N" "$r" "$t" "$ELAPSED" "$ETA"
        fi
    done
done

# ── Aggregate into bulk JSON ──
echo ""
echo "Aggregating $N roots -> $BULK"

python3 -c "
import json, sys, os, math

N = $N
outdir = '$OUTDIR'
msg = $MSG
k = $K

times = []
all_ok = True
for r in range(N):
    rp = f'{outdir}/N{N}_MSG{msg}_R{r}.json'
    with open(rp) as f:
        rec = json.load(f)
    times.append(rec['time_sec'])
    if rec.get('correct') != True and rec.get('correct') != 'yes':
        all_ok = False

mean = sum(times) / N
var = sum((t - mean)**2 for t in times) / N
stdev = math.sqrt(var)

bulk = {
    'algorithm': 'bbs',
    'nodes': N,
    'msg_bytes': msg,
    'nchunks': k,
    'n_roots': N,
    'mean_sec': mean,
    'stdev_sec': stdev,
    'min_sec': min(times),
    'max_sec': max(times),
    'correct': all_ok,
    'per_root_sec': times,
}

with open('$BULK', 'w') as f:
    json.dump(bulk, f, indent=2)
print(f'  Written: $BULK')
print(f'  mean={mean*1e6:.1f} us  min={min(times)*1e6:.1f} us  max={max(times)*1e6:.1f} us')
"

# Clean up per-root files
for ((r=0; r<N; r++)); do
    rm -f "$OUTDIR/N${N}_MSG${MSG}_R${r}.json"
done
echo "  Cleaned up per-root JSONs."
echo "Done."
