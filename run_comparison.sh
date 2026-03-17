#!/bin/bash
cd /Users/j/Desktop/A-Bcast-Model-for-Given-Topologies

PLATFORM="topo/2Dmesh/platform_2dmesh_8x16.xml"
HOSTFILE="topo/2Dmesh/hostfile_128"
TDAT="topo/2Dmesh/platform_2dmesh_8x16.tdat"
MSG=1048576

echo "=== N=128, MSG=1MB (1048576 bytes), 2Dmesh 8x16 ==="
echo ""
printf "%-6s  %-8s  %12s  %-7s\n" "Root" "Algo" "Time (ms)" "Correct"
printf "%-6s  %-8s  %12s  %-7s\n" "------" "--------" "------------" "-------"

for ROOT in 0 7 60 64 120 127; do
  for ALGO in bbs srda mpi; do
    if [ "$ALGO" = "bbs" ]; then
      OUT=$(smpirun -np 128 -platform "$PLATFORM" -hostfile "$HOSTFILE" \
        --cfg=smpi/host-speed:2000Gf --log=root.thres:warning \
        src/runner "$ALGO" "$MSG" 64 "$ROOT" _ "$TDAT" 2>/dev/null)
    else
      OUT=$(smpirun -np 128 -platform "$PLATFORM" -hostfile "$HOSTFILE" \
        --cfg=smpi/host-speed:2000Gf --log=root.thres:warning \
        src/runner "$ALGO" "$MSG" 64 "$ROOT" 2>/dev/null)
    fi
    TIME=$(echo "$OUT" | grep "time_sec" | sed 's/.*: //')
    CORRECT=$(echo "$OUT" | grep "correct" | sed 's/.*: //')
    TIME_MS=$(python3 -c "print(f'{float(\"$TIME\")*1000:.4f}')" 2>/dev/null || echo "ERROR")
    printf "%-6s  %-8s  %12s  %-7s\n" "$ROOT" "$ALGO" "$TIME_MS" "$CORRECT"
  done
done
