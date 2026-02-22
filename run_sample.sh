#!/bin/bash
# Quick ~100 datapoint BBS experiment run
set -euo pipefail

TOPOS=(2Dmesh Butterfly FatTree)
MSGS=(1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456)
N=128
ROOT=0

for topo in "${TOPOS[@]}"; do
  mkdir -p "data/$topo/bbs"
  PLATFORM=""
  HOSTFILE="topo/$topo/hostfile_${N}"
  case $topo in
    2Dmesh)    PLATFORM="topo/2Dmesh/platform_2dmesh_8x16.xml" ;;
    Butterfly) PLATFORM="topo/Butterfly/platform_butterfly_${N}.xml" ;;
    Dragonfly) PLATFORM="topo/Dragonfly/platform_dragonfly_${N}.xml" ;;
    FatTree)   PLATFORM="topo/FatTree/platform_fattree_${N}.xml" ;;
  esac
  PLAN="plans/$topo/${N}_root${ROOT}.plan"
  PARAMS="plans/$topo/${N}_root${ROOT}.params"

  for msg in "${MSGS[@]}"; do
    NC=$(python3 -c "
import json; p=json.load(open('$PARAMS'))
k=p.get('optimal_K',{}).get('$msg',{}).get('K_opt')
print(k if k else 4)" 2>/dev/null || echo 4)
    OUTJSON="data/$topo/bbs/N${N}_MSG${msg}_R${ROOT}.json"

    echo -n "$topo MSG=$msg NC=$NC => "
    smpirun -np "$N" -platform "$PLATFORM" -hostfile "$HOSTFILE" \
      --cfg=smpi/host-speed:2000Gf --log=root.thres:warning \
      src/runner bbs "$msg" "$NC" "$ROOT" "$PLAN" "$OUTJSON" 2>/dev/null \
      | grep -E 'rounds|time_sec|correct' | tr '\n' ' '
    echo ""
  done
done
