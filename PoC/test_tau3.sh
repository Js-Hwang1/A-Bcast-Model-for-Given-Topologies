#!/bin/bash
set -e
BASE=/Users/j/Desktop/A-Bcast-Model-for-Given-Topologies
ENC=$BASE/BBS/encode.c
PLAT=$BASE/topo/FatTree/platform_fattree_128.xml
HF=$BASE/topo/FatTree/hostfile_152
TOPO=$BASE/topo/FatTree/topo_128.cfg
RUNNER=$BASE/bin/runner

printf "%-8s %-6s" "msg" "K"
for TAU in 1 2 3 4 5 7; do printf "  tau=%-7d" "$TAU"; done
echo

for MSG in 65536 262144 524288 1048576; do
    K=$((MSG / 8192))
    printf "%-8d %-6d" "$MSG" "$K"
    for TAU in 1 2 3 4 5 7; do
        sed -i '' "s/int tau = [0-9]*/int tau = $TAU/" "$ENC"
        rm -f "$RUNNER"; make -C "$BASE/src" 2>/dev/null
        rm -rf "$BASE/encodings/FatTree/N=152/"; mkdir -p "$BASE/encodings/FatTree/N=152/"

        T=$(smpirun --cfg=smpi/simulate-computation:no \
            -np 152 -platform "$PLAT" -hostfile "$HF" \
            "$RUNNER" bbs "$MSG" "$K" 0 _ "$TOPO" 2>&1 \
            | grep time_sec | awk '{print $3}')
        T_US=$(printf "%.1f" "$(echo "$T * 1000000" | bc -l)")
        printf "  %-10s" "$T_US"
    done
    echo
done

# Restore tau=3
sed -i '' "s/int tau = [0-9]*/int tau = 3/" "$ENC"
rm -f "$RUNNER"; make -C "$BASE/src" 2>/dev/null
