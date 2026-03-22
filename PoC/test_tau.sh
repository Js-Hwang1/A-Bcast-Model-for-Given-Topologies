#!/bin/bash
# Test different tau values for FatTree BBS
set -e
BASE=/Users/j/Desktop/A-Bcast-Model-for-Given-Topologies
ENC=$BASE/BBS/encode.c
PLAT=$BASE/topo/FatTree/platform_fattree_128.xml
HF=$BASE/topo/FatTree/hostfile_152
TOPO=$BASE/topo/FatTree/topo_128.cfg
RUNNER=$BASE/bin/runner
MSG=1048576
NCHUNKS=32

printf "%-5s %-12s %-12s\n" "tau" "time_us" "theoretical"

for TAU in 1 2 3 4 5 6 7; do
    # Patch tau value
    sed -i '' "s/int tau = [0-9]*/int tau = $TAU/" "$ENC"

    # Rebuild
    rm -f "$RUNNER"
    make -C "$BASE/src" 2>/dev/null

    # Clear BBS cache
    rm -rf "$BASE/encodings/FatTree/N=152/"
    mkdir -p "$BASE/encodings/FatTree/N=152/"

    # Run single root=0
    T=$(smpirun --cfg=smpi/simulate-computation:no \
        -np 152 -platform "$PLAT" -hostfile "$HF" \
        "$RUNNER" bbs "$MSG" "$NCHUNKS" 0 _ "$TOPO" 2>&1 \
        | grep time_sec | awk '{print $3}')

    T_US=$(echo "$T * 1000000" | bc -l)

    # Theoretical: chunk/B * (tau*(d-1) + K)
    THEORY=$(echo "scale=1; 32768.0/12500000000.0 * ($TAU * 3 + 32) * 1000000" | bc -l)

    printf "%-5d %-12.1f %-12.1f\n" "$TAU" "$T_US" "$THEORY"
done

# Restore tau=3
sed -i '' "s/int tau = [0-9]*/int tau = 3/" "$ENC"
rm -f "$RUNNER"
make -C "$BASE/src" 2>/dev/null
