#!/usr/bin/env python3
"""Generate BBS-specific hostfiles for Dragonfly topologies.

BBS makes every router an MPI relay rank so the broadcast pipeline
stages through them explicitly (same approach as FatTree leaf/spine).

Rank mapping:
  0 .. Nc-1           : compute nodes  (node-0 .. node-(Nc-1))
  Nc .. Nc+Nr-1       : router ranks   (router-G-C-R in order)

Router rank = Nc + g*(NCHASSIS*NROUTERS) + c*NROUTERS + r
Nr = G * C * R
NP = Nc + Nr

Usage: python3 gen_bbs_hostfile.py <Nc>
Output: hostfile content to stdout
"""

import sys
import os

# Dragonfly configs: Nc -> (G, C, R, P)
DRAGONFLY_CONFIGS = {
    128:  (4,  4, 4, 2),
    256:  (4,  4, 4, 4),
    512:  (8,  4, 4, 4),
    1024: (16, 4, 4, 4),
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <Nc>", file=sys.stderr)
        sys.exit(1)

    Nc = int(sys.argv[1])

    if Nc not in DRAGONFLY_CONFIGS:
        # Try reading from cfg file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, f"topo_{Nc}.cfg")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                parts = f.read().split()
                # "dragonfly G C R P"
                if len(parts) == 5 and parts[0] == "dragonfly":
                    G, C, R, P = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    DRAGONFLY_CONFIGS[Nc] = (G, C, R, P)

    if Nc not in DRAGONFLY_CONFIGS:
        print(f"ERROR: unknown Nc={Nc}, no config found", file=sys.stderr)
        sys.exit(1)

    G, C, R, P = DRAGONFLY_CONFIGS[Nc]
    Nr = G * C * R
    NP = Nc + Nr

    if G * C * R * P != Nc:
        print(f"ERROR: G*C*R*P={G*C*R*P} != Nc={Nc}", file=sys.stderr)
        sys.exit(1)

    print(f"# BBS Dragonfly hostfile: Nc={Nc} Nr={Nr} NP={NP} "
          f"G={G} C={C} R={R} P={P}", file=sys.stderr)

    # Compute nodes
    for i in range(Nc):
        print(f"node-{i}")

    # Router ranks
    for g in range(G):
        for c in range(C):
            for r in range(R):
                print(f"router-{g}-{c}-{r}")


if __name__ == "__main__":
    main()
