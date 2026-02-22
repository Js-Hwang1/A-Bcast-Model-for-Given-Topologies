#!/usr/bin/env python3
"""
Generate SimGrid platform XML and hostfile for a Dragonfly topology.

Usage:  python3 gen_platform.py <num_nodes>

Outputs:
    platform_dragonfly_<N>.xml  — SimGrid platform description
    hostfile_<N>                — one host per line (node-0 .. node-N-1)

Topology: Dragonfly (Cray Aries-style)
    Total nodes = groups × chassis × routers × nodes_per_router
    - Intra-chassis: all-to-all between routers (green links)
    - Inter-chassis: all-to-all between chassis in a group (black links)
    - Inter-group:   links between each pair of groups (blue links)

Network parameters based on Cray Aries (Theta XC40 / Piz Daint XC50):
    Bandwidth: 5.25 GBps  (Aries electrical link, per direction)
    Latency:   100ns      (per router hop)
    Host CPU:  2000 GFlops (symbolic; broadcast is communication-bound)

References:
    - Theta (ALCF): 4,392 KNL nodes, Cray XC40 Aries dragonfly
      https://www.alcf.anl.gov/alcf-resources/theta
    - Piz Daint (CSCS): 5,704 nodes, Cray XC50 Aries dragonfly
      https://www.cscs.ch/computers/piz-daint
    - Kim et al., "Technology-Driven, Highly-Scalable Dragonfly Topology"
      ISCA 2008
    - Cray XC Network white paper (ALCF):
      https://www.alcf.anl.gov/files/CrayXCNetwork.pdf
"""

import sys


# Pre-computed configurations: target_N -> (groups, chassis, routers, nodes)
# Chosen so that groups * chassis * routers * nodes == exact_N
# and the dragonfly structure is well-balanced.
CONFIGS = {
    128:  (4, 4, 4, 2),     # 4 groups, 4 chassis, 4 routers, 2 nodes = 128
    256:  (4, 4, 4, 4),     # 4 groups, 4 chassis, 4 routers, 4 nodes = 256
    512:  (8, 4, 4, 4),     # 8 groups, 4 chassis, 4 routers, 4 nodes = 512
    1024: (16, 4, 4, 4),    # 16 groups, 4 chassis, 4 routers, 4 nodes = 1024
}


def find_config(n):
    """Find a balanced dragonfly config for n nodes."""
    if n in CONFIGS:
        return CONFIGS[n]
    # Try to factor n = g * c * r * p with balanced dimensions
    best = None
    for p in [2, 4, 8]:
        if n % p != 0:
            continue
        rem = n // p
        for r in [2, 4, 8]:
            if rem % r != 0:
                continue
            rem2 = rem // r
            for c in [2, 4, 6, 8]:
                if rem2 % c != 0:
                    continue
                g = rem2 // c
                if g >= 2:
                    cfg = (g, c, r, p)
                    if best is None or abs(g - c*r) < abs(best[0] - best[1]*best[2]):
                        best = cfg
    if best is None:
        print(f"Error: cannot factor {n} into a balanced dragonfly. "
              f"Use one of: {sorted(CONFIGS.keys())}", file=sys.stderr)
        sys.exit(1)
    return best


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <num_nodes>", file=sys.stderr)
        print(f"  Recommended sizes: {sorted(CONFIGS.keys())}",
              file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])
    groups, chassis, routers, nodes_per_router = find_config(n)
    actual_n = groups * chassis * routers * nodes_per_router

    if actual_n != n:
        print(f"Warning: adjusted to {actual_n} nodes "
              f"(requested {n})", file=sys.stderr)
        n = actual_n

    # SimGrid DRAGONFLY topo_parameters:
    #   "groups,blue ; chassis,black ; routers,green ; nodes"
    # blue  = inter-group links per group pair
    # black = inter-chassis links per chassis pair within a group
    # green = intra-chassis links per router pair within a chassis
    blue  = 1   # 1 link between each pair of groups
    black = 1   # 1 link between each pair of chassis in a group
    green = 1   # 1 link between each pair of routers in a chassis
    topo = (f"{groups},{blue} ; {chassis},{black} ; "
            f"{routers},{green} ; {nodes_per_router}")

    # ---- hostfile ----
    hf = f"hostfile_{n}"
    with open(hf, "w") as f:
        for i in range(n):
            f.write(f"node-{i}\n")

    # ---- platform XML ----
    fname = f"platform_dragonfly_{n}.xml"
    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM '
                '"https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')
        f.write(f"  <!-- Dragonfly: {n} compute nodes -->\n")
        f.write(f"  <!-- {groups} groups x {chassis} chassis x "
                f"{routers} routers x {nodes_per_router} nodes -->\n")
        f.write( "  <!--\n")
        f.write( "       Network: Cray Aries "
                 "(Theta XC40 / Piz Daint XC50)\n")
        f.write( "       Bandwidth: 5.25 GBps "
                 "(Aries electrical, per direction)\n")
        f.write( "       Latency:   100ns per router hop\n")
        f.write(f"       Inter-group (blue) links:   "
                f"{blue} per group pair\n")
        f.write(f"       Inter-chassis (black) links: "
                f"{black} per chassis pair\n")
        f.write(f"       Intra-chassis (green) links: "
                f"{green} per router pair\n")
        f.write( "       Ref: Kim et al., ISCA 2008; "
                 "Cray XC Network white paper\n")
        f.write( "  -->\n\n")

        f.write(f'  <cluster id="dragonfly" prefix="node-" suffix=""\n')
        f.write(f'           radical="0-{n-1}"\n')
        f.write( '           speed="2000Gf"\n')
        f.write( '           bw="5.25GBps" lat="100ns"\n')
        f.write( '           topology="DRAGONFLY"\n')
        f.write(f'           topo_parameters="{topo}"/>\n\n')

        f.write("</platform>\n")

    print(f"Generated: {fname}, {hf}  "
          f"({n} nodes: {groups}g x {chassis}c x {routers}r x "
          f"{nodes_per_router}n)")


if __name__ == "__main__":
    main()
