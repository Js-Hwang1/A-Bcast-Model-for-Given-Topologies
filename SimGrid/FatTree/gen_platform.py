#!/usr/bin/env python3
"""
Generate SimGrid platform XML and hostfile for a Fat-Tree topology.

Usage:  python3 gen_platform.py <num_nodes> [hosts_per_leaf]

Outputs:
    platform_fattree_<N>.xml  — SimGrid platform description
    hostfile_<N>              — one host per line (node-0 .. node-N-1)

Topology: 2-level non-blocking fat-tree (full bisection bandwidth)
    - num_leaf  = N / hosts_per_leaf   leaf switches
    - num_spine = hosts_per_leaf       spine switches
    - Each leaf switch connects to ALL spine switches (non-blocking)

Network parameters based on InfiniBand EDR (ORNL Summit / LLNL Sierra):
    Bandwidth: 12.5 GBps  (100 Gbps EDR per port)
    Latency:   100ns      (per switch hop)
    Host CPU:  2000 GFlops (symbolic; broadcast is communication-bound)

References:
    - Summit (ORNL): 4,608 nodes, Mellanox EDR dual-rail fat-tree
      https://www.olcf.ornl.gov/olcf-resources/compute-systems/summit/
    - Sierra (LLNL): 4,320 nodes, Mellanox EDR fat-tree
      https://en.wikipedia.org/wiki/Sierra_(supercomputer)
    - Dell/Mellanox: "Comparing FDR and EDR InfiniBand" white paper
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <num_nodes> [hosts_per_leaf]",
              file=sys.stderr)
        print("  num_nodes must be >= 4 and divisible by hosts_per_leaf",
              file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])
    hpl = int(sys.argv[2]) if len(sys.argv) > 2 else 16  # hosts per leaf

    # Auto-adjust for small N
    while hpl > n // 2 and hpl > 2:
        hpl //= 2

    if n < 4 or n % hpl != 0:
        print(f"Error: num_nodes ({n}) must be >= 4 and divisible by "
              f"hosts_per_leaf ({hpl})", file=sys.stderr)
        sys.exit(1)

    num_leaf = n // hpl
    num_spine = hpl  # non-blocking: uplinks == downlinks per leaf

    # SimGrid FAT_TREE topo_parameters:
    #   "levels ; m1,m2,...  ; w1,w2,...  ; p1,p2,..."
    # m_i = children per switch at level i
    # w_i = parents per node at level i-1
    # p_i = parallel links at level i
    # Total hosts = m1 * m2
    m1 = hpl
    m2 = num_leaf
    w1 = 1          # each host connects to 1 leaf switch
    w2 = m1         # non-blocking: each leaf has m1 uplinks
    p1 = 1
    p2 = 1
    topo = f"2 ; {m1},{m2} ; {w1},{w2} ; {p1},{p2}"

    # ---- hostfile ----
    hf = f"hostfile_{n}"
    with open(hf, "w") as f:
        for i in range(n):
            f.write(f"node-{i}\n")

    # ---- platform XML ----
    fname = f"platform_fattree_{n}.xml"
    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM '
                '"https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')
        f.write(f"  <!-- Fat-Tree: {n} compute nodes, "
                f"2-level non-blocking -->\n")
        f.write(f"  <!-- {num_leaf} leaf switches "
                f"({hpl} hosts each) + "
                f"{num_spine} spine switches -->\n")
        f.write( "  <!--\n")
        f.write( "       Network: InfiniBand EDR "
                 "(ORNL Summit / LLNL Sierra)\n")
        f.write( "       Bandwidth: 12.5 GBps (100 Gbps per port)\n")
        f.write( "       Latency:   100ns per switch hop\n")
        f.write( "       Ref: Mellanox EDR dual-rail fat-tree\n")
        f.write( "  -->\n\n")

        f.write(f'  <cluster id="fattree" prefix="node-" suffix=""\n')
        f.write(f'           radical="0-{n-1}"\n')
        f.write( '           speed="2000Gf"\n')
        f.write( '           bw="12.5GBps" lat="100ns"\n')
        f.write( '           topology="FAT_TREE"\n')
        f.write(f'           topo_parameters="{topo}"/>\n\n')

        f.write("</platform>\n")

    print(f"Generated: {fname}, {hf}  "
          f"({n} nodes, {num_leaf} leaf + {num_spine} spine switches)")


if __name__ == "__main__":
    main()
