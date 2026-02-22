#!/usr/bin/env python3
"""
Generate SimGrid platform XML and hostfile for a Butterfly topology.

Usage:  python3 gen_platform.py <num_nodes>

Outputs:
    platform_butterfly_<N>.xml  — SimGrid platform description
    hostfile_<N>                — one host per line (node-0 .. node-N-1)

Topology: Wrapped butterfly (k-ary n-fly with merged I/O stages).
    For k=2 this is equivalent to a binary hypercube:
    - N = 2^n nodes, each with n neighbors
    - Node i connects to node i XOR 2^j  for j = 0, 1, ..., n-1
    - Diameter = n, degree = n, bisection bandwidth = N/2 links

    This models the flattened butterfly topology proposed by Kim & Dally,
    which collapses multiple switch stages into direct node-to-node links.

Network parameters based on InfiniBand EDR (matching fat-tree for
fair cross-topology comparison):
    Bandwidth: 12.5 GBps  (100 Gbps EDR per port)
    Latency:   100ns      (per hop)
    Host CPU:  2000 GFlops (symbolic; broadcast is communication-bound)

References:
    - Kim & Dally, "Flattened Butterfly: A Cost-Efficient Topology
      for High-Radix Networks", ISCA 2007
      http://cva.stanford.edu/publications/2007/ISCA_FBFLY.pdf
    - Kim & Dally, "Efficient Topologies for Large-scale Cluster
      Networks", Google Research, 2008
"""

import sys
import math


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <num_nodes>", file=sys.stderr)
        print("  num_nodes must be a power of 2 (>= 4)", file=sys.stderr)
        sys.exit(1)

    n = int(sys.argv[1])
    dim = int(math.log2(n))

    if n < 4 or (1 << dim) != n:
        print(f"Error: num_nodes ({n}) must be a power of 2 >= 4",
              file=sys.stderr)
        sys.exit(1)

    # Build edge list: node i <-> node i XOR 2^j (for each dimension j)
    edges = []
    for i in range(n):
        for j in range(dim):
            neighbor = i ^ (1 << j)
            if neighbor > i:  # avoid duplicates
                edges.append((i, neighbor))

    num_edges = len(edges)  # should be n * dim / 2

    # ---- hostfile ----
    hf = f"hostfile_{n}"
    with open(hf, "w") as f:
        for i in range(n):
            f.write(f"node-{i}\n")

    # ---- platform XML ----
    fname = f"platform_butterfly_{n}.xml"
    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM '
                '"https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')
        f.write(f"  <!-- Wrapped Butterfly (Hypercube): "
                f"{n} nodes, {dim} dimensions, "
                f"{num_edges} links -->\n")
        f.write(f"  <!-- Degree = {dim}, Diameter = {dim} -->\n")
        f.write( "  <!--\n")
        f.write( "       Network: InfiniBand EDR "
                 "(matched to fat-tree for comparison)\n")
        f.write( "       Bandwidth: 12.5 GBps (100 Gbps per port)\n")
        f.write( "       Latency:   100ns per hop\n")
        f.write( "       Ref: Kim & Dally, \"Flattened Butterfly\", "
                 "ISCA 2007\n")
        f.write( "  -->\n\n")

        f.write('  <zone id="butterfly" routing="Floyd">\n\n')

        # Hosts
        f.write(f"    <!-- {n} hosts -->\n")
        for i in range(n):
            f.write(f'    <host id="node-{i}" speed="2000Gf"/>\n')
        f.write("\n")

        # Links (grouped by dimension)
        for j in range(dim):
            dim_edges = [(a, b) for a, b in edges
                         if (a ^ b) == (1 << j)]
            f.write(f"    <!-- Dimension {j}: {len(dim_edges)} links "
                    f"(stride {1 << j}) -->\n")
            for a, b in dim_edges:
                f.write(f'    <link id="link-{a}-{b}" '
                        f'bandwidth="12.5GBps" latency="100ns"/>\n')
            f.write("\n")

        # Routes
        f.write("    <!-- Direct neighbor routes "
                "(Floyd fills multi-hop) -->\n")
        for a, b in edges:
            f.write(f'    <route src="node-{a}" dst="node-{b}"> '
                    f'<link_ctn id="link-{a}-{b}"/> </route>\n')

        f.write("\n  </zone>\n")
        f.write("</platform>\n")

    print(f"Generated: {fname}, {hf}  "
          f"({n} nodes, {dim}-dimensional, {num_edges} links)")


if __name__ == "__main__":
    main()
