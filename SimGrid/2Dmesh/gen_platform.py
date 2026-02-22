#!/usr/bin/env python3
"""
Generate SimGrid platform XML and hostfile for a PxQ 2D mesh.

Usage:  python3 gen_platform.py <rows> <cols>

Outputs:
    platform_2dmesh_PxQ.xml   — SimGrid platform description
    hostfile                  — one host per line (node-0 .. node-N-1)

Network parameters: InfiniBand NDR 400 (50 GBps, 500 ns latency)
"""

import sys


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <rows> <cols>", file=sys.stderr)
        sys.exit(1)

    p, q = int(sys.argv[1]), int(sys.argv[2])
    n = p * q

    # ---- hostfile ----
    with open("hostfile", "w") as f:
        for i in range(n):
            f.write(f"node-{i}\n")

    # ---- platform XML ----
    fname = f"platform_2dmesh_{p}x{q}.xml"
    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')

        # ASCII art
        f.write(f"  <!-- {p}x{q} 2D Mesh ({n} nodes, "
                f"{(p-1)*q + p*(q-1)} links) -->\n\n")

        f.write('  <zone id="mesh2d" routing="Floyd">\n\n')

        # Hosts
        f.write(f"    <!-- {n} hosts -->\n")
        for i in range(n):
            f.write(f'    <host id="node-{i}" speed="2000Gf"/>\n')
        f.write("\n")

        # Links — horizontal
        h_links = []
        for r in range(p):
            for c in range(q - 1):
                a, b = r * q + c, r * q + c + 1
                h_links.append((a, b))

        # Links — vertical
        v_links = []
        for r in range(p - 1):
            for c in range(q):
                a, b = r * q + c, (r + 1) * q + c
                v_links.append((a, b))

        f.write(f"    <!-- {len(h_links)} horizontal links "
                f"(InfiniBand NDR 400) -->\n")
        for a, b in h_links:
            f.write(f'    <link id="link-{a}-{b}" '
                    f'bandwidth="50GBps" latency="500ns"/>\n')

        f.write(f"\n    <!-- {len(v_links)} vertical links -->\n")
        for a, b in v_links:
            f.write(f'    <link id="link-{a}-{b}" '
                    f'bandwidth="50GBps" latency="500ns"/>\n')
        f.write("\n")

        # Routes
        f.write("    <!-- Direct neighbor routes (Floyd fills multi-hop) -->\n")
        for a, b in h_links + v_links:
            f.write(f'    <route src="node-{a}" dst="node-{b}"> '
                    f'<link_ctn id="link-{a}-{b}"/> </route>\n')

        f.write("\n  </zone>\n")
        f.write("</platform>\n")

    print(f"Generated: {fname}, hostfile  ({n} nodes, "
          f"{len(h_links) + len(v_links)} links)")


if __name__ == "__main__":
    main()
