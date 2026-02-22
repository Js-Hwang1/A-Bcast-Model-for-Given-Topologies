#!/usr/bin/env python3
"""
Unified topology generator for SimGrid platform XMLs and hostfiles.

Usage:
    python3 topology_generater.py <topology> <nodes>
    python3 topology_generater.py --all
    python3 topology_generater.py --all --plot

Topologies: 2Dmesh, Butterfly, Dragonfly, FatTree

Outputs to ../topo/<Topology>/platform_<topo>_<N>.xml + hostfile_<N>

Node counts:
    2Dmesh:     128 (8x16), 256 (16x16), 512 (16x32), 1024 (32x32)
    Butterfly:  128, 256, 512, 1024  (power-of-2 hypercubes)
    Dragonfly:  128, 256, 512, 1024  (Cray Aries configs)
    FatTree:    128, 256, 512, 1024  (2-level non-blocking)
"""

import sys
import os
import math
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPO_DIR = os.path.join(SCRIPT_DIR, "..", "topo")

# ---- 2D Mesh sizing: N -> (rows, cols) ----
MESH_SIZES = {
    128:  (8, 16),
    256:  (16, 16),
    512:  (16, 32),
    1024: (32, 32),
}

# ---- Dragonfly configs: N -> (groups, chassis, routers, nodes_per_router) ----
DRAGONFLY_CONFIGS = {
    128:  (4, 4, 4, 2),
    256:  (4, 4, 4, 4),
    512:  (8, 4, 4, 4),
    1024: (16, 4, 4, 4),
}

ALL_TOPOS = ["2Dmesh", "Butterfly", "Dragonfly", "FatTree"]
ALL_SIZES = [128, 256, 512, 1024]


# ============================================================
# Hostfile writer
# ============================================================
def write_hostfile(outdir, n):
    path = os.path.join(outdir, f"hostfile_{n}")
    with open(path, "w") as f:
        for i in range(n):
            f.write(f"node-{i}\n")
    return path


# ============================================================
# 2D Mesh generator
# ============================================================
def gen_2dmesh(n, outdir):
    if n not in MESH_SIZES:
        print(f"Error: 2Dmesh supports {sorted(MESH_SIZES.keys())}, got {n}",
              file=sys.stderr)
        sys.exit(1)

    p, q = MESH_SIZES[n]

    hf = write_hostfile(outdir, n)
    fname = os.path.join(outdir, f"platform_2dmesh_{p}x{q}.xml")

    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')

        h_links = []
        for r in range(p):
            for c in range(q - 1):
                a, b = r * q + c, r * q + c + 1
                h_links.append((a, b))

        v_links = []
        for r in range(p - 1):
            for c in range(q):
                a, b = r * q + c, (r + 1) * q + c
                v_links.append((a, b))

        f.write(f"  <!-- {p}x{q} 2D Mesh ({n} nodes, "
                f"{len(h_links) + len(v_links)} links) -->\n")
        f.write("  <!-- Network: InfiniBand NDR 400 (50 GBps, 100 ns) -->\n\n")
        f.write('  <zone id="mesh2d" routing="Floyd">\n\n')

        f.write(f"    <!-- {n} hosts -->\n")
        for i in range(n):
            f.write(f'    <host id="node-{i}" speed="2000Gf"/>\n')
        f.write("\n")

        f.write(f"    <!-- {len(h_links)} horizontal links "
                f"(InfiniBand NDR 400) -->\n")
        for a, b in h_links:
            f.write(f'    <link id="link-{a}-{b}" '
                    f'bandwidth="50GBps" latency="100ns"/>\n')

        f.write(f"\n    <!-- {len(v_links)} vertical links -->\n")
        for a, b in v_links:
            f.write(f'    <link id="link-{a}-{b}" '
                    f'bandwidth="50GBps" latency="100ns"/>\n')
        f.write("\n")

        f.write("    <!-- Direct neighbor routes (Floyd fills multi-hop) -->\n")
        for a, b in h_links + v_links:
            f.write(f'    <route src="node-{a}" dst="node-{b}"> '
                    f'<link_ctn id="link-{a}-{b}"/> </route>\n')

        f.write("\n  </zone>\n")
        f.write("</platform>\n")

    print(f"  2Dmesh {p}x{q}: {fname}")
    return fname


# ============================================================
# Butterfly (Hypercube) generator
# ============================================================
def gen_butterfly(n, outdir):
    dim = int(math.log2(n))
    if n < 4 or (1 << dim) != n:
        print(f"Error: Butterfly needs power-of-2 >= 4, got {n}",
              file=sys.stderr)
        sys.exit(1)

    edges = []
    for i in range(n):
        for j in range(dim):
            neighbor = i ^ (1 << j)
            if neighbor > i:
                edges.append((i, neighbor))

    hf = write_hostfile(outdir, n)
    fname = os.path.join(outdir, f"platform_butterfly_{n}.xml")

    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM '
                '"https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')
        f.write(f"  <!-- Wrapped Butterfly (Hypercube): "
                f"{n} nodes, {dim} dimensions, {len(edges)} links -->\n")
        f.write(f"  <!-- Degree = {dim}, Diameter = {dim} -->\n")
        f.write("  <!-- Network: InfiniBand EDR (12.5 GBps, 100 ns) -->\n")
        f.write("  <!-- Ref: Kim & Dally, \"Flattened Butterfly\", "
                "ISCA 2007 -->\n\n")

        f.write('  <zone id="butterfly" routing="Floyd">\n\n')

        f.write(f"    <!-- {n} hosts -->\n")
        for i in range(n):
            f.write(f'    <host id="node-{i}" speed="2000Gf"/>\n')
        f.write("\n")

        for j in range(dim):
            dim_edges = [(a, b) for a, b in edges if (a ^ b) == (1 << j)]
            f.write(f"    <!-- Dimension {j}: {len(dim_edges)} links "
                    f"(stride {1 << j}) -->\n")
            for a, b in dim_edges:
                f.write(f'    <link id="link-{a}-{b}" '
                        f'bandwidth="12.5GBps" latency="100ns"/>\n')
            f.write("\n")

        f.write("    <!-- Direct neighbor routes (Floyd fills multi-hop) -->\n")
        for a, b in edges:
            f.write(f'    <route src="node-{a}" dst="node-{b}"> '
                    f'<link_ctn id="link-{a}-{b}"/> </route>\n')

        f.write("\n  </zone>\n")
        f.write("</platform>\n")

    print(f"  Butterfly {dim}D: {fname}")
    return fname


# ============================================================
# Dragonfly generator
# ============================================================
def gen_dragonfly(n, outdir):
    if n not in DRAGONFLY_CONFIGS:
        print(f"Error: Dragonfly supports {sorted(DRAGONFLY_CONFIGS.keys())}, got {n}",
              file=sys.stderr)
        sys.exit(1)

    G, C, R, P = DRAGONFLY_CONFIGS[n]
    total_routers = G * C * R
    num_green = G * C * (R * (R - 1) // 2)
    num_black = G * (C * (C - 1) // 2)
    num_blue = G * (G - 1) // 2

    hf = write_hostfile(outdir, n)
    fname = os.path.join(outdir, f"platform_dragonfly_{n}.xml")

    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM '
                '"https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')
        f.write(f"  <!-- Dragonfly: {n} compute nodes -->\n")
        f.write(f"  <!-- {G} groups x {C} chassis x "
                f"{R} routers x {P} nodes -->\n")
        f.write(f"  <!-- {total_routers} routers, "
                f"{num_green} green + {num_black} black + "
                f"{num_blue} blue links -->\n")
        f.write("  <!-- Network: Cray Aries (5.25 GBps) -->\n")
        f.write("  <!-- Latency: intra-chassis 100ns, "
                "inter-chassis 200ns, inter-group 400ns -->\n")
        f.write("  <!-- Ref: Kim et al., ISCA 2008; "
                "Cray XC Network white paper -->\n\n")

        f.write('  <zone id="dragonfly" routing="Floyd">\n\n')

        # Compute hosts
        f.write(f"    <!-- {n} compute hosts -->\n")
        for i in range(n):
            f.write(f'    <host id="node-{i}" speed="2000Gf"/>\n')
        f.write("\n")

        # Routers
        f.write(f"    <!-- {total_routers} routers -->\n")
        for g in range(G):
            for c in range(C):
                for r in range(R):
                    f.write(f'    <host id="router-{g}-{c}-{r}" '
                            f'speed="1Gf"/>\n')
        f.write("\n")

        # Node→Router links
        f.write(f"    <!-- {n} node-to-router links -->\n")
        node_id = 0
        for g in range(G):
            for c in range(C):
                for r in range(R):
                    for p in range(P):
                        f.write(
                            f'    <link id="link-node{node_id}'
                            f'-router{g}-{c}-{r}" '
                            f'bandwidth="5.25GBps" latency="100ns"/>\n')
                        node_id += 1
        f.write("\n")

        # Green links (intra-chassis: all-to-all within chassis)
        f.write(f"    <!-- {num_green} green links "
                f"(intra-chassis, 100ns) -->\n")
        for g in range(G):
            for c in range(C):
                for r1 in range(R):
                    for r2 in range(r1 + 1, R):
                        f.write(
                            f'    <link id="link-green-{g}-{c}-{r1}-{r2}" '
                            f'bandwidth="5.25GBps" latency="100ns"/>\n')
        f.write("\n")

        # Black links (inter-chassis: first router of each chassis pair)
        f.write(f"    <!-- {num_black} black links "
                f"(inter-chassis, 200ns) -->\n")
        for g in range(G):
            for c1 in range(C):
                for c2 in range(c1 + 1, C):
                    f.write(
                        f'    <link id="link-black-{g}-{c1}-{c2}" '
                        f'bandwidth="5.25GBps" latency="200ns"/>\n')
        f.write("\n")

        # Blue links (inter-group: first router of each group pair)
        f.write(f"    <!-- {num_blue} blue links "
                f"(inter-group, 400ns) -->\n")
        for g1 in range(G):
            for g2 in range(g1 + 1, G):
                f.write(
                    f'    <link id="link-blue-{g1}-{g2}" '
                    f'bandwidth="5.25GBps" latency="400ns"/>\n')
        f.write("\n")

        # Routes: node↔router
        f.write("    <!-- Direct routes (Floyd computes multi-hop) -->\n")
        node_id = 0
        for g in range(G):
            for c in range(C):
                for r in range(R):
                    for p in range(P):
                        f.write(
                            f'    <route src="node-{node_id}" '
                            f'dst="router-{g}-{c}-{r}"> '
                            f'<link_ctn id="link-node{node_id}'
                            f'-router{g}-{c}-{r}"/> </route>\n')
                        node_id += 1

        # Routes: green (intra-chassis)
        for g in range(G):
            for c in range(C):
                for r1 in range(R):
                    for r2 in range(r1 + 1, R):
                        f.write(
                            f'    <route src="router-{g}-{c}-{r1}" '
                            f'dst="router-{g}-{c}-{r2}"> '
                            f'<link_ctn id="link-green-{g}-{c}-{r1}-{r2}"/>'
                            f' </route>\n')

        # Routes: black (inter-chassis)
        for g in range(G):
            for c1 in range(C):
                for c2 in range(c1 + 1, C):
                    f.write(
                        f'    <route src="router-{g}-{c1}-0" '
                        f'dst="router-{g}-{c2}-0"> '
                        f'<link_ctn id="link-black-{g}-{c1}-{c2}"/>'
                        f' </route>\n')

        # Routes: blue (inter-group)
        for g1 in range(G):
            for g2 in range(g1 + 1, G):
                f.write(
                    f'    <route src="router-{g1}-0-0" '
                    f'dst="router-{g2}-0-0"> '
                    f'<link_ctn id="link-blue-{g1}-{g2}"/>'
                    f' </route>\n')

        f.write("\n  </zone>\n")
        f.write("</platform>\n")

    print(f"  Dragonfly {G}g x {C}c x {R}r x {P}n: {fname}")
    return fname


# ============================================================
# Fat-Tree generator
# ============================================================
def gen_fattree(n, outdir):
    if n < 4:
        print(f"Error: FatTree needs >= 4 nodes, got {n}", file=sys.stderr)
        sys.exit(1)

    hpl = 16
    while hpl > n // 2 and hpl > 2:
        hpl //= 2
    if n % hpl != 0:
        print(f"Error: {n} not divisible by hosts_per_leaf={hpl}",
              file=sys.stderr)
        sys.exit(1)

    num_leaf = n // hpl
    num_spine = hpl
    total_links = n + num_leaf * num_spine

    hf = write_hostfile(outdir, n)
    fname = os.path.join(outdir, f"platform_fattree_{n}.xml")

    with open(fname, "w") as f:
        f.write("<?xml version='1.0'?>\n")
        f.write('<!DOCTYPE platform SYSTEM '
                '"https://simgrid.org/simgrid.dtd">\n')
        f.write('<platform version="4.1">\n\n')
        f.write(f"  <!-- Fat-Tree: {n} compute nodes, "
                f"2-level non-blocking -->\n")
        f.write(f"  <!-- {num_leaf} leaf switches ({hpl} hosts each) + "
                f"{num_spine} spine switches -->\n")
        f.write(f"  <!-- {total_links} links total -->\n")
        f.write("  <!-- Network: InfiniBand EDR (12.5 GBps, 100 ns) -->\n")
        f.write("  <!-- Ref: ORNL Summit / LLNL Sierra -->\n\n")

        f.write('  <zone id="fattree" routing="Floyd">\n\n')

        # Compute hosts
        f.write(f"    <!-- {n} compute hosts -->\n")
        for i in range(n):
            f.write(f'    <host id="node-{i}" speed="2000Gf"/>\n')
        f.write("\n")

        # Leaf switches (modeled as hosts)
        f.write(f"    <!-- {num_leaf} leaf switches -->\n")
        for j in range(num_leaf):
            f.write(f'    <host id="leaf-{j}" speed="1Gf"/>\n')
        f.write("\n")

        # Spine switches (modeled as hosts)
        f.write(f"    <!-- {num_spine} spine switches -->\n")
        for k in range(num_spine):
            f.write(f'    <host id="spine-{k}" speed="1Gf"/>\n')
        f.write("\n")

        # Host→Leaf links
        f.write(f"    <!-- {n} host-to-leaf links "
                f"(InfiniBand EDR) -->\n")
        for i in range(n):
            leaf = i // hpl
            f.write(f'    <link id="link-node{i}-leaf{leaf}" '
                    f'bandwidth="12.5GBps" latency="100ns"/>\n')
        f.write("\n")

        # Leaf→Spine links (full mesh)
        f.write(f"    <!-- {num_leaf * num_spine} leaf-to-spine links -->\n")
        for j in range(num_leaf):
            for k in range(num_spine):
                f.write(f'    <link id="link-leaf{j}-spine{k}" '
                        f'bandwidth="12.5GBps" latency="100ns"/>\n')
        f.write("\n")

        # Routes: host↔leaf
        f.write("    <!-- Direct routes (Floyd computes multi-hop) -->\n")
        for i in range(n):
            leaf = i // hpl
            f.write(f'    <route src="node-{i}" dst="leaf-{leaf}"> '
                    f'<link_ctn id="link-node{i}-leaf{leaf}"/> </route>\n')

        # Routes: leaf↔spine
        for j in range(num_leaf):
            for k in range(num_spine):
                f.write(f'    <route src="leaf-{j}" dst="spine-{k}"> '
                        f'<link_ctn id="link-leaf{j}-spine{k}"/> </route>\n')

        f.write("\n  </zone>\n")
        f.write("</platform>\n")

    print(f"  FatTree {num_leaf}L+{num_spine}S: {fname}")
    return fname


# ============================================================
# Dispatch
# ============================================================
GENERATORS = {
    "2Dmesh":    gen_2dmesh,
    "Butterfly": gen_butterfly,
    "Dragonfly": gen_dragonfly,
    "FatTree":   gen_fattree,
}


def generate(topo_name, n, do_plot=False):
    if topo_name not in GENERATORS:
        print(f"Error: unknown topology '{topo_name}'. "
              f"Choose from: {ALL_TOPOS}", file=sys.stderr)
        sys.exit(1)

    outdir = os.path.join(TOPO_DIR, topo_name)
    os.makedirs(outdir, exist_ok=True)
    xml_path = GENERATORS[topo_name](n, outdir)

    if do_plot:
        plot_script = os.path.join(SCRIPT_DIR, "plot_topology.py")
        if os.path.exists(plot_script):
            png_path = xml_path.replace(".xml", ".png")
            os.system(f'python3 "{plot_script}" "{xml_path}" "{png_path}"')
        else:
            print(f"  Warning: plot_topology.py not found, skipping plot")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SimGrid platform XMLs for broadcast experiments")
    parser.add_argument("topology", nargs="?",
                        help=f"Topology name: {', '.join(ALL_TOPOS)}")
    parser.add_argument("nodes", nargs="?", type=int,
                        help="Number of nodes (128, 256, 512, 1024)")
    parser.add_argument("--all", action="store_true",
                        help="Generate all topologies x all sizes")
    parser.add_argument("--plot", action="store_true",
                        help="Also generate PNG visualizations")
    args = parser.parse_args()

    if args.all:
        print("Generating all topologies...")
        for topo in ALL_TOPOS:
            for n in ALL_SIZES:
                generate(topo, n, args.plot)
        print("Done.")
    elif args.topology and args.nodes:
        generate(args.topology, args.nodes, args.plot)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
