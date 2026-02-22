#!/usr/bin/env python3
"""
Visualize SimGrid platform topology XMLs as publication-quality PNGs.

Usage:
    python3 plot_topology.py <platform.xml> [output.png]

Supports: FAT_TREE, DRAGONFLY, BUTTERFLY (hypercube)

Produces 300 DPI PNGs suitable for conference/journal papers.
"""

import sys
import os
import re
import math
import xml.etree.ElementTree as ET
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


# ──────────────────────────── XML parsing ────────────────────────────

def parse_xml(path):
    """Return (topo_type, num_hosts, extra_data)."""
    tree = ET.parse(path)
    root = tree.getroot()

    # <cluster> element → FatTree or Dragonfly (legacy format)
    for elem in root.iter("cluster"):
        topo = (elem.get("topology") or "").upper()
        params = elem.get("topo_parameters", "")
        radical = elem.get("radical", "")
        m = re.match(r"(\d+)-(\d+)", radical)
        n = int(m.group(2)) + 1 if m else 0
        return topo, n, params

    # Explicit <zone> — detect by zone id attribute
    for zone in root.iter("zone"):
        zone_id = (zone.get("id") or "").lower()
        hosts = [e for e in zone if e.tag == "host"]

        if zone_id == "fattree":
            # Count node-*, leaf-*, spine-* hosts
            nodes = [h for h in hosts
                     if h.get("id", "").startswith("node-")]
            leaves = [h for h in hosts
                      if h.get("id", "").startswith("leaf-")]
            spines = [h for h in hosts
                      if h.get("id", "").startswith("spine-")]
            n = len(nodes)
            num_leaf = len(leaves)
            num_spine = len(spines)
            hpl = n // num_leaf if num_leaf else n
            # Reconstruct params string for _parse_fattree_params()
            m1, m2 = hpl, num_leaf
            w1, w2 = 1, m1
            params = f"2 ; {m1},{m2} ; {w1},{w2} ; 1,1"
            return "FAT_TREE", n, params

        if zone_id == "dragonfly":
            # Parse router-g-c-r IDs to find G, C, R, P
            nodes = [h for h in hosts
                     if h.get("id", "").startswith("node-")]
            routers = [h for h in hosts
                       if h.get("id", "").startswith("router-")]
            n = len(nodes)
            max_g = max_c = max_r = 0
            for rh in routers:
                parts = rh.get("id", "").split("-")
                # router-g-c-r
                max_g = max(max_g, int(parts[1]))
                max_c = max(max_c, int(parts[2]))
                max_r = max(max_r, int(parts[3]))
            G, C, R = max_g + 1, max_c + 1, max_r + 1
            P = n // len(routers) if routers else 1
            # Reconstruct params string for _parse_dragonfly_params()
            params = f"{G},1 ; {C},1 ; {R},1 ; {P}"
            return "DRAGONFLY", n, params

        if zone_id == "mesh2d":
            # No plotter for mesh yet; return early
            n = len([h for h in hosts
                     if h.get("id", "").startswith("node-")])
            return "MESH2D", n, None

        # Default: Butterfly (zone id="butterfly" or legacy)
        n = len(hosts)
        edges = []
        for lk in (e for e in zone if e.tag == "link"):
            lid = lk.get("id", "")
            em = re.match(r"link-(\d+)-(\d+)", lid)
            if em:
                edges.append((int(em.group(1)), int(em.group(2))))
        return "BUTTERFLY", n, edges

    sys.exit("Error: could not detect topology type in XML")


# ─────────────────────── adaptive sizing helpers ────────────────────

def _node_size(n, base=60):
    """Shrink nodes for large topologies."""
    if n <= 64:
        return base
    if n <= 256:
        return base * 64 / n
    return max(2, base * 64 / n)


def _edge_lw(n, base=0.8):
    if n <= 64:
        return base
    if n <= 256:
        return base * 0.5
    return max(0.08, base * 32 / n)


def _edge_alpha(n, base=0.7):
    if n <= 128:
        return base
    if n <= 256:
        return 0.45
    if n <= 512:
        return 0.30
    return 0.18


# ───────────────────────── FAT-TREE plotter ─────────────────────────

def _parse_fattree_params(n, params_str):
    parts = [p.strip() for p in params_str.split(";")]
    ms = [int(x) for x in parts[1].split(",")]
    ws = [int(x) for x in parts[2].split(",")]
    m1 = ms[0]
    w2 = ws[1]
    m2 = ms[1]
    num_leaf = n // m1
    num_spine = (num_leaf * w2) // m2
    return m1, num_leaf, num_spine


def plot_fattree(n, params_str, ax):
    m1, num_leaf, num_spine = _parse_fattree_params(n, params_str)

    margin = 0.02
    usable = 1.0 - 2 * margin

    # ── positions ──
    host_x = np.linspace(margin, margin + usable, n)
    host_y = np.zeros(n)

    leaf_x = np.array([
        host_x[i * m1:(i + 1) * m1].mean() for i in range(num_leaf)
    ])
    leaf_y = np.ones(num_leaf) * 1.0

    spine_x = np.linspace(margin, margin + usable, num_spine)
    spine_y = np.ones(num_spine) * 2.0

    # ── edges: host → leaf ──
    hl_segs = []
    for h in range(n):
        leaf = h // m1
        hl_segs.append([(host_x[h], host_y[h]),
                         (leaf_x[leaf], leaf_y[leaf])])

    # ── edges: leaf → spine ──
    ls_segs = []
    for lf in range(num_leaf):
        for sp in range(num_spine):
            ls_segs.append([(leaf_x[lf], leaf_y[lf]),
                             (spine_x[sp], spine_y[sp])])

    lw = _edge_lw(n)
    al = _edge_alpha(n)

    ax.add_collection(LineCollection(
        hl_segs, colors="#9ecae1", linewidths=lw * 0.6,
        alpha=al * 0.7, zorder=1))
    ax.add_collection(LineCollection(
        ls_segs, colors="#fc9272", linewidths=lw,
        alpha=al, zorder=1))

    ns = _node_size(n)

    ax.scatter(host_x, host_y, s=ns, c="#2171b5", edgecolors="none",
               zorder=3, label=f"Compute nodes ({n})")
    ax.scatter(leaf_x, leaf_y, s=ns * 3, c="#ef6548", marker="s",
               edgecolors="k", linewidths=0.3, zorder=3,
               label=f"Leaf switches ({num_leaf})")
    ax.scatter(spine_x, spine_y, s=ns * 4, c="#7a0177", marker="s",
               edgecolors="k", linewidths=0.3, zorder=3,
               label=f"Spine switches ({num_spine})")

    ax.set_ylim(-0.35, 2.6)
    ax.set_xlim(-0.02, 1.04)

    # layer labels
    ax.text(-0.01, 0, "Hosts", ha="right", va="center",
            fontsize=8, fontstyle="italic", color="#555")
    ax.text(-0.01, 1.0, "Leaf", ha="right", va="center",
            fontsize=8, fontstyle="italic", color="#555")
    ax.text(-0.01, 2.0, "Spine", ha="right", va="center",
            fontsize=8, fontstyle="italic", color="#555")

    ax.set_title(
        f"Fat-Tree  |  {n} nodes  |  "
        f"{num_leaf} leaf + {num_spine} spine switches\n"
        f"InfiniBand EDR 12.5 GBps, 100 ns  "
        f"(Summit / Sierra)",
        fontsize=10, fontweight="bold", pad=10)

    legend = ax.legend(loc="lower right", fontsize=7,
                       framealpha=0.9, edgecolor="#ccc")
    legend.set_zorder(5)


# ──────────────────────── DRAGONFLY plotter ─────────────────────────

def _parse_dragonfly_params(params_str):
    parts = [p.strip() for p in params_str.split(";")]
    g_parts = [int(x) for x in parts[0].split(",")]
    c_parts = [int(x) for x in parts[1].split(",")]
    r_parts = [int(x) for x in parts[2].split(",")]
    nodes_per_router = int(parts[3])
    return (g_parts[0], g_parts[1],    # groups, blue links
            c_parts[0], c_parts[1],    # chassis, black links
            r_parts[0], r_parts[1],    # routers, green links
            nodes_per_router)


def plot_dragonfly(n, params_str, ax):
    (num_groups, blue, num_chassis, black,
     num_routers, green, nodes_per_router) = _parse_dragonfly_params(
        params_str)

    routers_per_group = num_chassis * num_routers
    total_routers = num_groups * routers_per_group

    # ── group centers on a circle ──
    group_radius = 3.5 if num_groups <= 8 else 4.5
    group_cx = np.array([
        group_radius * math.cos(2 * math.pi * g / num_groups - math.pi / 2)
        for g in range(num_groups)])
    group_cy = np.array([
        group_radius * math.sin(2 * math.pi * g / num_groups - math.pi / 2)
        for g in range(num_groups)])

    # ── place routers inside each group in a small grid ──
    inner_r = 0.6 if routers_per_group <= 16 else 0.9
    router_pos = {}   # global_router_id -> (x, y)
    compute_pos = {}  # global_compute_id -> (x, y)

    for g in range(num_groups):
        cx, cy = group_cx[g], group_cy[g]
        for c in range(num_chassis):
            for r in range(num_routers):
                idx = c * num_routers + r
                angle = 2 * math.pi * idx / routers_per_group - math.pi / 2
                rx = cx + inner_r * math.cos(angle)
                ry = cy + inner_r * math.sin(angle)
                gid = g * routers_per_group + idx
                router_pos[gid] = (rx, ry)

                # place compute nodes in a tiny ring around the router
                for nd in range(nodes_per_router):
                    cid = gid * nodes_per_router + nd
                    na = angle + (nd - nodes_per_router / 2 + 0.5) * 0.15
                    nr = inner_r * 0.55
                    compute_pos[cid] = (cx + nr * math.cos(na),
                                        cy + nr * math.sin(na))

    # ── edges ──
    # Intra-chassis (green): all-to-all routers within chassis
    green_segs = []
    for g in range(num_groups):
        for c in range(num_chassis):
            base = g * routers_per_group + c * num_routers
            for r1 in range(num_routers):
                for r2 in range(r1 + 1, num_routers):
                    p1 = router_pos[base + r1]
                    p2 = router_pos[base + r2]
                    green_segs.append([p1, p2])

    # Inter-chassis (black): link between chassis within group
    black_segs = []
    for g in range(num_groups):
        for c1 in range(num_chassis):
            for c2 in range(c1 + 1, num_chassis):
                r1_id = g * routers_per_group + c1 * num_routers
                r2_id = g * routers_per_group + c2 * num_routers
                black_segs.append([router_pos[r1_id],
                                   router_pos[r2_id]])

    # Inter-group (blue): link between each pair of groups
    blue_segs = []
    for g1 in range(num_groups):
        for g2 in range(g1 + 1, num_groups):
            r1_id = g1 * routers_per_group
            r2_id = g2 * routers_per_group
            blue_segs.append([router_pos[r1_id], router_pos[r2_id]])

    # Host-to-router
    hr_segs = []
    for cid, pos in compute_pos.items():
        rid = cid // nodes_per_router
        hr_segs.append([pos, router_pos[rid]])

    lw = _edge_lw(n)
    al = _edge_alpha(n)

    if hr_segs:
        ax.add_collection(LineCollection(
            hr_segs, colors="#bdbdbd", linewidths=lw * 0.3,
            alpha=al * 0.5, zorder=1))
    if green_segs:
        ax.add_collection(LineCollection(
            green_segs, colors="#2ca02c", linewidths=lw * 1.2,
            alpha=al, zorder=2, label="Intra-chassis"))
    if black_segs:
        ax.add_collection(LineCollection(
            black_segs, colors="#333333", linewidths=lw * 1.0,
            alpha=al * 0.9, zorder=2, label="Inter-chassis"))
    if blue_segs:
        ax.add_collection(LineCollection(
            blue_segs, colors="#1f77b4", linewidths=lw * 1.5,
            alpha=min(1.0, al * 1.3), zorder=2, label="Inter-group"))

    # ── draw group boundary circles ──
    for g in range(num_groups):
        circle = plt.Circle(
            (group_cx[g], group_cy[g]),
            inner_r * 1.35, fill=False,
            edgecolor="#cccccc", linewidth=0.6,
            linestyle="--", zorder=0)
        ax.add_patch(circle)
        ax.text(group_cx[g], group_cy[g] + inner_r * 1.55,
                f"G{g}", ha="center", va="center",
                fontsize=6, color="#888", fontweight="bold")

    # ── draw nodes ──
    ns = _node_size(n)

    # Compute nodes
    if compute_pos:
        cxs = [p[0] for p in compute_pos.values()]
        cys = [p[1] for p in compute_pos.values()]
        # Color by group
        c_groups = [cid // (routers_per_group * nodes_per_router)
                     for cid in compute_pos.keys()]
        cmap = plt.cm.tab20 if num_groups > 10 else plt.cm.tab10
        colors = [cmap(g % cmap.N) for g in c_groups]
        ax.scatter(cxs, cys, s=ns * 0.5, c=colors,
                   edgecolors="none", alpha=0.7, zorder=3)

    # Router nodes
    rxs = [p[0] for p in router_pos.values()]
    rys = [p[1] for p in router_pos.values()]
    r_groups = [rid // routers_per_group for rid in router_pos.keys()]
    cmap_r = plt.cm.tab20 if num_groups > 10 else plt.cm.tab10
    colors_r = [cmap_r(g % cmap_r.N) for g in r_groups]
    ax.scatter(rxs, rys, s=ns * 2, c=colors_r, marker="s",
               edgecolors="k", linewidths=0.3, zorder=4)

    ax.set_aspect("equal")
    pad = group_radius * 0.45
    ax.set_xlim(-group_radius - pad, group_radius + pad)
    ax.set_ylim(-group_radius - pad, group_radius + pad)

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               markersize=5, label=f"Compute nodes ({n})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#666",
               markeredgecolor="k", markersize=6,
               label=f"Routers ({total_routers})"),
        Line2D([0], [0], color="#2ca02c", lw=1.5,
               label="Intra-chassis (green)"),
        Line2D([0], [0], color="#333", lw=1.2,
               label="Inter-chassis (black)"),
        Line2D([0], [0], color="#1f77b4", lw=1.8,
               label="Inter-group (blue)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=6,
              framealpha=0.9, edgecolor="#ccc").set_zorder(5)

    ax.set_title(
        f"Dragonfly  |  {n} nodes  |  "
        f"{num_groups}g \u00d7 {num_chassis}c \u00d7 "
        f"{num_routers}r \u00d7 {nodes_per_router}n\n"
        f"Cray Aries 5.25 GBps, 100 ns  "
        f"(Theta XC40 / Piz Daint XC50)",
        fontsize=10, fontweight="bold", pad=10)


# ──────────────────────── BUTTERFLY plotter ─────────────────────────

def plot_butterfly(n, edges, ax):
    dim = int(math.log2(n))

    # ── Circular layout with Gray-code ordering ──
    # Gray code ensures adjacent positions differ in exactly 1 bit,
    # so low-dimension edges become short chords and high-dimension
    # edges become long chords — the classic hypercube visualization.
    gray_code = [i ^ (i >> 1) for i in range(n)]
    pos_index = [0] * n
    for i, g in enumerate(gray_code):
        pos_index[g] = i

    radius = 1.0
    pos = {}
    for node in range(n):
        angle = 2 * math.pi * pos_index[node] / n - math.pi / 2
        pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

    # ── Distinct colors per dimension (tab10 for ≤10 dims) ──
    cmap = plt.cm.tab10
    dim_colors = [cmap(d % 10) for d in range(dim)]

    # ── Draw edges: HIGH dims first (background), LOW dims last (top) ──
    # Low-dim edges are short arcs along the rim (local neighbors),
    # high-dim edges are long chords across the circle (global links).
    # Drawing low dims on top makes the local structure pop visually.
    # Butterfly-specific line width and alpha — the circular layout
    # gives edges more visual room than the dense layered layouts.
    if n <= 128:
        lw_base, al_base = 0.7, 0.65
    elif n <= 256:
        lw_base, al_base = 0.45, 0.50
    elif n <= 512:
        lw_base, al_base = 0.30, 0.40
    else:
        lw_base, al_base = 0.22, 0.30

    for d in reversed(range(dim)):
        segs = [(pos[a], pos[b]) for a, b in edges
                if (a ^ b) == (1 << d)]
        if segs:
            # Low dims: thicker + more opaque; high dims: thinner + fainter
            t = d / max(1, dim - 1)           # 0 = lowest dim, 1 = highest
            lw = lw_base * (1.5 - 0.9 * t)    # low→thick, high→thin
            al = al_base * (1.2 - 0.7 * t)    # low→vivid, high→faint
            al = min(1.0, max(0.10, al))
            ax.add_collection(LineCollection(
                segs, colors=[dim_colors[d]],
                linewidths=lw, alpha=al, zorder=1 + (dim - d)))

    # ── Nodes on top ──
    ns = _node_size(n, base=40)
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=ns, c="#2171b5", edgecolors="white",
               linewidths=0.2, zorder=dim + 2)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")

    # ── Dimension color legend ──
    legend_elems = [
        Line2D([0], [0], color=dim_colors[d], lw=1.8,
               label=f"Dim {d} (stride {1 << d})")
        for d in range(dim)
    ]
    ncol = 2 if dim > 6 else 1
    ax.legend(handles=legend_elems, loc="lower right", fontsize=5.5,
              framealpha=0.92, edgecolor="#ccc", ncol=ncol,
              title="Hypercube dimensions", title_fontsize=6
              ).set_zorder(dim + 5)

    ax.set_title(
        f"Butterfly (Hypercube)  |  {n} nodes  |  "
        f"{dim}D, {len(edges)} links\n"
        f"InfiniBand EDR 12.5 GBps, 100 ns  "
        f"(Kim & Dally, ISCA 2007)",
        fontsize=10, fontweight="bold", pad=10)


# ──────────────────────────── main ──────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <platform.xml> [output.png]",
              file=sys.stderr)
        sys.exit(1)

    xml_path = sys.argv[1]
    topo_type, n, extra = parse_xml(xml_path)

    # Output path
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(xml_path))[0]
        out_path = base + ".png"

    # Figure sizing
    fw, fh = (14, 7) if topo_type == "FAT_TREE" else (10, 10)
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
    ax.set_axis_off()

    if topo_type == "FAT_TREE":
        plot_fattree(n, extra, ax)
    elif topo_type == "DRAGONFLY":
        plot_dragonfly(n, extra, ax)
    elif topo_type == "BUTTERFLY":
        plot_butterfly(n, extra, ax)
    else:
        sys.exit(f"Unknown topology type: {topo_type}")

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}  ({topo_type}, {n} nodes)")


if __name__ == "__main__":
    main()
