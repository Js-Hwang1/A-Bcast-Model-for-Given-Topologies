#!/usr/bin/env python3
"""
visualize_trees.py — Visualize BBS spanning trees on a 2D mesh.

Generates a 6x6 mesh, runs the BBS tree-building algorithm (Python port
of encode.c's bbs_compute_trees), and draws each tree on the mesh grid.

Usage:
    python3 BBS/visualize_trees.py [--rows 6] [--cols 6] [--root 0]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque


# ── Build 2D mesh topology data ──────────────────────────────────────

def build_mesh(rows, cols):
    """Build adjacency, latency, flink, and path data for a 2D mesh.

    Mirrors what topo_preprocess.py produces: Floyd-Warshall over the
    mesh graph, storing latency, first-link, next-hop, and full path
    link arrays — exactly the data that encode.c expects.

    Returns:
        N          : number of nodes
        adj        : NxN bool adjacency (1-hop neighbors)
        lat        : NxN float latency matrix (hop count as proxy)
        flink      : NxN int first-link ID matrix (0 = self/no link)
        path_len   : NxN int number of physical links per path
        path_links : NxN x max_hops int full path link IDs
        max_hops   : int maximum path length
        pos        : dict node -> (col, row) for plotting
        link_id    : dict (a,b) -> unique link ID (1-based, a<b)
    """
    N = rows * cols

    # Assign grid positions: node i = (row i//cols, col i%cols)
    pos = {}
    for i in range(N):
        r, c = divmod(i, cols)
        pos[i] = (c, rows - 1 - r)  # (x, y) with y flipped for display

    # Build edges and assign link IDs
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # right neighbor
            if c + 1 < cols:
                edges.append((node, node + 1))
            # down neighbor
            if r + 1 < rows:
                edges.append((node, (r + 1) * cols + c))

    link_id = {}
    for idx, (a, b) in enumerate(edges):
        lo, hi = min(a, b), max(a, b)
        link_id[(lo, hi)] = idx + 1  # 1-based

    # Adjacency
    adj = np.zeros((N, N), dtype=bool)
    for a, b in edges:
        adj[a, b] = True
        adj[b, a] = True

    # Floyd-Warshall for latency + first-link + next-hop
    INF = float('inf')
    lat = np.full((N, N), INF)
    flink = np.zeros((N, N), dtype=int)
    next_hop = np.full((N, N), -1, dtype=int)
    np.fill_diagonal(lat, 0.0)

    # Direct-edge link IDs for path reconstruction
    direct_links = {}  # (a, b) -> [link_ids]

    for a, b in edges:
        lo, hi = min(a, b), max(a, b)
        lid = link_id[(lo, hi)]
        lat[a, b] = 1.0
        lat[b, a] = 1.0
        flink[a, b] = lid
        flink[b, a] = lid
        next_hop[a, b] = b
        next_hop[b, a] = a
        direct_links[(a, b)] = [lid]
        direct_links[(b, a)] = [lid]

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if lat[i, k] + lat[k, j] < lat[i, j]:
                    lat[i, j] = lat[i, k] + lat[k, j]
                    flink[i, j] = flink[i, k]
                    next_hop[i, j] = next_hop[i, k]

    # Reconstruct full paths (same logic as topo_preprocess.py)
    all_paths = {}
    for ci in range(N):
        for cj in range(N):
            if ci == cj:
                all_paths[(ci, cj)] = []
                continue
            path_lids = []
            current = ci
            seen = set()
            while current != cj and current >= 0:
                if current in seen:
                    break
                seen.add(current)
                nxt = next_hop[current, cj]
                if nxt < 0:
                    break
                key = (current, nxt)
                if key in direct_links:
                    path_lids.extend(direct_links[key])
                current = nxt
            all_paths[(ci, cj)] = path_lids

    max_hops = max(len(p) for p in all_paths.values()) if all_paths else 1
    if max_hops == 0:
        max_hops = 1

    path_len = np.zeros((N, N), dtype=int)
    path_links = np.zeros((N, N, max_hops), dtype=int)
    for (ci, cj), lids in all_paths.items():
        path_len[ci, cj] = len(lids)
        for k, lid in enumerate(lids):
            path_links[ci, cj, k] = lid

    return N, adj, lat, flink, path_len, path_links, max_hops, pos, link_id


# ── BBS tree computation (port of encode.c) ──────────────────────────

def tree_depth(parent, N):
    maxd = 0
    for i in range(N):
        d = 0
        v = i
        while parent[v] >= 0:
            v = parent[v]
            d += 1
            if d > N:
                return N
        maxd = max(maxd, d)
    return maxd


BBS_MAX_TREES = 8


def bbs_compute_trees(N, adj, lat, flink, path_len, path_links, max_hops,
                      root):
    """Exact port of encode.c bbs_compute_trees().

    Returns:
        tau     : number of trees
        trees   : list of parent arrays (each length N, -1 = root)
    """
    # ---- 1-hop threshold ----
    lat_1hop = 1e30
    for i in range(N):
        for j in range(N):
            l = lat[i, j]
            if l > 0.0 and l < lat_1hop:
                lat_1hop = l
    lat_thresh = lat_1hop * 1.01

    # ---- Build 1-hop adjacency matrix ----
    adj_1hop = np.zeros((N, N), dtype=bool)
    total_edges = 0
    for i in range(N):
        for j in range(i + 1, N):
            l = lat[i, j]
            if l > 0.0 and l <= lat_thresh:
                adj_1hop[i, j] = True
                adj_1hop[j, i] = True
                total_edges += 1

    # ---- Root's 1-hop neighbors ----
    root_nbrs = []
    for v in range(N):
        if adj_1hop[root, v]:
            root_nbrs.append(v)
    root_nn = len(root_nbrs)

    # ---- Compute baseline single-tree BFS depth ----
    visited = [False] * N
    queue = [0] * N
    t0_bfs = [-1] * N
    visited[root] = True
    qh, qt = 0, 0
    queue[qt] = root
    qt += 1
    while qh < qt:
        u = queue[qh]
        qh += 1
        for v in range(N):
            if visited[v] or not adj_1hop[u, v]:
                continue
            visited[v] = True
            t0_bfs[v] = u
            queue[qt] = v
            qt += 1
    depth0 = tree_depth(t0_bfs, N)

    # ---- Determine tau from root degree ----
    nw_bound = total_edges // (N - 1) if N > 1 else 0
    tau = root_nn
    if tau > BBS_MAX_TREES:
        tau = BBS_MAX_TREES
    if tau < 1:
        tau = 1

    # ---- Find max flink ID and count distinct root flinks ----
    max_fl = int(np.max(flink)) + 2

    root_distinct_fl = 0
    seen_fl = []
    for i in range(root_nn):
        fl = flink[root, root_nbrs[i]]
        if fl not in seen_fl:
            seen_fl.append(fl)
            root_distinct_fl += 1

    print(f"BBS: edges={total_edges} root_deg={root_nn} NW={nw_bound} "
          f"root_flinks={root_distinct_fl} max_fl={max_fl} "
          f"-> trying tau={tau} (baseline_depth={depth0})")

    if tau < 2:
        print("BBS: tau=1 (root degree < 2)")
        return 1, [t0_bfs]

    # ---- Flink usage tracking ----
    flink_used = [0] * max_fl

    # ---- Build tau trees with flink-aware two-pass BFS ----
    trees = []
    has_path_data = (max_hops > 0 and path_len is not None
                     and path_links is not None)

    for t in range(tau):
        tree = [-1] * N
        visited = [False] * N
        visited[root] = True
        queue = [0] * (N + 1)
        qh, qt = 0, 0
        queue[qt] = root
        qt += 1

        # Seed: root neighbors at indices t, t+tau, t+2*tau, ...
        for i in range(t, root_nn, tau):
            v = root_nbrs[i]
            if not visited[v]:
                visited[v] = True
                tree[v] = root
                queue[qt] = v
                qt += 1

        # Two-pass BFS for full-path disjointness
        qh = 1  # skip root
        if has_path_data:
            # Pass 1: BFS using only path-disjoint edges
            mh = max_hops
            while qh < qt:
                u = queue[qh]
                qh += 1
                for v in range(N):
                    if visited[v] or not adj_1hop[u, v]:
                        continue
                    # Check if ANY link on path(u,v) is already used
                    plen = path_len[u, v]
                    conflict = False
                    for h in range(plen):
                        lid = path_links[u, v, h]
                        if lid > 0 and lid < max_fl and flink_used[lid] > 0:
                            conflict = True
                            break
                    if conflict:
                        continue
                    visited[v] = True
                    tree[v] = u
                    queue[qt] = v
                    qt += 1

            # Pass 2: fill remaining via any adjacency edge
            qi = 0
            while qi < qt:
                u = queue[qi]
                qi += 1
                for v in range(N):
                    if visited[v] or not adj_1hop[u, v]:
                        continue
                    visited[v] = True
                    tree[v] = u
                    queue[qt] = v
                    qt += 1
        else:
            # Fallback: standard BFS (no path data)
            while qh < qt:
                u = queue[qh]
                qh += 1
                for v in range(N):
                    if visited[v] or not adj_1hop[u, v]:
                        continue
                    visited[v] = True
                    tree[v] = u
                    queue[qt] = v
                    qt += 1

        # ---- Bridge disconnected components via latency ----
        while qt < N:
            bv, bu = -1, -1
            best = 1e30
            for v in range(N):
                if visited[v]:
                    continue
                for u in range(N):
                    if not visited[u]:
                        continue
                    l = lat[u, v]
                    if l > 0.0 and l < best:
                        best = l
                        bu = u
                        bv = v
            if bv < 0:
                break
            visited[bv] = True
            tree[bv] = bu
            queue[qt] = bv
            qt += 1

        # ---- Mark ALL links on this tree's edge paths as used ----
        links_fresh, links_shared = 0, 0
        for i in range(N):
            if tree[i] < 0:
                continue
            p = tree[i]
            if has_path_data:
                mh = max_hops
                plen = path_len[p, i]
                for h in range(plen):
                    lid = path_links[p, i, h]
                    if lid > 0 and lid < max_fl:
                        if flink_used[lid] == 0:
                            links_fresh += 1
                        else:
                            links_shared += 1
                        flink_used[lid] += 1
            else:
                fl = flink[p, i]
                if fl > 0 and fl < max_fl:
                    if flink_used[fl] == 0:
                        links_fresh += 1
                    else:
                        links_shared += 1
                    flink_used[fl] += 1

        print(f"BBS:   T{t}: {links_fresh} fresh links, "
              f"{links_shared} shared links")
        trees.append(tree)

    # ---- Compute depths and stats ----
    max_depth = 0
    depth_str = "BBS: tree depths:"
    for t in range(tau):
        d = tree_depth(trees[t], N)
        rc = sum(1 for i in range(N) if trees[t][i] == root)
        depth_str += f" T{t}={d}(rc={rc})"
        if d > max_depth:
            max_depth = d
    print(depth_str)

    # ---- Count flink collisions between tree pairs ----
    flink_collisions = 0
    for a in range(tau):
        for b in range(a + 1, tau):
            ta = trees[a]
            tb = trees[b]
            for i in range(N):
                if ta[i] < 0:
                    continue
                fl_a = flink[ta[i], i]
                if fl_a == 0:
                    continue
                for j in range(N):
                    if tb[j] < 0:
                        continue
                    fl_b = flink[tb[j], j]
                    if fl_a == fl_b:
                        flink_collisions += 1
                        break

    print(f"BBS: tau={tau} max_depth={max_depth} baseline={depth0} "
          f"flink_collisions={flink_collisions}")

    # ---- Fallback: if max depth > 2x baseline, reduce tau ----
    while tau > 1 and max_depth > 2 * depth0:
        tau -= 1
        print(f"BBS: max_depth {max_depth} > 2*baseline {depth0}, "
              f"reducing to tau={tau}")
        max_depth = 0
        for t in range(tau):
            d = tree_depth(trees[t], N)
            if d > max_depth:
                max_depth = d

    if tau == 1:
        trees = [t0_bfs]
        print("BBS: fallback to tau=1 (baseline BFS)")
    else:
        trees = trees[:tau]

    # ---- Verify all trees are spanning ----
    for t in range(tau):
        covered = sum(1 for i in range(N) if trees[t][i] >= 0 or i == root)
        if covered != N:
            print(f"WARNING: Tree {t} covers only {covered}/{N} nodes!")

    return tau, trees


# ── Visualization ─────────────────────────────────────────────────────

# Qualitative colors for trees
TREE_COLORS = [
    '#e6194b',  # red
    '#3cb44b',  # green
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
]


def draw_tree(ax, tree, N, pos, root, color, label, rows, cols,
              draw_mesh=True):
    """Draw a single spanning tree on mesh grid."""
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=13, fontweight='bold')

    # Draw mesh edges (light gray background)
    if draw_mesh:
        for i in range(N):
            for j in range(i + 1, N):
                ri, ci = divmod(i, cols)
                rj, cj = divmod(j, cols)
                if abs(ri - rj) + abs(ci - cj) == 1:
                    xi, yi = pos[i]
                    xj, yj = pos[j]
                    ax.plot([xi, xj], [yi, yj], '-',
                            color='#e0e0e0', linewidth=1.0, zorder=1)

    # Draw tree edges with arrows
    for i in range(N):
        p = tree[i]
        if p < 0:
            continue
        xp, yp = pos[p]
        xi, yi = pos[i]
        dx = xi - xp
        dy = yi - yp
        ax.annotate('', xy=(xi, yi), xytext=(xp, yp),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2.2, shrinkA=6, shrinkB=6),
                    zorder=3)

    # Draw nodes
    for i in range(N):
        x, y = pos[i]
        if i == root:
            ax.plot(x, y, 'o', markersize=14, color=color,
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6)
        else:
            ax.plot(x, y, 'o', markersize=10, color='white',
                    markeredgecolor=color, markeredgewidth=1.5, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=6, color='#333333', zorder=6)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_overlay(ax, trees, N, pos, root, rows, cols):
    """Draw all trees overlaid with different colors + offsets."""
    ax.set_aspect('equal')
    ax.set_title('All Trees Overlaid', fontsize=13, fontweight='bold')
    tau = len(trees)

    # Draw mesh edges
    for i in range(N):
        for j in range(i + 1, N):
            ri, ci = divmod(i, cols)
            rj, cj = divmod(j, cols)
            if abs(ri - rj) + abs(ci - cj) == 1:
                xi, yi = pos[i]
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], '-',
                        color='#e0e0e0', linewidth=1.0, zorder=1)

    # Offset each tree slightly so edges don't overlap
    offsets = []
    for t in range(tau):
        angle = 2 * np.pi * t / tau
        offsets.append((0.06 * np.cos(angle), 0.06 * np.sin(angle)))

    for t, tree in enumerate(trees):
        color = TREE_COLORS[t % len(TREE_COLORS)]
        ox, oy = offsets[t]
        for i in range(N):
            p = tree[i]
            if p < 0:
                continue
            xp, yp = pos[p]
            xi, yi = pos[i]
            ax.annotate('', xy=(xi + ox, yi + oy),
                        xytext=(xp + ox, yp + oy),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, shrinkA=5, shrinkB=5,
                                        alpha=0.8),
                        zorder=3)

    # Draw nodes (no offset)
    for i in range(N):
        x, y = pos[i]
        if i == root:
            ax.plot(x, y, 'o', markersize=14, color='#333333',
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6)
        else:
            ax.plot(x, y, 'o', markersize=10, color='white',
                    markeredgecolor='#666666', markeredgewidth=1.5, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=6, color='#333333', zorder=6)

    # Legend
    handles = [mpatches.Patch(color=TREE_COLORS[t % len(TREE_COLORS)],
                              label=f'Tree {t}')
               for t in range(tau)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_flink_usage(ax, trees, N, flink, root, link_id, pos, rows, cols):
    """Color each mesh edge by how many trees share it."""
    ax.set_aspect('equal')
    ax.set_title('Link Sharing Heatmap', fontsize=13, fontweight='bold')

    # Count how many trees use each physical link
    max_lid = max(link_id.values()) + 1
    usage = [0] * max_lid
    for tree in trees:
        used_in_tree = set()
        for i in range(N):
            p = tree[i]
            if p < 0:
                continue
            lo, hi = min(p, i), max(p, i)
            fl = flink[p, i]
            if fl > 0 and fl not in used_in_tree:
                used_in_tree.add(fl)
        for fl in used_in_tree:
            if fl < max_lid:
                usage[fl] += 1

    tau = len(trees)
    # Draw edges colored by usage count
    for (a, b), lid in link_id.items():
        xa, ya = pos[a]
        xb, yb = pos[b]
        u = usage[lid]
        if u == 0:
            color = '#e0e0e0'
            lw = 1.0
        elif u == 1:
            color = '#3cb44b'  # green = exclusive
            lw = 2.5
        elif u == 2:
            color = '#f58231'  # orange = shared by 2
            lw = 3.0
        else:
            color = '#e6194b'  # red = shared by 3+
            lw = 3.5
        ax.plot([xa, xb], [ya, yb], '-', color=color, linewidth=lw, zorder=2)

    # Draw nodes
    for i in range(N):
        x, y = pos[i]
        if i == root:
            ax.plot(x, y, 'o', markersize=12, color='#333333',
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6)
        else:
            ax.plot(x, y, 'o', markersize=9, color='white',
                    markeredgecolor='#666666', markeredgewidth=1.2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=6, color='#333333', zorder=6)

    # Legend
    handles = [
        mpatches.Patch(color='#e0e0e0', label='Unused'),
        mpatches.Patch(color='#3cb44b', label='1 tree (exclusive)'),
        mpatches.Patch(color='#f58231', label='2 trees (shared)'),
        mpatches.Patch(color='#e6194b', label='3+ trees (contended)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize BBS spanning trees on a 2D mesh')
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--root', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Save figure to file instead of showing')
    args = parser.parse_args()

    rows, cols, root = args.rows, args.cols, args.root
    N = rows * cols
    assert 0 <= root < N, f"root must be in [0, {N})"

    print(f"Building {rows}x{cols} mesh ({N} nodes)...")
    (N, adj, lat, flink, path_len, path_links, max_hops,
     pos, link_id) = build_mesh(rows, cols)

    print(f"Computing BBS trees (root={root})...")
    tau, trees = bbs_compute_trees(N, adj, lat, flink, path_len,
                                   path_links, max_hops, root)

    # Layout: individual trees + overlay + heatmap
    n_panels = tau + 2
    ncols_fig = min(4, n_panels)
    nrows_fig = (n_panels + ncols_fig - 1) // ncols_fig

    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(5 * ncols_fig, 5 * nrows_fig))
    if nrows_fig == 1 and ncols_fig == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Hide extra axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    # Draw individual trees
    for t in range(tau):
        color = TREE_COLORS[t % len(TREE_COLORS)]
        draw_tree(axes[t], trees[t], N, pos, root, color,
                  f'Tree {t} (depth={tree_depth(trees[t], N)})',
                  rows, cols)

    # Overlay
    draw_overlay(axes[tau], trees, N, pos, root, rows, cols)

    # Heatmap
    draw_flink_usage(axes[tau + 1], trees, N, flink, root,
                     link_id, pos, rows, cols)

    fig.suptitle(f'BBS Trees on {rows}x{cols} Mesh  |  root={root}, '
                 f'tau={tau}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
